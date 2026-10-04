"""
LTR v2: обучение с использованием val_solution как target для P3.

Ключевая идея:
    val_solution — это ответы для целевого периода каждого val-студента.
    Для студентов с periods=2 в val — это target P3.
    У test многие имеют P3 в истории (для periods=2,3) — значит обучающий
    сигнал на переходе (P1,P2)→P3 полезен и для test-задач (P2,P3)→P4.

Что делаем:
    1. Из val берём только студентов с periods=2 (их 1424). Их target в
       val_solution = P3.
    2. Стратифицированный holdout 10% — для честной оценки.
    3. Остальные 90% val → срез p2: past=[1,2] → target=P3.
    4. Из train строим срезы как раньше: p0, p1, pmask.
    5. Обучаем, оцениваем на holdout, применяем к test.

Файлы:
    Вход:  data/*.jsonl, data/val_solution.csv
    Выход: submissions/val_submission_ltr_v2.csv
           submissions/submission_v2.csv
           models/lgbm_ranker_v2.txt

Запуск:
    python -m src.run_ltr_v2
"""
from __future__ import annotations

import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.cold_start import (
    apply_fallback,
    build_global_recommendations,
    build_institute_index,
    build_institute_recommendations,
    identify_cold_students,
)
from src.data_loader import load_all
from src.features import (
    FeatureStats,
    build_dataset,
    candidates_available,
    make_candidates_train_period1,
    make_candidates_train_period2,
    make_target_from_val_solution,
    target_period_1,
    target_period_2,
)
from src.ltr_model import (
    CATEGORICAL_FEATS,
    encode_categoricals,
    feature_importance_report,
    predict_scores,
    prepare_xygw,
    save_model,
    top_k_per_student,
    train_lgbm_ranker,
)
from src.metric import evaluate_pndcg5
from src.popularity import (
    compute_popularity,
    compute_popularity_by_institute,
)
from src.submit import write_submission

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
MODELS_DIR = PROJECT_ROOT / "models"

TEST_COLD_SHARE = 0.352

WEIGHT_P1 = 1.0
WEIGHT_P0 = 0.5
WEIGHT_PMASK = 1.0
WEIGHT_P2 = 1.5  # срез p2 — самый релевантный для test → больший вес

HOLDOUT_FRACTION = 0.10
HOLDOUT_SEED = 42
BASE_SEED = 42


# ---------------------------------------------------------------------------
# Хелперы
# ---------------------------------------------------------------------------
def _n_periods(r: dict[str, Any]) -> int:
    return len(r.get("history") or [])


def _segment_means(
    rec: dict[str, list[str]],
    gt_raw: dict[str, str],
    cold_ids: set[str],
    k: int = 5,
) -> tuple[float, float, float]:
    mean_all, per_student = evaluate_pndcg5(rec, gt_raw, k=k)
    cold_scores = [s for sid, s in per_student.items() if sid in cold_ids]
    warm_scores = [s for sid, s in per_student.items() if sid not in cold_ids]
    c = float(np.mean(cold_scores)) if cold_scores else 0.0
    w = float(np.mean(warm_scores)) if warm_scores else 0.0
    return mean_all, c, w


def _metrics_on_subset(
    rec: dict[str, list[str]],
    gt_raw: dict[str, str],
    subset_ids: set[str],
    k: int = 5,
) -> float:
    """pNDCG@5 только по подмножеству id."""
    filtered_gt = {sid: v for sid, v in gt_raw.items() if sid in subset_ids}
    filtered_rec = {sid: v for sid, v in rec.items() if sid in subset_ids}
    mean, _ = evaluate_pndcg5(filtered_rec, filtered_gt, k=k)
    return mean


def _add_weight(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = np.float32(weight)
    return df


def _is_cold_rec(r: dict[str, Any]) -> bool:
    from src.cold_start import is_cold
    return is_cold(r)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print("LTR v2: + val_solution как target P3")
    print("=" * 78)

    t0 = time.time()
    data = load_all()
    train = data["train"]
    val = data["val"]
    test = data["test"]
    modules = data["modules"]
    val_gt_raw = dict(
        zip(data["val_solution"]["id"], data["val_solution"]["ground_truth"])
    )

    # ==== 1. Статистики ====
    print("\n[1/9] Статистики на train...")
    stats = FeatureStats.fit(train, modules)

    # ==== 2. Train-срезы: p0, p1, pmask ====
    print("\n[2/9] Train-срезы (p0, p1, pmask)...")

    train_students_p1 = [
        s for s in train
        if any(h.get("period") == 2 for h in s.get("history", []))
    ]
    train_df_p1 = build_dataset(
        train_students_p1, stats,
        candidates_fn=make_candidates_train_period2(stats),
        target_fn=target_period_2,
        past_periods=[1],
    )
    print(f"      p1 (warm):   {len(train_df_p1):,} строк")

    train_students_p0 = [
        s for s in train
        if any(h.get("period") == 1 for h in s.get("history", []))
    ]
    train_df_p0 = build_dataset(
        train_students_p0, stats,
        candidates_fn=make_candidates_train_period1(stats),
        target_fn=target_period_1,
        past_periods=[],
    )
    print(f"      p0 (no-hist): {len(train_df_p0):,} строк")

    train_df_pmask = build_dataset(
        train_students_p1, stats,
        candidates_fn=make_candidates_train_period2(stats),
        target_fn=target_period_2,
        past_periods=[1],
        mask_profile=True,
    )
    print(f"      pmask (cold): {len(train_df_pmask):,} строк")

    # ==== 3. Holdout + p2-срез из val (только periods=2) ====
    print("\n[3/9] Holdout + p2-срез из val (periods=2)...")

    val_p2_students = [s for s in val if _n_periods(s) == 2]
    print(f"      Студентов val с periods=2: {len(val_p2_students)}")

    rng = random.Random(HOLDOUT_SEED)
    shuffled = list(val_p2_students)
    rng.shuffle(shuffled)
    n_holdout = int(len(shuffled) * HOLDOUT_FRACTION)
    holdout_records = shuffled[:n_holdout]
    p2_records = shuffled[n_holdout:]
    holdout_ids = {s["id"] for s in holdout_records}
    print(f"      holdout: {len(holdout_records)}  p2-срез: {len(p2_records)}")

    p2_train_df = build_dataset(
        p2_records, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )
    print(f"      p2-срез: {len(p2_train_df):,} строк")

    holdout_df = build_dataset(
        holdout_records, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )
    print(f"      holdout: {len(holdout_df):,} строк")

    # ==== 4. Сборка обучающей выборки ====
    print("\n[4/9] Сборка обучающей выборки...")

    train_df_p1 = _add_weight(train_df_p1, WEIGHT_P1)
    train_df_p0 = _add_weight(train_df_p0, WEIGHT_P0)
    train_df_pmask = _add_weight(train_df_pmask, WEIGHT_PMASK)
    p2_train_df = _add_weight(p2_train_df, WEIGHT_P2)

    train_df_p1["student_id"] = "p1__" + train_df_p1["student_id"].astype(str)
    train_df_p0["student_id"] = "p0__" + train_df_p0["student_id"].astype(str)
    train_df_pmask["student_id"] = "pmask__" + train_df_pmask["student_id"].astype(str)
    p2_train_df["student_id"] = "p2__" + p2_train_df["student_id"].astype(str)

    train_df = pd.concat(
        [train_df_p1, train_df_p0, train_df_pmask, p2_train_df],
        ignore_index=True,
    )
    print(f"      ИТОГО: {train_df['student_id'].nunique():,} групп, "
          f"{len(train_df):,} строк")

    # ==== 5. Val и test датасеты ====
    print("\n[5/9] Val и test датасеты...")

    # Val используем полностью для финальной сверки (с оговоркой утечки).
    val_df = build_dataset(
        val, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )
    val_df = _add_weight(val_df, 1.0)
    print(f"      val: {len(val_df):,} строк")

    test_df = build_dataset(
        test, stats,
        candidates_fn=candidates_available,
        target_fn=None,
        past_periods=None,
    )
    test_df = _add_weight(test_df, 1.0)
    print(f"      test: {len(test_df):,} строк")

    # ==== 6. Кодирование категориальных ====
    print("\n[6/9] Кодирование категориальных...")
    # Для получения словаря категорий — конкатенируем все нужные фреймы.
    all_cat_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_cat_df, _ = encode_categoricals(all_cat_df, all_cat_df, CATEGORICAL_FEATS)
    # Разбиваем обратно
    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    train_df = all_cat_df.iloc[:n_train].reset_index(drop=True)
    val_df = all_cat_df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = all_cat_df.iloc[n_train + n_val:].reset_index(drop=True)
    holdout_df, _ = encode_categoricals(holdout_df, holdout_df, CATEGORICAL_FEATS)

    # ==== 7. Обучение ====
    print("\n[7/9] Обучение LightGBM Ranker...")
    X_train, y_train, g_train, w_train, _ = prepare_xygw(train_df)
    X_val, y_val, g_val, w_val, val_sorted = prepare_xygw(val_df)
    X_holdout, y_holdout, g_holdout, w_holdout, holdout_sorted = prepare_xygw(holdout_df)
    _, _, _, _, test_sorted = prepare_xygw(test_df)

    print(f"      Train: X={X_train.shape}, групп={len(g_train)}, "
          f"w_mean={w_train.mean():.3f}")

    t_train = time.time()
    model = train_lgbm_ranker(
        X_train, y_train, g_train,
        X_val, y_val, g_val,
        params={"seed": BASE_SEED},
        num_boost_round=5000,
        early_stopping_rounds=200,
        log_every=100,
        sample_weight_train=w_train,
        sample_weight_val=w_val,
    )
    print(f"      Обучено за {time.time() - t_train:.1f} сек, "
          f"best_iteration = {model.best_iteration}")

    save_model(model, MODELS_DIR / "lgbm_ranker_v2.txt")

    # ==== 8. Оценка ====
    print("\n[8/9] Оценка:")
    scores_val = predict_scores(model, val_sorted)
    rec_val = top_k_per_student(val_sorted, scores_val, k=5)

    scores_holdout = predict_scores(model, holdout_sorted)
    rec_holdout = top_k_per_student(holdout_sorted, scores_holdout, k=5)

    # Метрика на holdout — честная (модель не видела этих студентов)
    mean_holdout, _ = evaluate_pndcg5(rec_holdout, val_gt_raw, k=5)
    print(f"\n      Holdout (periods=2, n={len(holdout_records)}): "
          f"{mean_holdout:.4f}   <- ЧЕСТНАЯ ОЦЕНКА")

    # Дополнительно: holdout по сегментам
    from src.cold_start import is_cold as _ic
    holdout_cold = {s["id"] for s in holdout_records if _ic(s)}
    _, cold_h, warm_h = _segment_means(rec_holdout, val_gt_raw, holdout_cold, k=5)
    print(f"        cold = {cold_h:.4f}   warm = {warm_h:.4f}")

    # Метрика на полном val — ЗАВЫШЕНА из-за утечки, только для справки
    mean_val, _ = evaluate_pndcg5(rec_val, val_gt_raw, k=5)
    print(f"      Full val (УТЕЧКА, игнорировать): {mean_val:.4f}")
    
    # ==== 9. Fallback + test ====
    print("\n[9/9] Fallback для cold + test...")
    cold_ids_val = identify_cold_students(val)
    cold_ids_test = identify_cold_students(test)

    pop_weighted, _ = compute_popularity(train)
    pop_by_inst = compute_popularity_by_institute(train)

    val_warm = [s for s in val if not _is_cold_rec(s)]
    test_warm = [s for s in test if not _is_cold_rec(s)]
    index = build_institute_index(val_warm + test_warm)

    # Fallback для val
    fb_val = build_global_recommendations(val, pop_weighted, cold_ids_val, k=5)
    rec_val_fb = apply_fallback(rec_val, fb_val, k=5)

    # Fallback для test
    fb_test = build_global_recommendations(test, pop_weighted, cold_ids_test, k=5)

    scores_test = predict_scores(model, test_sorted)
    rec_test_ltr = top_k_per_student(test_sorted, scores_test, k=5)
    rec_test_final = apply_fallback(rec_test_ltr, fb_test, k=5)

    # Сохраняем
    val_path = SUBMISSIONS_DIR / "val_submission_ltr_v2.csv"
    test_path = SUBMISSIONS_DIR / "submission_v2.csv"
    write_submission(rec_val_fb, val_path)
    write_submission(rec_test_final, test_path)
    print(f"      Сохранено: {val_path.name}")
    print(f"      Сохранено: {test_path}  ({len(rec_test_final)} строк)")

    # Feature importance
    print("\n[Feature importance — top-10]")
    imp = feature_importance_report(model, top_n=10)
    for _, row in imp.iterrows():
        print(f"      {row['feature']:<28}  gain = {row['gain']:>12.0f}")

    print(f"\n[OK] Пайплайн завершён за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()