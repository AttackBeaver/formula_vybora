"""
LTR v3: исправлено кодирование категориальных + честный holdout.

Что исправлено по сравнению с v2:
    * Кодирование категориальных: строим ЕДИНЫЙ словарь на train+val+test+holdout,
      а не отдельно для каждой группы. Иначе коды категорий несовместимы,
      и holdout метрика становится мусором (0.0226 вместо ожидаемых 0.35+).
    * Early stopping — на HONEST holdout (10-20% val periods=2), не на val.
      Иначе модель переобучается под val, который на 90% совпадает с p2-срезом.
    * Гиперпараметры ускорены: lr=0.05, num_boost_round=2000. Обучение
      должно занимать 1-2 минуты вместо 6.
    * Вес p2-среза снижен с 1.5 до 1.0.

Схема:
    Train: p0 ([P1] → нет) + p1 ([P1] → P2) + pmask (cold) + p2 (90% val periods=2 → P3)
    Early stopping: holdout (10-20% val periods=2 → P3)
    Final: применяем к val и test.

Запуск:
    python -m src.run_ltr_v3
"""
from __future__ import annotations

import random
import time
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
    is_cold,
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
WEIGHT_P2 = 1.0

HOLDOUT_FRACTION = 0.20
HOLDOUT_SEED = 42
BASE_SEED = 42

# Ускоренные гиперпараметры — чтобы перебирать эксперименты за минуты, не часы.
FAST_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": BASE_SEED,
}

NUM_BOOST_ROUND = 2000
EARLY_STOPPING = 100


# ---------------------------------------------------------------------------
# Хелперы
# ---------------------------------------------------------------------------
def _n_periods(r: dict[str, Any]) -> int:
    return len(r.get("history") or [])


def _add_weight(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = np.float32(weight)
    return df


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print("LTR v3: + val_solution (p2-срез), честный holdout, единый словарь категорий")
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
    print("\n[1/8] Статистики на train...")
    stats = FeatureStats.fit(train, modules)

    # ==== 2. Train-срезы: p0, p1, pmask ====
    print("\n[2/8] Train-срезы p0, p1, pmask...")

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
    train_df_p0 = build_dataset(
        [s for s in train if any(h.get("period") == 1 for h in s.get("history", []))],
        stats,
        candidates_fn=make_candidates_train_period1(stats),
        target_fn=target_period_1,
        past_periods=[],
    )
    train_df_pmask = build_dataset(
        train_students_p1, stats,
        candidates_fn=make_candidates_train_period2(stats),
        target_fn=target_period_2,
        past_periods=[1],
        mask_profile=True,
    )
    print(f"      p1={len(train_df_p1):,}  p0={len(train_df_p0):,}  "
          f"pmask={len(train_df_pmask):,}")

    # ==== 3. Holdout + p2-срез ====
    print("\n[3/8] Holdout + p2-срез из val periods=2...")

    val_p2 = [s for s in val if _n_periods(s) == 2]
    rng = random.Random(HOLDOUT_SEED)
    shuffled = list(val_p2)
    rng.shuffle(shuffled)
    n_holdout = int(len(shuffled) * HOLDOUT_FRACTION)
    holdout_records = shuffled[:n_holdout]
    p2_records = shuffled[n_holdout:]

    print(f"      p2-срез: {len(p2_records)}  holdout: {len(holdout_records)}")

    p2_df = build_dataset(
        p2_records, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )
    holdout_df = build_dataset(
        holdout_records, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )
    print(f"      p2_df={len(p2_df):,} строк  holdout_df={len(holdout_df):,} строк")

    # ==== 4. Val и test датасеты ====
    print("\n[4/8] Val и test датасеты...")

    val_df = build_dataset(
        val, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )
    test_df = build_dataset(
        test, stats,
        candidates_fn=candidates_available,
        target_fn=None,
        past_periods=None,
    )
    print(f"      val={len(val_df):,}  test={len(test_df):,}")

    # ==== 5. Единое кодирование категориальных ====
    print("\n[5/8] Кодирование категориальных (единый словарь)...")

    train_df_p1 = _add_weight(train_df_p1, WEIGHT_P1)
    train_df_p0 = _add_weight(train_df_p0, WEIGHT_P0)
    train_df_pmask = _add_weight(train_df_pmask, WEIGHT_PMASK)
    p2_df = _add_weight(p2_df, WEIGHT_P2)
    holdout_df = _add_weight(holdout_df, 1.0)
    val_df = _add_weight(val_df, 1.0)
    test_df = _add_weight(test_df, 1.0)

    train_df_p1["student_id"] = "p1__" + train_df_p1["student_id"].astype(str)
    train_df_p0["student_id"] = "p0__" + train_df_p0["student_id"].astype(str)
    train_df_pmask["student_id"] = "pmask__" + train_df_pmask["student_id"].astype(str)
    p2_df["student_id"] = "p2__" + p2_df["student_id"].astype(str)
    holdout_df["student_id"] = "ho__" + holdout_df["student_id"].astype(str)

    train_df = pd.concat(
        [train_df_p1, train_df_p0, train_df_pmask, p2_df],
        ignore_index=True,
    )
    print(f"      Train: {train_df['student_id'].nunique():,} групп, "
          f"{len(train_df):,} строк")

    # Объединяем ВСЕ фреймы, кодируем одной операцией, разбиваем обратно.
    n_train = len(train_df)
    n_ho = len(holdout_df)
    n_val = len(val_df)
    n_test = len(test_df)

    all_df = pd.concat([train_df, holdout_df, val_df, test_df], ignore_index=True)
    all_df, _ = encode_categoricals(all_df, all_df, CATEGORICAL_FEATS)

    train_df = all_df.iloc[:n_train].reset_index(drop=True)
    holdout_df = all_df.iloc[n_train:n_train + n_ho].reset_index(drop=True)
    val_df = all_df.iloc[n_train + n_ho:n_train + n_ho + n_val].reset_index(drop=True)
    test_df = all_df.iloc[n_train + n_ho + n_val:].reset_index(drop=True)

    print(f"      Единый словарь применён ко всем 4 фреймам.")

    # ==== 6. Обучение ====
    print("\n[6/8] Обучение LightGBM Ranker (early stopping на holdout)...")

    X_train, y_train, g_train, w_train, _ = prepare_xygw(train_df)
    X_ho, y_ho, g_ho, w_ho, ho_sorted = prepare_xygw(holdout_df)
    X_val, y_val, g_val, w_val, val_sorted = prepare_xygw(val_df)
    _, _, _, _, test_sorted = prepare_xygw(test_df)

    print(f"      Train: X={X_train.shape}, групп={len(g_train)}")
    print(f"      Holdout: X={X_ho.shape}, групп={len(g_ho)}")

    t_train = time.time()
    model = train_lgbm_ranker(
        X_train, y_train, g_train,
        X_ho, y_ho, g_ho,          # early stopping на holdout, не на val!
        params=FAST_PARAMS,
        num_boost_round=NUM_BOOST_ROUND,
        early_stopping_rounds=EARLY_STOPPING,
        log_every=100,
        sample_weight_train=w_train,
        sample_weight_val=w_ho,
    )
    print(f"      Обучено за {time.time() - t_train:.1f} сек, "
          f"best_iteration = {model.best_iteration}")

    save_model(model, MODELS_DIR / "lgbm_ranker_v3.txt")

    # ==== 7. Оценка ====
    print("\n[7/8] Оценка:")

    scores_ho = predict_scores(model, ho_sorted)
    rec_ho = top_k_per_student(ho_sorted, scores_ho, k=5)
    # Убираем префикс ho__ для сверки с val_gt_raw
    rec_ho_clean = {sid.replace("ho__", ""): v for sid, v in rec_ho.items()}
    # КРИТИЧНО: фильтруем ground truth по holdout_ids.
    # Иначе evaluate_pndcg5 усредняет по всем val (2759 лишних студентов с pred=[]).
    ho_ids = {s["id"] for s in holdout_records}
    gt_ho = {sid: gt for sid, gt in val_gt_raw.items() if sid in ho_ids}
    mean_ho, _ = evaluate_pndcg5(rec_ho_clean, gt_ho, k=5)
    print(f"\n      Holdout (n={len(holdout_records)}): {mean_ho:.4f}   <- ЧЕСТНАЯ ОЦЕНКА")
    
    scores_val = predict_scores(model, val_sorted)
    rec_val = top_k_per_student(val_sorted, scores_val, k=5)
    mean_val, _, _ = _segment_means(rec_val, val_gt_raw, identify_cold_students(val))
    print(f"      Full val (с утечкой): {mean_val:.4f}")

    # ==== 8. Fallback + test ====
    print("\n[8/8] Fallback для cold + test...")
    cold_ids_val = identify_cold_students(val)
    cold_ids_test = identify_cold_students(test)

    pop_weighted, _ = compute_popularity(train)
    pop_by_inst = compute_popularity_by_institute(train)

    val_warm = [s for s in val if not is_cold(s)]
    test_warm = [s for s in test if not is_cold(s)]
    index = build_institute_index(val_warm + test_warm)

    fb_val = build_global_recommendations(val, pop_weighted, cold_ids_val, k=5)
    rec_val_fb = apply_fallback(rec_val, fb_val, k=5)

    scores_test = predict_scores(model, test_sorted)
    rec_test_ltr = top_k_per_student(test_sorted, scores_test, k=5)
    fb_test = build_global_recommendations(test, pop_weighted, cold_ids_test, k=5)
    rec_test_final = apply_fallback(rec_test_ltr, fb_test, k=5)

    val_path = SUBMISSIONS_DIR / "val_submission_ltr_v3.csv"
    test_path = SUBMISSIONS_DIR / "submission_v3.csv"
    write_submission(rec_val_fb, val_path)
    write_submission(rec_test_final, test_path)
    print(f"      Сохранено: {val_path.name}")
    print(f"      Сохранено: {test_path}  ({len(rec_test_final)} строк)")

    print("\n[Feature importance — top-10]")
    imp = feature_importance_report(model, top_n=10)
    for _, row in imp.iterrows():
        print(f"      {row['feature']:<28}  gain = {row['gain']:>12.0f}")

    print(f"\n[OK] Пайплайн завершён за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()