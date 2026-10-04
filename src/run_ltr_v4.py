"""
LTR v4: относительные периоды (pop_last/prev/first) вместо абсолютных.

Отличие от v3:
    * Используется FEATURE_COLS_V4 (31 признак): pop_period_1/2 заменены
      на pop_last_period / pop_prev_period / pop_first_period. Эти фичи
      означают одно и то же для train и test → нет сдвига распределения
      для 62.7% test-студентов с историей до P3/P4.
    * FeatureStats.add_val_solution_popularity(val_solution) заполняет
      pop_by_period[3] — это позволяет фиче pop_last_period быть ненулевой
      для студентов с историей до P3.

Состав:
    Train: p0 + p1 + pmask + p2 (90% val periods=2 → target=P3)
    Early stopping: holdout (10-20% val periods=2, отфильтрованный)
    Final: val + test.

Запуск:
    python -m src.run_ltr_v4
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
    FEATURE_COLS_V4,
    encode_categoricals,
    feature_importance_report,
    predict_scores,
    prepare_xygw,
    save_model,
    top_k_per_student,
    train_lgbm_ranker,
)
from src.metric import evaluate_pndcg5
from src.popularity import compute_popularity
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

FAST_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": BASE_SEED,
}

NUM_BOOST_ROUND = 2500
EARLY_STOPPING = 100


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


def main() -> None:
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print("LTR v4: относительные периоды (pop_last/prev/first)")
    print("=" * 78)

    t0 = time.time()
    data = load_all()
    train = data["train"]
    val = data["val"]
    test = data["test"]
    modules = data["modules"]
    val_solution = data["val_solution"]
    val_gt_raw = dict(zip(val_solution["id"], val_solution["ground_truth"]))

    # ==== 1. Статистики ====
    print("\n[1/9] Статистики на train + val_solution...")
    stats = FeatureStats.fit(train, modules)
    print(f"      pop_by_period до: {sorted(stats.pop_by_period.keys())}")
    stats.add_val_solution_popularity(val_solution)
    print(f"      pop_by_period после: {sorted(stats.pop_by_period.keys())}")

    # ==== 2. Train-срезы ====
    print("\n[2/9] Train-срезы p0, p1, pmask...")

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
    print("\n[3/9] Holdout + p2-срез из val periods=2...")

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

    # ==== 4. Val и test ====
    print("\n[4/9] Val и test датасеты...")

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

    # ==== 5. Сборка + кодирование ====
    print("\n[5/9] Сборка + единый словарь категорий...")

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

    # ==== 6. Обучение ====
    print("\n[6/9] Обучение LightGBM Ranker (FEATURE_COLS_V4)...")

    X_train, y_train, g_train, w_train, _ = prepare_xygw(
        train_df, feature_cols=FEATURE_COLS_V4,
    )
    X_ho, y_ho, g_ho, w_ho, ho_sorted = prepare_xygw(
        holdout_df, feature_cols=FEATURE_COLS_V4,
    )
    X_val, y_val, g_val, w_val, val_sorted = prepare_xygw(
        val_df, feature_cols=FEATURE_COLS_V4,
    )
    _, _, _, _, test_sorted = prepare_xygw(
        test_df, feature_cols=FEATURE_COLS_V4,
    )

    print(f"      Train: X={X_train.shape}, групп={len(g_train)}, "
          f"w_mean={w_train.mean():.3f}")
    print(f"      Holdout: X={X_ho.shape}, групп={len(g_ho)}")

    t_train = time.time()
    model = train_lgbm_ranker(
        X_train, y_train, g_train,
        X_ho, y_ho, g_ho,
        params=FAST_PARAMS,
        num_boost_round=NUM_BOOST_ROUND,
        early_stopping_rounds=EARLY_STOPPING,
        log_every=100,
        sample_weight_train=w_train,
        sample_weight_val=w_ho,
    )
    print(f"      Обучено за {time.time() - t_train:.1f} сек, "
          f"best_iteration = {model.best_iteration}")

    save_model(model, MODELS_DIR / "lgbm_ranker_v4.txt")

    # ==== 7. Оценка ====
    print("\n[7/9] Оценка:")

    scores_ho = predict_scores(model, ho_sorted, feature_cols=FEATURE_COLS_V4)
    rec_ho = top_k_per_student(ho_sorted, scores_ho, k=5)
    rec_ho_clean = {sid.replace("ho__", ""): v for sid, v in rec_ho.items()}
    ho_ids = {s["id"] for s in holdout_records}
    gt_ho = {sid: gt for sid, gt in val_gt_raw.items() if sid in ho_ids}
    mean_ho, _ = evaluate_pndcg5(rec_ho_clean, gt_ho, k=5)
    print(f"\n      Holdout (n={len(holdout_records)}): {mean_ho:.4f}   "
          f"<- ЧЕСТНАЯ ОЦЕНКА")

    scores_val = predict_scores(model, val_sorted, feature_cols=FEATURE_COLS_V4)
    rec_val_ltr = top_k_per_student(val_sorted, scores_val, k=5)
    mean_val_ltr, cold_ltr, warm_ltr = _segment_means(
        rec_val_ltr, val_gt_raw, identify_cold_students(val),
    )
    print(f"      Full val (LTR raw): {mean_val_ltr:.4f}   "
          f"cold={cold_ltr:.4f}  warm={warm_ltr:.4f}")

    # ==== 8. Fallback для cold на val ====
    print("\n[8/9] Cold-start fallback на val...")
    cold_ids_val = identify_cold_students(val)
    pop_weighted, _ = compute_popularity(train)
    fb_val = build_global_recommendations(val, pop_weighted, cold_ids_val, k=5)
    rec_val_fb = apply_fallback(rec_val_ltr, fb_val, k=5)

    mean_val, cold_val, warm_val = _segment_means(
        rec_val_fb, val_gt_raw, cold_ids_val,
    )
    est_val = TEST_COLD_SHARE * cold_val + (1 - TEST_COLD_SHARE) * warm_val
    print(f"      Full val (+ fallback): mean={mean_val:.4f}  "
          f"cold={cold_val:.4f}  warm={warm_val:.4f}")
    print(f"      test_estimate (cold_share={TEST_COLD_SHARE:.3f}): {est_val:.4f}")

    val_path = SUBMISSIONS_DIR / "val_submission_ltr_v4.csv"
    write_submission(rec_val_fb, val_path)
    print(f"      Сохранено: {val_path.name}")

    # ==== 9. Test ====
    print("\n[9/9] Инференс test + финальный submission...")
    scores_test = predict_scores(model, test_sorted, feature_cols=FEATURE_COLS_V4)
    rec_test_ltr = top_k_per_student(test_sorted, scores_test, k=5)
    cold_ids_test = identify_cold_students(test)
    fb_test = build_global_recommendations(test, pop_weighted, cold_ids_test, k=5)
    rec_test_final = apply_fallback(rec_test_ltr, fb_test, k=5)

    test_path = SUBMISSIONS_DIR / "submission_v4.csv"
    write_submission(rec_test_final, test_path)
    print(f"      Cold-студентов в test: {len(cold_ids_test)} "
          f"({100 * len(cold_ids_test) / len(test):.1f}%)")
    print(f"      Сохранено: {test_path}  ({len(rec_test_final)} строк)")

    # ==== Feature importance ====
    print("\n[Feature importance — top-12 по gain]")
    imp = feature_importance_report(model, feature_cols=FEATURE_COLS_V4, top_n=12)
    for _, row in imp.iterrows():
        print(f"      {row['feature']:<28}  gain = {row['gain']:>12.0f}")

    print(f"\n[OK] Пайплайн завершён за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()