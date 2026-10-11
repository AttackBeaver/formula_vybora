"""
LTR v22: v21 с умеренным весом val_p1 + правильный test_estimate.

Правки относительно v21:
    1. WEIGHT_VALP1 = 1.0 (было 2.5). Меньше разбавления p2-сигнала.
    2. test_estimate считается ТОЛЬКО по чистым студентам (holdout_ids),
       не включая p2_records (которые были в обучении с w=3.0).
    3. Диагностика: печатаем отдельно валидацию на:
         - holdout_warm (284, чистый)
         - cold_val (88, чистый)
         - p2_clean (1140, ГРЯЗНЫЙ — тренировались, только для справки)
         - val_p1 (1531, ГРЯЗНЫЙ — тренировались, только для справки)

Остальное — как v21:
    - Срезы p0 (w=0.5), p1 (w=1.0), pmask (w=1.0), p2 (w=3.0), valp1 (w=1.0).
    - FEATURE_COLS_V4 (31).
    - max_history_periods=2 для test.
    - truncation_level=15.
    - multi-seed=5.

Запуск:
    python -m src.run_ltr_v22 --n-seeds 5
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.cold_start import identify_cold_students
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
from src.submit import write_submission

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
MODELS_DIR = PROJECT_ROOT / "models"

TEST_COLD_SHARE = 0.352
WEIGHT_P1 = 1.0
WEIGHT_P0 = 0.5
WEIGHT_PMASK = 1.0
WEIGHT_P2 = 3.0
WEIGHT_VALP1 = 1.0   # <-- главное отличие от v21

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
    "lambdarank_truncation_level": 15,
}

NUM_BOOST_ROUND = 2500
EARLY_STOPPING = 100

V7_HOLDOUT = 0.5134
V7_COLD_VAL = 0.2474


def _n_periods(r: dict[str, Any]) -> int:
    return len(r.get("history") or [])


def _add_weight(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = np.float32(weight)
    return df


def _mean_for_ids(
    rec: dict[str, list[str]],
    gt_raw: dict[str, str],
    ids: set[str],
    k: int = 5,
) -> float:
    _, per_student = evaluate_pndcg5(rec, gt_raw, k=k)
    vals = [s for sid, s in per_student.items() if sid in ids]
    return float(np.mean(vals)) if vals else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="LTR v22: v21 с WEIGHT_VALP1=1.0")
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()
    n_seeds = max(1, int(args.n_seeds))
    seeds = [BASE_SEED + i for i in range(n_seeds)]

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print(f"LTR v22: WEIGHT_VALP1={WEIGHT_VALP1}, n_seeds={n_seeds}")
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
    print("\n[1/9] Статистики...")
    stats = FeatureStats.fit(train, modules)
    stats.add_val_solution_popularity(val_solution)

    # ==== 2. Train-срезы ====
    print("\n[2/9] Train-срезы (p0/p1/pmask)...")
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

    # ==== 3. Holdout + p2 из val_p2 ====
    print("\n[3/9] Holdout + p2 (из val_p2)...")
    val_p2 = [s for s in val if _n_periods(s) == 2]
    rng = random.Random(HOLDOUT_SEED)
    shuffled = list(val_p2)
    rng.shuffle(shuffled)
    n_holdout = int(len(shuffled) * HOLDOUT_FRACTION)
    holdout_records = shuffled[:n_holdout]
    p2_records = shuffled[n_holdout:]
    holdout_ids = {s["id"] for s in holdout_records}
    p2_ids = {s["id"] for s in p2_records}
    print(f"      p2={len(p2_records)}  holdout={len(holdout_records)}")

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

    # ==== 4. val_p1 срез ====
    print("\n[4/9] val_p1 срез (warm-val P1)...")
    val_p1_records = [s for s in val if _n_periods(s) == 1]
    val_p1_ids = {s["id"] for s in val_p1_records}
    print(f"      val_p1={len(val_p1_records)} студентов")

    valp1_df = build_dataset(
        val_p1_records, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1],
    )
    print(f"      valp1_df={len(valp1_df):,} строк")

    # ==== 5. Val и test ====
    print("\n[5/9] Val и test...")
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
        max_history_periods=2,
    )
    print(f"      val={len(val_df):,}  test={len(test_df):,}")

    # ==== 6. Веса, префиксы, категориальные ====
    print("\n[6/9] Веса, префиксы, категориальные...")
    train_df_p1 = _add_weight(train_df_p1, WEIGHT_P1)
    train_df_p0 = _add_weight(train_df_p0, WEIGHT_P0)
    train_df_pmask = _add_weight(train_df_pmask, WEIGHT_PMASK)
    p2_df = _add_weight(p2_df, WEIGHT_P2)
    valp1_df = _add_weight(valp1_df, WEIGHT_VALP1)
    holdout_df = _add_weight(holdout_df, 1.0)
    val_df = _add_weight(val_df, 1.0)
    test_df = _add_weight(test_df, 1.0)

    train_df_p1["student_id"] = "p1__" + train_df_p1["student_id"].astype(str)
    train_df_p0["student_id"] = "p0__" + train_df_p0["student_id"].astype(str)
    train_df_pmask["student_id"] = "pmask__" + train_df_pmask["student_id"].astype(str)
    p2_df["student_id"] = "p2__" + p2_df["student_id"].astype(str)
    valp1_df["student_id"] = "vp1__" + valp1_df["student_id"].astype(str)
    holdout_df["student_id"] = "ho__" + holdout_df["student_id"].astype(str)

    train_df = pd.concat(
        [train_df_p1, train_df_p0, train_df_pmask, p2_df, valp1_df],
        ignore_index=True,
    )
    print(f"      train: 5 срезов, всего {len(train_df):,} строк")

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

    # ==== 7. Обучение ====
    print(f"\n[7/9] Обучение {n_seeds} моделей...")

    X_train, y_train, g_train, w_train, _ = prepare_xygw(
        train_df, feature_cols=FEATURE_COLS_V4,
    )
    X_ho, y_ho, g_ho, w_ho, ho_sorted = prepare_xygw(
        holdout_df, feature_cols=FEATURE_COLS_V4,
    )
    _, _, _, _, val_sorted = prepare_xygw(
        val_df, feature_cols=FEATURE_COLS_V4,
    )
    _, _, _, _, test_sorted = prepare_xygw(
        test_df, feature_cols=FEATURE_COLS_V4,
    )

    print(f"      Train: X={X_train.shape}, групп={len(g_train)}")

    scores_ho_list: list[np.ndarray] = []
    scores_val_list: list[np.ndarray] = []
    scores_test_list: list[np.ndarray] = []
    best_iters: list[int] = []

    for i, seed in enumerate(seeds):
        params = {**FAST_PARAMS, "seed": seed}
        t_train = time.time()
        model = train_lgbm_ranker(
            X_train, y_train, g_train,
            X_ho, y_ho, g_ho,
            params=params,
            num_boost_round=NUM_BOOST_ROUND,
            early_stopping_rounds=EARLY_STOPPING,
            log_every=500,
            sample_weight_train=w_train,
            sample_weight_val=w_ho,
        )
        best_iters.append(model.best_iteration)
        print(f"      seed={seed}: best_iter={model.best_iteration}, "
              f"{time.time() - t_train:.1f} сек")
        scores_ho_list.append(predict_scores(model, ho_sorted, feature_cols=FEATURE_COLS_V4))
        scores_val_list.append(predict_scores(model, val_sorted, feature_cols=FEATURE_COLS_V4))
        scores_test_list.append(predict_scores(model, test_sorted, feature_cols=FEATURE_COLS_V4))
        last_model = model

    scores_ho = np.mean(scores_ho_list, axis=0)
    scores_val = np.mean(scores_val_list, axis=0)
    scores_test = np.mean(scores_test_list, axis=0)
    print(f"\n      best_iter: mean={np.mean(best_iters):.0f}, "
          f"min={min(best_iters)}, max={max(best_iters)}")

    save_model(last_model, MODELS_DIR / "lgbm_ranker_v22.txt")

    # ==== 8. Оценка ====
    print("\n[8/9] Оценка:")

    rec_ho = top_k_per_student(ho_sorted, scores_ho, k=5)
    rec_ho_clean = {sid.replace("ho__", ""): v for sid, v in rec_ho.items()}
    gt_ho = {sid: gt for sid, gt in val_gt_raw.items() if sid in holdout_ids}
    mean_ho, _ = evaluate_pndcg5(rec_ho_clean, gt_ho, k=5)

    print(f"      Holdout (n=284, чистый):  {mean_ho:.4f}   "
          f"(v7 = {V7_HOLDOUT:.4f}, Δ = {mean_ho - V7_HOLDOUT:+.4f})")

    cold_ids_val = identify_cold_students(val)
    rec_val = top_k_per_student(val_sorted, scores_val, k=5)

    m_val_cold = _mean_for_ids(rec_val, val_gt_raw, cold_ids_val)
    m_val_hold = _mean_for_ids(rec_val, val_gt_raw, holdout_ids)
    m_val_p2 = _mean_for_ids(rec_val, val_gt_raw, p2_ids)     # ГРЯЗНЫЙ
    m_val_p1 = _mean_for_ids(rec_val, val_gt_raw, val_p1_ids) # ГРЯЗНЫЙ

    print(f"\n      Val разбивка:")
    print(f"          cold           (n={len(cold_ids_val)}):       {m_val_cold:.4f}   "
          f"(v7 = {V7_COLD_VAL:.4f}, Δ = {m_val_cold - V7_COLD_VAL:+.4f})")
    print(f"          holdout_warm   (n={len(holdout_ids)}, чистый): {m_val_hold:.4f}")
    print(f"          p2 (n={len(p2_ids)}, ГРЯЗНЫЙ):              {m_val_p2:.4f}")
    print(f"          val_p1 (n={len(val_p1_ids)}, ГРЯЗНЫЙ):         {m_val_p1:.4f}")

    # ЧИСТЫЙ test_estimate: только holdout_warm + cold_val
    est_val_clean = TEST_COLD_SHARE * m_val_cold + (1 - TEST_COLD_SHARE) * m_val_hold
    print(f"\n      test_estimate (ЧИСТЫЙ: cold+holdout) = {est_val_clean:.4f}")
    print(f"      (для сравнения: v7 ≈ 0.4017, v21 (мой ошибочный расчёт) ≈ 0.4758)")

    val_path = SUBMISSIONS_DIR / "val_submission_ltr_v22.csv"
    write_submission(rec_val, val_path)
    print(f"      Сохранено: {val_path}")

    # ==== 9. Test ====
    print("\n[9/9] Test submission...")
    rec_test = top_k_per_student(test_sorted, scores_test, k=5)
    test_path = SUBMISSIONS_DIR / "submission_v22.csv"
    write_submission(rec_test, test_path)
    cold_test = identify_cold_students(test)
    print(f"      Cold в test: {len(cold_test)}  Сохранено: {test_path}")

    print("\n[Feature importance — top-15]")
    imp = feature_importance_report(last_model, feature_cols=FEATURE_COLS_V4, top_n=15)
    for _, row in imp.iterrows():
        print(f"      {row['feature']:<32}  gain = {row['gain']:>12.0f}")

    print(f"\n[OK] Пайплайн завершён за {time.time() - t0:.1f} сек.")

    print("\n" + "=" * 78)
    print("ИНТЕРПРЕТАЦИЯ (для решения о сабмите):")
    print("=" * 78)
    print(f"  holdout    = {mean_ho:.4f}   (v7 = {V7_HOLDOUT:.4f})")
    print(f"  cold_val   = {m_val_cold:.4f}   (v7 = {V7_COLD_VAL:.4f})")
    print(f"  test_est   = {est_val_clean:.4f}   (v7 ≈ 0.4017)")
    print()
    if mean_ho >= 0.510 and m_val_cold >= 0.27:
        print("[+] Оба условия выполнены:")
        print("    - holdout не провалился (≥ 0.510)")
        print("    - cold вырос (≥ 0.27)")
        print("    -> Кандидат на сабмит.")
    elif mean_ho >= 0.510:
        print("[~] holdout ок, но cold слабый. Вряд ли даст больше v7.")
    else:
        print("[-] holdout провалился. Откат к v7.")


if __name__ == "__main__":
    main()