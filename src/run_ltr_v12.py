"""
LTR v12: глобальные ранги модулей внутри available_modules.

Отличие от v7 (champion):
    + Три новых признака:
        pop_rank_global_in_available      — ранг по pop_weighted_global
        pop_rank_last_period_in_available — ранг по pop_last_period
        pop_rank_first_period_in_available — ранг по pop_first_period
      Считаются как ранжирование модуля среди кандидатов КОНКРЕТНОГО студента.
      Устраняют провал v11: для cold-студентов поп_rank_in_available
      вырождался (институт = UNKNOWN), теперь есть глобальная альтернатива.

    Всё остальное (срезы, веса, multi-seed=5, truncation=15,
    max_history_periods=2) — идентично v7.

Fallback не применяется (v7 уже показал: LTR > fallback на cold).

Запуск:
    python -m src.run_ltr_v12
    python -m src.run_ltr_v12 --n-seeds 3
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
# НОВОЕ: ранги по популярности внутри available_modules КОНКРЕТНОГО студента
# ---------------------------------------------------------------------------
def add_global_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавляет три признака-ранга в df, сгруппированные по student_id.

    Ранг 0 = самый популярный модуль среди кандидатов студента.
    Для cold-студентов признаки pop_last/prev/first_period = 0 (истории нет),
    поэтому их ранги будут равны 0 для всех кандидатов — модель увидит
    константу и просто проигнорирует. А pop_weighted_global есть ВСЕГДА,
    и именно он даст сигнал на cold.
    """
    df = df.copy()
    specs = [
        ("pop_rank_global_in_available", "pop_weighted_global"),
        ("pop_rank_last_period_in_available", "pop_last_period"),
        ("pop_rank_first_period_in_available", "pop_first_period"),
    ]
    for new_col, src_col in specs:
        if src_col not in df.columns:
            raise KeyError(f"Ожидался столбец {src_col!r}, но его нет в df")
        # method='min' — детерминированные ties, ascending=False → 0 = топ.
        rank = (
            df.groupby("student_id")[src_col]
              .rank(method="min", ascending=False)
              .astype(np.float32)
              - 1.0
        )
        df[new_col] = rank
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="LTR v12: глобальные ранги in-available")
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()
    n_seeds = max(1, int(args.n_seeds))
    seeds = [BASE_SEED + i for i in range(n_seeds)]

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print(f"LTR v12: global-rank features, n_seeds={n_seeds}, seeds={seeds}")
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
    print("\n[1/8] Статистики...")
    stats = FeatureStats.fit(train, modules)
    stats.add_val_solution_popularity(val_solution)

    # ==== 2. Train-срезы ====
    print("\n[2/8] Train-срезы...")
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

    # ==== 3. Holdout + p2 ====
    print("\n[3/8] Holdout + p2...")
    val_p2 = [s for s in val if _n_periods(s) == 2]
    rng = random.Random(HOLDOUT_SEED)
    shuffled = list(val_p2)
    rng.shuffle(shuffled)
    n_holdout = int(len(shuffled) * HOLDOUT_FRACTION)
    holdout_records = shuffled[:n_holdout]
    p2_records = shuffled[n_holdout:]
    holdout_ids = {s["id"] for s in holdout_records}
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

    # ==== 4. Val и test ====
    print("\n[4/8] Val и test...")
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

    # ==== 5. НОВЫЕ признаки ====
    print("\n[5/8] Глобальные ранги in-available...")
    train_df_p1 = add_global_rank_features(train_df_p1)
    train_df_p0 = add_global_rank_features(train_df_p0)
    train_df_pmask = add_global_rank_features(train_df_pmask)
    p2_df = add_global_rank_features(p2_df)
    holdout_df = add_global_rank_features(holdout_df)
    val_df = add_global_rank_features(val_df)
    test_df = add_global_rank_features(test_df)

    # Диагностика: для cold-студентов в val должно быть pop_rank_global в норме
    cold_ids_val = identify_cold_students(val)
    val_cold_mask = val_df["student_id"].isin(cold_ids_val)
    sample_cold = val_df[val_cold_mask]
    if len(sample_cold) > 0:
        print(f"      cold-val: pop_rank_global_in_available "
              f"mean={sample_cold['pop_rank_global_in_available'].mean():.2f}, "
              f"max={sample_cold['pop_rank_global_in_available'].max():.0f}")
        print(f"                pop_rank_in_available (институтский, старый) "
              f"mean={sample_cold['pop_rank_in_available'].mean():.2f}, "
              f"max={sample_cold['pop_rank_in_available'].max():.0f}")

    # ==== 6. Веса, префиксы, категориальные ====
    print("\n[6/8] Веса, префиксы, категориальные...")
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

    FEATURE_COLS_V12: list[str] = FEATURE_COLS_V4 + [
        "pop_rank_global_in_available",
        "pop_rank_last_period_in_available",
        "pop_rank_first_period_in_available",
    ]

    # ==== 7. Multi-seed обучение ====
    print(f"\n[7/8] Обучение {n_seeds} моделей, {len(FEATURE_COLS_V12)} признаков...")

    X_train, y_train, g_train, w_train, _ = prepare_xygw(
        train_df, feature_cols=FEATURE_COLS_V12,
    )
    X_ho, y_ho, g_ho, w_ho, ho_sorted = prepare_xygw(
        holdout_df, feature_cols=FEATURE_COLS_V12,
    )
    _, _, _, _, val_sorted = prepare_xygw(
        val_df, feature_cols=FEATURE_COLS_V12,
    )
    _, _, _, _, test_sorted = prepare_xygw(
        test_df, feature_cols=FEATURE_COLS_V12,
    )

    print(f"      Train: X={X_train.shape}, групп={len(g_train)}")

    scores_ho_list: list[np.ndarray] = []
    scores_val_list: list[np.ndarray] = []
    scores_test_list: list[np.ndarray] = []
    best_iters: list[int] = []
    last_model: Any = None

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

        scores_ho_list.append(predict_scores(model, ho_sorted, feature_cols=FEATURE_COLS_V12))
        scores_val_list.append(predict_scores(model, val_sorted, feature_cols=FEATURE_COLS_V12))
        scores_test_list.append(predict_scores(model, test_sorted, feature_cols=FEATURE_COLS_V12))
        last_model = model

    scores_ho = np.mean(scores_ho_list, axis=0)
    scores_val = np.mean(scores_val_list, axis=0)
    scores_test = np.mean(scores_test_list, axis=0)
    print(f"\n      best_iter: mean={np.mean(best_iters):.0f}, "
          f"min={min(best_iters)}, max={max(best_iters)}")

    save_model(last_model, MODELS_DIR / "lgbm_ranker_v12.txt")

    # ==== 8. Оценка и submissions ====
    print("\n[8/8] Оценка и submission:")

    rec_ho = top_k_per_student(ho_sorted, scores_ho, k=5)
    rec_ho_clean = {sid.replace("ho__", ""): v for sid, v in rec_ho.items()}
    gt_ho = {sid: gt for sid, gt in val_gt_raw.items() if sid in holdout_ids}
    mean_ho, _ = evaluate_pndcg5(rec_ho_clean, gt_ho, k=5)

    print(f"      Holdout (n={len(holdout_records)}): {mean_ho:.4f}")
    print(f"          v7:  {V7_HOLDOUT:.4f}")
    print(f"          v12: {mean_ho:.4f}   (Δ = {mean_ho - V7_HOLDOUT:+.4f})")

    rec_val_ltr = top_k_per_student(val_sorted, scores_val, k=5)
    mean_val, cold_val, warm_val = _segment_means(rec_val_ltr, val_gt_raw, cold_ids_val)
    est_val = TEST_COLD_SHARE * cold_val + (1 - TEST_COLD_SHARE) * warm_val

    print(f"\n      Val LTR raw: mean={mean_val:.4f}  cold={cold_val:.4f}  "
          f"warm={warm_val:.4f}")
    print(f"          v7 cold (ref) = {V7_COLD_VAL:.4f}, "
          f"v12 cold = {cold_val:.4f}  (Δ = {cold_val - V7_COLD_VAL:+.4f})")
    print(f"      test_estimate = {est_val:.4f}")

    val_path = SUBMISSIONS_DIR / "val_submission_ltr_v12.csv"
    write_submission(rec_val_ltr, val_path)
    print(f"      Сохранено: {val_path}")

    rec_test = top_k_per_student(test_sorted, scores_test, k=5)
    test_path = SUBMISSIONS_DIR / "submission_v12.csv"
    write_submission(rec_test, test_path)
    print(f"      Сохранено: {test_path}")

    print("\n[Feature importance — top-15]")
    imp = feature_importance_report(last_model, feature_cols=FEATURE_COLS_V12, top_n=15)
    for _, row in imp.iterrows():
        print(f"      {row['feature']:<32}  gain = {row['gain']:>12.0f}")

    print(f"\n[OK] Пайплайн завершён за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()