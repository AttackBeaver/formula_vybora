"""
LTR v11: специализированный cold-ranker.

Гипотеза: cold-сегмент (35.2% test) недоучивается общей LTR-моделью,
потому что 88% обучающего сигнала — warm-паттерны. Обучаем ДВЕ модели:

    Warm-ranker: train = p1(w=1.0) + p2(w=3.0)   (как v7)
    Cold-ranker: train = pmask(w=1.0) + p0(w=0.5)

На инференсе роутим по is_cold:
    cold-студент  → усреднение cold-моделей
    warm-студент  → усреднение warm-моделей

Отличия от v7:
    - Нет p1_prob (не помогает, v10b).
    - truncation_level=15 (5 катастрофически вредит, v10b).
    - Два ансамбля вместо одного.
    - У cold-модели своя early-stopping валидация (20% pmask).
    - Первый прогон — n_seeds=1 для быстрой проверки гипотезы.

Fallback НЕ применяется (по README диагностика показала: LTR > fallback на cold).

Запуск:
    python -m src.run_ltr_v11
    python -m src.run_ltr_v11 --n-seeds 3
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

# Веса срезов — как в v7
WEIGHT_P1 = 1.0
WEIGHT_P0 = 0.5
WEIGHT_PMASK = 1.0
WEIGHT_P2 = 3.0

HOLDOUT_FRACTION = 0.20
HOLDOUT_SEED = 42
BASE_SEED = 42

# ВАЖНО: truncation_level=15 (v10b показал, что 5 вредит)
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

# V7 для сравнения
V7_HOLDOUT = 0.5134
V7_COLD_VAL = 0.2474  # из v10b B, на полном val


def _n_periods(r: dict[str, Any]) -> int:
    return len(r.get("history") or [])


def _add_weight(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = np.float32(weight)
    return df


def _strip_prefix(sid_series: pd.Series) -> pd.Series:
    return sid_series.str.replace(r"^[a-z0-9]+__", "", regex=True)


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
# Multi-seed обучение на произвольном train + собственном holdout
# ---------------------------------------------------------------------------
def train_ensemble(
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    seeds: list[int],
) -> tuple[list[Any], list[int]]:
    """Обучает ансамбль на train_df, валидируясь на val_df. Возвращает модели и best_iters."""
    X_train, y_train, g_train, w_train, _ = prepare_xygw(
        train_df, feature_cols=feature_cols,
    )
    X_val, y_val, g_val, w_val, _ = prepare_xygw(
        val_df, feature_cols=feature_cols,
    )
    print(f"      [{name}] Train: X={X_train.shape}, групп={len(g_train)}")
    print(f"      [{name}] Val:   X={X_val.shape}, групп={len(g_val)}")

    models: list[Any] = []
    best_iters: list[int] = []
    for i, seed in enumerate(seeds):
        params = {**FAST_PARAMS, "seed": seed}
        t0 = time.time()
        model = train_lgbm_ranker(
            X_train, y_train, g_train,
            X_val, y_val, g_val,
            params=params,
            num_boost_round=NUM_BOOST_ROUND,
            early_stopping_rounds=EARLY_STOPPING,
            log_every=500,
            sample_weight_train=w_train,
            sample_weight_val=w_val,
        )
        best_iters.append(model.best_iteration)
        print(f"        [{name}] seed={seed}: best_iter={model.best_iteration}, "
              f"{time.time() - t0:.1f} сек")
        models.append(model)
    return models, best_iters


def predict_ensemble_mean(
    models: list[Any],
    df_sorted: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    scores_list = [predict_scores(m, df_sorted, feature_cols=feature_cols) for m in models]
    return np.mean(scores_list, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="LTR v11: cold-ranker + warm-ranker")
    parser.add_argument("--n-seeds", type=int, default=1,
                        help="Сколько моделей на каждую группу (warm / cold).")
    args = parser.parse_args()
    n_seeds = max(1, int(args.n_seeds))
    seeds = [BASE_SEED + i for i in range(n_seeds)]

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print(f"LTR v11: cold-ranker + warm-ranker, n_seeds={n_seeds}, seeds={seeds}")
    print(f"truncation_level=15 (возврат к v7)")
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

    # ==== 3. Warm holdout из val (periods=2) + p2-срез ====
    print("\n[3/8] Warm holdout + p2-срез...")
    val_p2 = [s for s in val if _n_periods(s) == 2]
    rng = random.Random(HOLDOUT_SEED)
    shuffled = list(val_p2)
    rng.shuffle(shuffled)
    n_holdout = int(len(shuffled) * HOLDOUT_FRACTION)
    warm_holdout_records = shuffled[:n_holdout]
    p2_records = shuffled[n_holdout:]
    warm_holdout_ids = {s["id"] for s in warm_holdout_records}
    print(f"      p2={len(p2_records)}  warm_holdout={len(warm_holdout_records)}")

    p2_df = build_dataset(
        p2_records, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )
    warm_holdout_df = build_dataset(
        warm_holdout_records, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )

    # ==== 4. Cold holdout: 20% pmask-студентов (GroupSplit) ====
    print("\n[4/8] Cold holdout: 20% pmask-студентов...")
    pmask_unique_ids = sorted(train_df_pmask["student_id"].unique().tolist())
    rng2 = random.Random(HOLDOUT_SEED + 1)
    rng2.shuffle(pmask_unique_ids)
    n_cold_holdout = int(len(pmask_unique_ids) * 0.20)
    cold_holdout_ids = set(pmask_unique_ids[:n_cold_holdout])
    cold_train_ids = set(pmask_unique_ids[n_cold_holdout:])

    train_df_pmask_holdout = train_df_pmask[
        train_df_pmask["student_id"].isin(cold_holdout_ids)
    ].copy()
    train_df_pmask_train = train_df_pmask[
        train_df_pmask["student_id"].isin(cold_train_ids)
    ].copy()
    print(f"      pmask_train: {len(train_df_pmask_train):,}, "
          f"pmask_holdout: {len(train_df_pmask_holdout):,}, "
          f"студентов в holdout={len(cold_holdout_ids)}")

    # ==== 5. Val и test датасеты ====
    print("\n[5/8] Val и test...")
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
    print("\n[6/8] Веса, префиксы, категориальные...")
    train_df_p1 = _add_weight(train_df_p1, WEIGHT_P1)
    train_df_p0 = _add_weight(train_df_p0, WEIGHT_P0)
    train_df_pmask_train = _add_weight(train_df_pmask_train, WEIGHT_PMASK)
    train_df_pmask_holdout = _add_weight(train_df_pmask_holdout, 1.0)
    p2_df = _add_weight(p2_df, WEIGHT_P2)
    warm_holdout_df = _add_weight(warm_holdout_df, 1.0)
    val_df = _add_weight(val_df, 1.0)
    test_df = _add_weight(test_df, 1.0)

    # Префиксы
    train_df_p1["student_id"] = "p1__" + train_df_p1["student_id"].astype(str)
    train_df_p0["student_id"] = "p0__" + train_df_p0["student_id"].astype(str)
    train_df_pmask_train["student_id"] = "pmask__" + train_df_pmask_train["student_id"].astype(str)
    train_df_pmask_holdout["student_id"] = "pmaskho__" + train_df_pmask_holdout["student_id"].astype(str)
    p2_df["student_id"] = "p2__" + p2_df["student_id"].astype(str)
    warm_holdout_df["student_id"] = "who__" + warm_holdout_df["student_id"].astype(str)

    # Warm train и cold train
    warm_train_df = pd.concat([train_df_p1, p2_df], ignore_index=True)
    cold_train_df = pd.concat([train_df_pmask_train, train_df_p0], ignore_index=True)

    # Единый словарь категориальных
    all_parts = [
        warm_train_df, cold_train_df,
        warm_holdout_df, train_df_pmask_holdout,
        val_df, test_df,
    ]
    n_parts = [len(p) for p in all_parts]
    all_df = pd.concat(all_parts, ignore_index=True)
    all_df, _ = encode_categoricals(all_df, all_df, CATEGORICAL_FEATS)

    idx = 0
    splits = []
    for n in n_parts:
        splits.append(all_df.iloc[idx:idx + n].reset_index(drop=True))
        idx += n
    warm_train_df, cold_train_df, warm_holdout_df, cold_holdout_df, val_df, test_df = splits

    print(f"      warm_train: {len(warm_train_df):,}, cold_train: {len(cold_train_df):,}")
    print(f"      warm_holdout: {len(warm_holdout_df):,}, cold_holdout: {len(cold_holdout_df):,}")

    # ==== 7. Обучение двух ансамблей ====
    print(f"\n[7/8] Обучение (n_seeds={n_seeds} на каждую группу)...")

    warm_models, warm_iters = train_ensemble(
        name="warm",
        train_df=warm_train_df, val_df=warm_holdout_df,
        feature_cols=FEATURE_COLS_V4,
        seeds=seeds,
    )
    cold_models, cold_iters = train_ensemble(
        name="cold",
        train_df=cold_train_df, val_df=cold_holdout_df,
        feature_cols=FEATURE_COLS_V4,
        seeds=seeds,
    )

    save_model(warm_models[-1], MODELS_DIR / "lgbm_ranker_v11_warm.txt")
    save_model(cold_models[-1], MODELS_DIR / "lgbm_ranker_v11_cold.txt")

    # ==== 8. Оценка ====
    print("\n[8/8] Оценка и submission:")

    # Warm holdout (только warm-модель)
    _, _, _, _, who_sorted = prepare_xygw(warm_holdout_df, feature_cols=FEATURE_COLS_V4)
    scores_who = predict_ensemble_mean(warm_models, who_sorted, FEATURE_COLS_V4)
    rec_who = top_k_per_student(who_sorted, scores_who, k=5)
    rec_who_clean = {sid.replace("who__", ""): v for sid, v in rec_who.items()}
    gt_who = {sid: gt for sid, gt in val_gt_raw.items() if sid in warm_holdout_ids}
    mean_who, _ = evaluate_pndcg5(rec_who_clean, gt_who, k=5)

    # Cold holdout (только cold-модель)
    _, _, _, _, cho_sorted = prepare_xygw(cold_holdout_df, feature_cols=FEATURE_COLS_V4)
    scores_cho = predict_ensemble_mean(cold_models, cho_sorted, FEATURE_COLS_V4)
    rec_cho = top_k_per_student(cho_sorted, scores_cho, k=5)
    rec_cho_clean = {sid.replace("pmaskho__", ""): v for sid, v in rec_cho.items()}
    # ground truth для pmask-студентов = target_period_2 (что они выбрали в P2)
    pmask_gt: dict[str, str] = {}
    for s in train_students_p1:
        sid = s["id"]
        if sid in cold_holdout_ids:
            # Собираем строку 'm_XXX:priority ...' из target_period_2(s)
            tgt = target_period_2(s)
            pmask_gt[sid] = " ".join(f"{m}:{p}" for m, p in tgt.items())
    mean_cho, _ = evaluate_pndcg5(rec_cho_clean, pmask_gt, k=5)

    print(f"      --- Холдауты:")
    print(f"          warm_holdout (n={len(warm_holdout_records)}): {mean_who:.4f}   "
          f"(v7 reference = {V7_HOLDOUT:.4f})")
    print(f"          cold_holdout (n={len(cold_holdout_ids)}): {mean_cho:.4f}   "
          f"(v7 cold val = {V7_COLD_VAL:.4f})")
    print(f"          best_iter warm: {warm_iters}")
    print(f"          best_iter cold: {cold_iters}")

    # --- Val: роутинг ---
    cold_ids_val = identify_cold_students(val)
    warm_ids_val = {s["id"] for s in val if s["id"] not in cold_ids_val}

    _, _, _, _, val_sorted = prepare_xygw(val_df, feature_cols=FEATURE_COLS_V4)
    scores_val_warm = predict_ensemble_mean(warm_models, val_sorted, FEATURE_COLS_V4)
    scores_val_cold = predict_ensemble_mean(cold_models, val_sorted, FEATURE_COLS_V4)

    # Роутинг: для cold-студентов — scores от cold-модели,
    # для warm — от warm-модели.
    # Для этого соберём финальный скор по строкам val_sorted.
    is_cold_series = val_sorted["student_id"].isin(cold_ids_val).to_numpy()
    scores_val_route = np.where(is_cold_series, scores_val_cold, scores_val_warm)

    rec_val_route = top_k_per_student(val_sorted, scores_val_route, k=5)
    mean_val, cold_val, warm_val = _segment_means(rec_val_route, val_gt_raw, cold_ids_val)
    est_val = TEST_COLD_SHARE * cold_val + (1 - TEST_COLD_SHARE) * warm_val

    print(f"\n      --- Val (routing cold/warm):")
    print(f"          mean={mean_val:.4f}  cold={cold_val:.4f}  warm={warm_val:.4f}")
    print(f"          test_estimate = {est_val:.4f}")

    # Сравнение с v7 на том же val
    print(f"\n      --- Сравнение на val (холодные студенты):")
    print(f"          v7 (v10b B) cold = 0.2474")
    print(f"          v11 cold-модель  = {cold_val:.4f}   (Δ = {cold_val - 0.2474:+.4f})")

    val_path = SUBMISSIONS_DIR / "val_submission_ltr_v11.csv"
    write_submission(rec_val_route, val_path)
    print(f"      Сохранено: {val_path}")

    # --- Test: роутинг ---
    cold_ids_test = identify_cold_students(test)
    _, _, _, _, test_sorted = prepare_xygw(test_df, feature_cols=FEATURE_COLS_V4)
    scores_test_warm = predict_ensemble_mean(warm_models, test_sorted, FEATURE_COLS_V4)
    scores_test_cold = predict_ensemble_mean(cold_models, test_sorted, FEATURE_COLS_V4)
    is_cold_test = test_sorted["student_id"].isin(cold_ids_test).to_numpy()
    scores_test_route = np.where(is_cold_test, scores_test_cold, scores_test_warm)
    rec_test = top_k_per_student(test_sorted, scores_test_route, k=5)

    test_path = SUBMISSIONS_DIR / "submission_v11.csv"
    write_submission(rec_test, test_path)
    print(f"      Сохранено: {test_path}  (cold в test: {len(cold_ids_test)})")

    print(f"\n[OK] Пайплайн завершён за {time.time() - t0:.1f} сек.")

    print("\n[Feature importance — warm-модель]")
    imp_w = feature_importance_report(warm_models[-1], feature_cols=FEATURE_COLS_V4, top_n=10)
    for _, row in imp_w.iterrows():
        print(f"      {row['feature']:<28}  gain = {row['gain']:>12.0f}")

    print("\n[Feature importance — cold-модель]")
    imp_c = feature_importance_report(cold_models[-1], feature_cols=FEATURE_COLS_V4, top_n=10)
    for _, row in imp_c.iterrows():
        print(f"      {row['feature']:<28}  gain = {row['gain']:>12.0f}")


if __name__ == "__main__":
    main()