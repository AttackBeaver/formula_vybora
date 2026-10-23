"""
eval_clean.py — честная оценка БЕЗ val в обучении.
"""
from __future__ import annotations

import argparse
import random
import time
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
    target_period_1,
    target_period_2,
)
from src.ltr_model import (
    CATEGORICAL_FEATS,
    COMPATIBLE_FEATURE_SETS,
    FEATURE_SETS,
    encode_categoricals,
    predict_scores,
    prepare_xygw,
    top_k_per_student,
    train_lgbm_ranker,
)
from src.metric import evaluate_pndcg5

TEST_SHARES = {"cold": 0.352, "p1": 0.020, "p2": 0.324, "p3": 0.304}

WEIGHT_P1 = 1.0
WEIGHT_P0 = 0.5
WEIGHT_PMASK = 1.0

ES_FRACTION = 0.10
ES_SEED = 42
BASE_SEED = 42

FAST_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
}
NUM_BOOST_ROUND = 2500
EARLY_STOPPING = 100


def _add_weight(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = np.float32(weight)
    return df


def _segment_of(s: dict[str, Any]) -> str:
    if s["id"] in _cold_cache:
        return "cold"
    n = len(s.get("history") or [])
    if n == 0:
        return "cold"
    return f"p{n}"


_cold_cache: set[str] = set()


def _seg_mean(per_student: dict[str, float], ids: list[str]) -> tuple[float, int]:
    vals = [per_student[sid] for sid in ids if sid in per_student]
    if not vals:
        return 0.0, 0
    return float(np.mean(vals)), len(vals)


def _validate_combo(content_backend: str, feature_set: str) -> None:
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"feature_set={feature_set!r}, ожидалось одно из {sorted(FEATURE_SETS)}"
        )
    allowed = COMPATIBLE_FEATURE_SETS.get(content_backend, ())
    if feature_set not in allowed:
        raise ValueError(
            f"feature_set={feature_set!r} требует content_backend ∈ "
            f"{sorted(allowed)}, получено {content_backend!r}."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Honest eval без val в обучении")
    ap.add_argument(
        "--content-backend", choices=("tfidf", "embed", "both", "none"),
        default="tfidf",
    )
    ap.add_argument(
        "--feature-set", choices=tuple(FEATURE_SETS.keys()), default="v4",
    )
    ap.add_argument("--n-seeds", type=int, default=3)
    args = ap.parse_args()

    try:
        _validate_combo(args.content_backend, args.feature_set)
    except ValueError as e:
        print(f"ERROR: {e}")
        raise SystemExit(2)

    n_seeds = max(1, int(args.n_seeds))
    seeds = [BASE_SEED + i for i in range(n_seeds)]
    feature_cols = FEATURE_SETS[args.feature_set]

    print("=" * 78)
    print("eval_clean: обучение только на train, оценка на всей val")
    print(f"  content_backend = {args.content_backend}")
    print(f"  feature_set     = {args.feature_set} ({len(feature_cols)} признаков)")
    print(f"  n_seeds         = {n_seeds}")
    print(f"  ES_FRACTION     = {ES_FRACTION}")
    print("=" * 78)

    t0 = time.time()
    data = load_all()
    train = data["train"]
    val = data["val"]
    modules = data["modules"]
    val_solution = data["val_solution"]
    gt_raw = dict(zip(val_solution["id"], val_solution["ground_truth"]))

    global _cold_cache
    _cold_cache = identify_cold_students(val)
    print(f"\nval cold: {len(_cold_cache)}, warm: {len(val) - len(_cold_cache)}")

    # 1. Внутренний ES-split по train
    rng = random.Random(ES_SEED)
    train_shuffled = list(train)
    rng.shuffle(train_shuffled)
    n_es = int(len(train_shuffled) * ES_FRACTION)
    es_students = train_shuffled[:n_es]
    core_students = train_shuffled[n_es:]
    print(f"\n[1/5] ES-split: core={len(core_students)}, es={len(es_students)}")

    # 2. FeatureStats на train (без val_solution — никаких val-таргетов)
    print(f"\n[2/5] FeatureStats на train (content_backend='{args.content_backend}')...")
    stats = FeatureStats.fit(train, modules, content_backend=args.content_backend)

    # 3. Датасеты
    print("\n[3/5] Строю train и ES срезы...")

    core_p1 = [s for s in core_students
               if any(h.get("period") == 2 for h in s.get("history", []))]
    core_p0 = [s for s in core_students
               if any(h.get("period") == 1 for h in s.get("history", []))]

    train_df_p1 = build_dataset(
        core_p1, stats,
        candidates_fn=make_candidates_train_period2(stats),
        target_fn=target_period_2,
        past_periods=[1],
    )
    train_df_p0 = build_dataset(
        core_p0, stats,
        candidates_fn=make_candidates_train_period1(stats),
        target_fn=target_period_1,
        past_periods=[],
    )
    train_df_pmask = build_dataset(
        core_p1, stats,
        candidates_fn=make_candidates_train_period2(stats),
        target_fn=target_period_2,
        past_periods=[1],
        mask_profile=True,
    )
    es_df = build_dataset(
        es_students, stats,
        candidates_fn=make_candidates_train_period2(stats),
        target_fn=target_period_2,
        past_periods=[1],
    )
    print(f"      p1={len(train_df_p1):,}  p0={len(train_df_p0):,}  "
          f"pmask={len(train_df_pmask):,}  es={len(es_df):,}")

    train_df_p1 = _add_weight(train_df_p1, WEIGHT_P1)
    train_df_p0 = _add_weight(train_df_p0, WEIGHT_P0)
    train_df_pmask = _add_weight(train_df_pmask, WEIGHT_PMASK)
    es_df = _add_weight(es_df, 1.0)

    train_df_p1["student_id"] = "p1__" + train_df_p1["student_id"].astype(str)
    train_df_p0["student_id"] = "p0__" + train_df_p0["student_id"].astype(str)
    train_df_pmask["student_id"] = "pmask__" + train_df_pmask["student_id"].astype(str)
    es_df["student_id"] = "es__" + es_df["student_id"].astype(str)

    train_df = pd.concat(
        [train_df_p1, train_df_p0, train_df_pmask], ignore_index=True,
    )

    # Val — одна таблица без target (target подтянем отдельно).
    val_df = build_dataset(
        val, stats,
        candidates_fn=candidates_available,
        target_fn=None,
        past_periods=None,
    )
    val_df = _add_weight(val_df, 1.0)
    print(f"      train: {len(train_df):,} строк, "
          f"val: {len(val_df):,} строк")

    # Единый словарь категорий
    n_tr = len(train_df)
    n_es = len(es_df)
    n_va = len(val_df)
    all_df = pd.concat([train_df, es_df, val_df], ignore_index=True)
    all_df, _ = encode_categoricals(all_df, all_df, CATEGORICAL_FEATS)
    train_df = all_df.iloc[:n_tr].reset_index(drop=True)
    es_df = all_df.iloc[n_tr:n_tr + n_es].reset_index(drop=True)
    val_df = all_df.iloc[n_tr + n_es:].reset_index(drop=True)

    X_train, y_train, g_train, w_train, _ = prepare_xygw(
        train_df, feature_cols=feature_cols)
    X_es, y_es, g_es, w_es, es_sorted = prepare_xygw(
        es_df, feature_cols=feature_cols)
    _, _, _, _, val_sorted = prepare_xygw(val_df, feature_cols=feature_cols)
    print(f"      Train: X={X_train.shape}, групп={len(g_train)}")
    print(f"      ES:    X={X_es.shape},   групп={len(g_es)}")

    # 4. Multi-seed обучение (early stopping на train-ES)
    print(f"\n[4/5] Обучение {n_seeds} моделей (early stopping на train-ES)...")
    val_scores_list: list[np.ndarray] = []
    best_iters: list[int] = []
    for i, seed in enumerate(seeds):
        params = {**FAST_PARAMS, "seed": seed}
        t_tr = time.time()
        model = train_lgbm_ranker(
            X_train, y_train, g_train,
            X_es, y_es, g_es,
            params=params,
            num_boost_round=NUM_BOOST_ROUND,
            early_stopping_rounds=EARLY_STOPPING,
            log_every=500,
            sample_weight_train=w_train,
            sample_weight_val=w_es,
        )
        best_iters.append(model.best_iteration)
        print(f"      seed={seed} за {time.time() - t_tr:.1f} сек, "
              f"best_iter={model.best_iteration}")
        val_scores_list.append(
            predict_scores(model, val_sorted, feature_cols=feature_cols)
        )

    val_scores = np.mean(val_scores_list, axis=0)
    print(f"      best_iter: mean={np.mean(best_iters):.0f}, "
          f"min={min(best_iters)}, max={max(best_iters)}")

    # 5. Оценка на val
    print("\n[5/5] Сегментные оценки на val:")
    rec_val = top_k_per_student(val_sorted, val_scores, k=5)
    _, per_student = evaluate_pndcg5(rec_val, gt_raw, k=5)

    cold_ids = [s["id"] for s in val if s["id"] in _cold_cache]
    p1_ids = [s["id"] for s in val
              if s["id"] not in _cold_cache
              and len(s.get("history") or []) == 1]
    p2_ids = [s["id"] for s in val
              if s["id"] not in _cold_cache
              and len(s.get("history") or []) >= 2]

    m_cold, n_cold = _seg_mean(per_student, cold_ids)
    m_p1, n_p1 = _seg_mean(per_student, p1_ids)
    m_p2, n_p2 = _seg_mean(per_student, p2_ids)

    print(f"  cold : n={n_cold:>4}  pNDCG@5 = {m_cold:.4f}")
    print(f"  p1   : n={n_p1:>4}  pNDCG@5 = {m_p1:.4f}")
    print(f"  p2   : n={n_p2:>4}  pNDCG@5 = {m_p2:.4f}")

    clean = (
        TEST_SHARES["cold"] * m_cold
        + TEST_SHARES["p1"] * m_p1
        + (TEST_SHARES["p2"] + TEST_SHARES["p3"]) * m_p2
    )

    print(f"\n  clean_composite = {clean:.4f}")
    print(f"      0.352·{m_cold:.4f} + 0.020·{m_p1:.4f} + "
          f"0.628·{m_p2:.4f}")
    print("\n  Для сравнения (наивные, ОТРАВЛЕННЫЕ):")
    print("      composite script   (estimate_test_score) ≈ 0.408–0.466")
    print("      holdout p2 (run_ltr_v29)                  = 0.5147")
    print("      public v7 (факт)                          = 0.3348")
    print("\n  Если clean_composite ∈ [0.30, 0.38] — метрика калибрована.")
    print("  Если > 0.40 — утечка была не в val_p2, копаем дальше.")

    print(f"\n[OK] eval_clean завершён за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()