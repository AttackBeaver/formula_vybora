"""
KFold-evaluator: честная метрика на warm-val + cold-val.

ЗАЧЕМ:
    Старый holdout (284 val_p2) шумит ±0.0075. Все приросты v10-v25 были
    в пределах шума. KFold даёт шум ±0.0023 — в 3 раза точнее.

КАК:
    5-fold по всем warm-val студентам (val_p1 + val_p2 = 2955).
    На каждом фолде:
        - 80% warm-val → обучающие срезы valp1/valp2 (с фиксированным весом).
        - 20% warm-val → OOF-holdout.
    Train p0/p1/pmask — фиксированные, всегда в обучении.
    Cold-val (88) — оценивается через global pop (fallback отключён).

    Composite = 0.352 × cold + 0.648 × warm_OOF.

ПАРАМЕТРЫ:
    --apply-fix {0,1}       — перезаписывать pop_last/prev/first по истинным периодам (default 1)
    --weight-valp1 FLOAT    — вес val_p1 в обучении (0 = отключить срез, default 1.0)
    --weight-p2 FLOAT       — вес val_p2 в обучении (default 3.0)
    --n-seeds INT           — seeds на фолд (default 1)
    --n-folds INT           — число фолдов (default 5)

ПРИМЕРЫ:
    python -m src.kfold_eval                              # v23-базовая
    python -m src.kfold_eval --weight-valp1 0.0           # v24 (без val_p1)
    python -m src.kfold_eval --apply-fix 0                # чистый v7 через KFold
    python -m src.kfold_eval --n-seeds 3                  # 3 seeds на фолд
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

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
    predict_scores,
    prepare_xygw,
    top_k_per_student,
    train_lgbm_ranker,
)
from src.metric import evaluate_pndcg5
from src.popularity import compute_popularity

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_COLD_SHARE = 0.352

WEIGHT_P1 = 1.0
WEIGHT_P0 = 0.5
WEIGHT_PMASK = 1.0

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


def _n_periods(r: dict[str, Any]) -> int:
    return len(r.get("history") or [])


def _add_weight(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = np.float32(weight)
    return df


def _strip_prefix(sid_series: pd.Series) -> pd.Series:
    return sid_series.str.replace(r"^[a-z0-9]+__", "", regex=True)


def add_correct_period_features(
    df: pd.DataFrame,
    students: list[dict],
    pop_by_period: dict[int, dict[str, float]],
    past_periods: list[int] | None,
) -> pd.DataFrame:
    period_map: dict[str, tuple[int | None, int | None, int | None]] = {}
    for s in students:
        sid = s.get("id") or s.get("student_id")
        if sid is None:
            continue
        periods = sorted({h.get("period") for h in (s.get("history") or [])
                          if h.get("period") is not None})
        if past_periods is not None:
            periods = [p for p in periods if p in past_periods]
        if not periods:
            period_map[sid] = (None, None, None)
        elif len(periods) == 1:
            period_map[sid] = (periods[0], None, periods[0])
        else:
            period_map[sid] = (periods[-1], periods[-2], periods[0])

    n = len(df)
    pop_last = np.zeros(n, dtype=np.float32)
    pop_prev = np.zeros(n, dtype=np.float32)
    pop_first = np.zeros(n, dtype=np.float32)
    sids_raw = _strip_prefix(df["student_id"].astype(str)).to_numpy()
    mids = df["module_id"].astype(str).to_numpy()
    for i in range(n):
        sid = sids_raw[i]
        mid = mids[i]
        last_p, prev_p, first_p = period_map.get(sid, (None, None, None))
        if last_p is not None:
            pop_last[i] = float(pop_by_period.get(last_p, {}).get(mid, 0.0))
        if prev_p is not None:
            pop_prev[i] = float(pop_by_period.get(prev_p, {}).get(mid, 0.0))
        if first_p is not None:
            pop_first[i] = float(pop_by_period.get(first_p, {}).get(mid, 0.0))

    df = df.copy()
    df["pop_last_period"] = pop_last
    df["pop_prev_period"] = pop_prev
    df["pop_first_period"] = pop_first
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


def _cold_rec_global_pop(
    cold_students: list[dict],
    pop_weighted: dict[str, float],
    k: int = 5,
) -> dict[str, list[str]]:
    rec: dict[str, list[str]] = {}
    for s in cold_students:
        sid = s["id"]
        avail = list(s.get("available_modules") or [])
        scored = [(m, pop_weighted.get(m, 0.0)) for m in avail]
        scored.sort(key=lambda x: (-x[1], x[0]))
        rec[sid] = [m for m, _ in scored[:k]]
    return rec


def _prepare_static_slices(
    stats: FeatureStats,
    train_students_p1: list[dict],
    train_students_p0: list[dict],
    apply_fix: bool,
    val_gt_raw: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """p0, p1, pmask — фиксированные, не зависят от фолда."""
    train_df_p1 = build_dataset(
        train_students_p1, stats,
        candidates_fn=make_candidates_train_period2(stats),
        target_fn=target_period_2,
        past_periods=[1],
    )
    train_df_p0 = build_dataset(
        train_students_p0, stats,
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
    if apply_fix:
        train_df_p1 = add_correct_period_features(
            train_df_p1, train_students_p1, stats.pop_by_period, past_periods=[1])
        train_df_p0 = add_correct_period_features(
            train_df_p0, train_students_p0, stats.pop_by_period, past_periods=[])
        train_df_pmask = add_correct_period_features(
            train_df_pmask, train_students_p1, stats.pop_by_period, past_periods=[1])

    train_df_p1 = _add_weight(train_df_p1, WEIGHT_P1)
    train_df_p0 = _add_weight(train_df_p0, WEIGHT_P0)
    train_df_pmask = _add_weight(train_df_pmask, WEIGHT_PMASK)

    train_df_p1["student_id"] = "p1__" + train_df_p1["student_id"].astype(str)
    train_df_p0["student_id"] = "p0__" + train_df_p0["student_id"].astype(str)
    train_df_pmask["student_id"] = "pmask__" + train_df_pmask["student_id"].astype(str)
    return train_df_p1, train_df_p0, train_df_pmask


def main() -> None:
    parser = argparse.ArgumentParser(description="KFold-evaluator")
    parser.add_argument("--apply-fix", type=int, default=1,
                        help="1 = перезаписать pop_last/prev по истинным периодам")
    parser.add_argument("--weight-valp1", type=float, default=1.0,
                        help="вес val_p1 среза (0 = отключить)")
    parser.add_argument("--weight-p2", type=float, default=3.0,
                        help="вес val_p2 среза")
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    apply_fix = bool(args.apply_fix)
    weight_valp1 = float(args.weight_valp1)
    weight_p2 = float(args.weight_p2)
    n_seeds = max(1, int(args.n_seeds))
    n_folds = max(2, int(args.n_folds))
    seeds = [BASE_SEED + i for i in range(n_seeds)]

    print("=" * 78)
    print(f"KFold-evaluator")
    print(f"  apply_fix={apply_fix}")
    print(f"  weight_valp1={weight_valp1}")
    print(f"  weight_p2={weight_p2}")
    print(f"  n_seeds={n_seeds}, n_folds={n_folds}")
    print("=" * 78)

    t0 = time.time()
    data = load_all()
    train = data["train"]
    val = data["val"]
    modules = data["modules"]
    val_solution = data["val_solution"]
    val_gt_raw = dict(zip(val_solution["id"], val_solution["ground_truth"]))

    # ==== 1. Статистики ====
    print("\n[1/4] Статистики...")
    stats = FeatureStats.fit(train, modules)
    stats.add_val_solution_popularity(val_solution)

    # ==== 2. Фиксированные срезы ====
    print("\n[2/4] Фиксированные срезы (p0/p1/pmask)...")
    train_students_p1 = [
        s for s in train
        if any(h.get("period") == 2 for h in s.get("history", []))
    ]
    train_students_p0 = [
        s for s in train if any(h.get("period") == 1 for h in s.get("history", []))
    ]
    train_df_p1, train_df_p0, train_df_pmask = _prepare_static_slices(
        stats, train_students_p1, train_students_p0, apply_fix, val_gt_raw,
    )
    print(f"      p1={len(train_df_p1):,}  p0={len(train_df_p0):,}  "
          f"pmask={len(train_df_pmask):,}")

    # ==== 3. KFold по warm-val ====
    print(f"\n[3/4] {n_folds}-fold по warm-val...")
    val_p1_records = [s for s in val if _n_periods(s) == 1]
    val_p2_records = [s for s in val if _n_periods(s) == 2]
    all_warm_val = [(s, 'p1') for s in val_p1_records] + [(s, 'p2') for s in val_p2_records]
    rng = random.Random(BASE_SEED)
    rng.shuffle(all_warm_val)
    print(f"      val_p1={len(val_p1_records)}, val_p2={len(val_p2_records)}, "
          f"total={len(all_warm_val)}")

    kf = KFold(n_splits=n_folds, shuffle=False)
    fold_splits = list(kf.split(all_warm_val))

    oof_rec: dict[str, list[str]] = {}
    fold_stats: list[dict[str, Any]] = []

    for fold_idx, (train_idx, hold_idx) in enumerate(fold_splits):
        fold_hold = [all_warm_val[i] for i in hold_idx]
        fold_train = [all_warm_val[i] for i in train_idx]

        fold_valp1_train = [s for s, t in fold_train if t == 'p1']
        fold_valp2_train = [s for s, t in fold_train if t == 'p2']
        fold_valp1_hold = [s for s, t in fold_hold if t == 'p1']
        fold_valp2_hold = [s for s, t in fold_hold if t == 'p2']

        # --- Срезы этого фолда ---
        train_parts = [train_df_p1, train_df_p0, train_df_pmask]

        if weight_valp1 > 0 and fold_valp1_train:
            df_vp1 = build_dataset(
                fold_valp1_train, stats,
                candidates_fn=candidates_available,
                target_fn=make_target_from_val_solution(val_gt_raw),
                past_periods=[1],
            )
            if apply_fix:
                df_vp1 = add_correct_period_features(
                    df_vp1, fold_valp1_train, stats.pop_by_period, past_periods=[1])
            df_vp1 = _add_weight(df_vp1, weight_valp1)
            df_vp1["student_id"] = "vp1__" + df_vp1["student_id"].astype(str)
            train_parts.append(df_vp1)

        if weight_p2 > 0 and fold_valp2_train:
            df_vp2 = build_dataset(
                fold_valp2_train, stats,
                candidates_fn=candidates_available,
                target_fn=make_target_from_val_solution(val_gt_raw),
                past_periods=[1, 2],
            )
            if apply_fix:
                df_vp2 = add_correct_period_features(
                    df_vp2, fold_valp2_train, stats.pop_by_period, past_periods=[1, 2])
            df_vp2 = _add_weight(df_vp2, weight_p2)
            df_vp2["student_id"] = "vp2__" + df_vp2["student_id"].astype(str)
            train_parts.append(df_vp2)

        # --- Holdout срезы ---
        hold_parts = []
        if fold_valp1_hold:
            df_hp1 = build_dataset(
                fold_valp1_hold, stats,
                candidates_fn=candidates_available,
                target_fn=make_target_from_val_solution(val_gt_raw),
                past_periods=[1],
            )
            if apply_fix:
                df_hp1 = add_correct_period_features(
                    df_hp1, fold_valp1_hold, stats.pop_by_period, past_periods=[1])
            df_hp1 = _add_weight(df_hp1, 1.0)
            df_hp1["student_id"] = "hp1__" + df_hp1["student_id"].astype(str)
            hold_parts.append(df_hp1)

        if fold_valp2_hold:
            df_hp2 = build_dataset(
                fold_valp2_hold, stats,
                candidates_fn=candidates_available,
                target_fn=make_target_from_val_solution(val_gt_raw),
                past_periods=[1, 2],
            )
            if apply_fix:
                df_hp2 = add_correct_period_features(
                    df_hp2, fold_valp2_hold, stats.pop_by_period, past_periods=[1, 2])
            df_hp2 = _add_weight(df_hp2, 1.0)
            df_hp2["student_id"] = "hp2__" + df_hp2["student_id"].astype(str)
            hold_parts.append(df_hp2)

        train_df_fold = pd.concat(train_parts, ignore_index=True)
        hold_df_fold = pd.concat(hold_parts, ignore_index=True)

        # Единое кодирование
        n_train = len(train_df_fold)
        n_hold = len(hold_df_fold)
        all_df = pd.concat([train_df_fold, hold_df_fold], ignore_index=True)
        all_df, _ = encode_categoricals(all_df, all_df, CATEGORICAL_FEATS)
        train_df_fold = all_df.iloc[:n_train].reset_index(drop=True)
        hold_df_fold = all_df.iloc[n_train:].reset_index(drop=True)

        X_train, y_train, g_train, w_train, _ = prepare_xygw(
            train_df_fold, feature_cols=FEATURE_COLS_V4)
        X_hold, y_hold, g_hold, w_hold, hold_sorted = prepare_xygw(
            hold_df_fold, feature_cols=FEATURE_COLS_V4)

        # --- Модели на всех seeds ---
        hold_preds_list: list[np.ndarray] = []
        for seed in seeds:
            params = {**FAST_PARAMS, "seed": seed + fold_idx * 100}
            model = train_lgbm_ranker(
                X_train, y_train, g_train,
                X_hold, y_hold, g_hold,
                params=params,
                num_boost_round=NUM_BOOST_ROUND,
                early_stopping_rounds=EARLY_STOPPING,
                log_every=1000,
                sample_weight_train=w_train,
                sample_weight_val=w_hold,
            )
            hold_preds_list.append(
                predict_scores(model, hold_sorted, feature_cols=FEATURE_COLS_V4))

        hold_preds = np.mean(hold_preds_list, axis=0)
        rec_fold = top_k_per_student(hold_sorted, hold_preds, k=5)

        # Снимаем префиксы и копим OOF
        rec_fold_clean: dict[str, list[str]] = {}
        for sid_pref, rec in rec_fold.items():
            sid_clean = sid_pref.replace("hp1__", "").replace("hp2__", "")
            rec_fold_clean[sid_clean] = rec
            oof_rec[sid_clean] = rec

        # --- Диагностика фолда (правильная, по чистым ID) ---
        p1_ids_hold = {s["id"] for s in fold_valp1_hold}
        p2_ids_hold = {s["id"] for s in fold_valp2_hold}
        m_p1 = _mean_for_ids(rec_fold_clean, val_gt_raw, p1_ids_hold)
        m_p2 = _mean_for_ids(rec_fold_clean, val_gt_raw, p2_ids_hold)

        fold_stats.append({
            "fold": fold_idx + 1,
            "n_p1_hold": len(p1_ids_hold),
            "n_p2_hold": len(p2_ids_hold),
            "m_p1": m_p1,
            "m_p2": m_p2,
            "n_train_slices": len(train_parts),
            "n_train_rows": n_train,
        })
        print(f"      fold {fold_idx+1}: p1_ho={len(p1_ids_hold)} p2_ho={len(p2_ids_hold)} "
              f"| OOF p1={m_p1:.4f} p2={m_p2:.4f} | n_train={n_train:,}")

    # ==== 4. Итог ====
    print(f"\n[4/4] Итог:")

    p1_ids_all = {s["id"] for s in val_p1_records}
    p2_ids_all = {s["id"] for s in val_p2_records}
    cold_ids = identify_cold_students(val)

    m_p1_all = _mean_for_ids(oof_rec, val_gt_raw, p1_ids_all)
    m_p2_all = _mean_for_ids(oof_rec, val_gt_raw, p2_ids_all)
    m_warm_all = _mean_for_ids(oof_rec, val_gt_raw, p1_ids_all | p2_ids_all)

    # Cold через global pop (fallback)
    pop_w, _ = compute_popularity(train)
    cold_students = [s for s in val if s["id"] in cold_ids]
    rec_cold = _cold_rec_global_pop(cold_students, pop_w)
    m_cold = _mean_for_ids(rec_cold, val_gt_raw, cold_ids)

    composite = TEST_COLD_SHARE * m_cold + (1 - TEST_COLD_SHARE) * m_warm_all

    print(f"\n      Warm OOF:")
    print(f"          p1 (n={len(p1_ids_all)}):     {m_p1_all:.4f}")
    print(f"          p2 (n={len(p2_ids_all)}):     {m_p2_all:.4f}")
    print(f"          всего (n={len(p1_ids_all|p2_ids_all)}): {m_warm_all:.4f}")
    print(f"\n      Cold (n={len(cold_ids)}, global pop): {m_cold:.4f}")
    print(f"\n      Composite = {composite:.4f}")
    print(f"          (0.352 × {m_cold:.4f} + 0.648 × {m_warm_all:.4f})")

    # Разброс по фолдам
    p1_folds = [f["m_p1"] for f in fold_stats if f["n_p1_hold"] > 0]
    p2_folds = [f["m_p2"] for f in fold_stats if f["n_p2_hold"] > 0]
    if p1_folds:
        print(f"\n      Разброс по фолдам (p1): "
              f"std={np.std(p1_folds, ddof=1):.4f}")
    if p2_folds:
        print(f"      Разброс по фолдам (p2): "
              f"std={np.std(p2_folds, ddof=1):.4f}")

    print(f"\n[OK] Evaluator завершён за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()