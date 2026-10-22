"""
run_ltr_v42.py — эксперименты в V11-семье с выбором feature-set.

ОСИ:
    --feature-set NAME        любой из FEATURE_SETS (v4, v5, v6, v7, v8,
                              v11, v12, v14, v15, v16)
    --weight-p0mask FLOAT     вес p0_mask-среза (default 0.0 = как v41)
                              ВНИМАНИЕ: p0mask на cold ломает test (v43).
                              Использовать только для warm-экспериментов.
    --weight-p2 FLOAT         вес p2-среза (default 3.0 = как v41)
    --drop-global-rank-in-avail  исключить pop_rank_global_in_avail

ЗАПУСК:
    # контроль: v41-реплика
    python -m src.run_ltr_v42 --feature-set v11 --n-seeds 5
    # ablation: без pop_rank_global_frac
    python -m src.run_ltr_v42 --feature-set v15 --n-seeds 3
    # ablation: без title_len/desc_len
    python -m src.run_ltr_v42 --feature-set v16 --n-seeds 3

РЕШЕНИЕ О САБМИТЕ (raw-LTR на cold, без fallback):
    cold raw ≥ 0.337 (не хуже v41) И warm raw ≥ 0.487 (лучше v41)  →  сабмит
    cold raw ≥ 0.337 И warm raw ∈ [0.483, 0.487]                  →  на грани
    cold raw < 0.337 или warm raw < 0.483                         →  не сабмит
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
    COMPATIBLE_FEATURE_SETS,
    FEATURE_SETS,
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

WEIGHT_P1 = 1.0
WEIGHT_P0 = 0.5
WEIGHT_PMASK = 1.0

HOLDOUT_FRACTION = 0.20
HOLDOUT_SEED = 42
BASE_SEED = 42

CONTENT_BACKEND = "tfidf"

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


def _validate_combo(content_backend: str, feature_set: str) -> None:
    if feature_set not in FEATURE_SETS:
        raise ValueError(
            f"feature_set={feature_set!r}, ожидалось одно из "
            f"{sorted(FEATURE_SETS)}"
        )
    allowed = COMPATIBLE_FEATURE_SETS.get(content_backend, ())
    if feature_set not in allowed:
        raise ValueError(
            f"feature_set={feature_set!r} требует content_backend ∈ "
            f"{sorted(allowed)}, получено {content_backend!r}."
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="LTR v42: V11-семья, feature-set + p0mask/p2/drop"
    )
    ap.add_argument(
        "--feature-set", choices=tuple(FEATURE_SETS.keys()), default="v11",
        help="v11 = V4 + глобальные rank (champion v41); "
             "v15 = V11 − pop_rank_global_frac; "
             "v16 = V11 − title_len − desc_len; "
             "v14 = V11 + pop_delta",
    )
    ap.add_argument("--weight-p0mask", type=float, default=0.0)
    ap.add_argument("--weight-p2", type=float, default=3.0)
    ap.add_argument("--drop-global-rank-in-avail", action="store_true")
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()

    try:
        _validate_combo(CONTENT_BACKEND, args.feature_set)
    except ValueError as e:
        print(f"ERROR: {e}")
        raise SystemExit(2)

    feature_set = args.feature_set
    weight_p0mask = float(args.weight_p0mask)
    weight_p2 = float(args.weight_p2)
    drop_gra = bool(args.drop_global_rank_in_avail)
    n_seeds = max(1, int(args.n_seeds))
    seeds = [BASE_SEED + i for i in range(n_seeds)]

    feature_cols = list(FEATURE_SETS[feature_set])
    if drop_gra:
        feature_cols = [c for c in feature_cols if c != "pop_rank_global_in_avail"]

    tag_parts = [feature_set, f"p2w{weight_p2}", f"p0mask{weight_p0mask}"]
    if drop_gra:
        tag_parts.append("noGRA")
    tag = "_".join(p.replace(".", "p") for p in tag_parts)

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print(f"LTR v42: feature={feature_set}"
          f"{' − pop_rank_global_in_avail' if drop_gra else ''}")
    print(f"  {len(feature_cols)} признаков, content='{CONTENT_BACKEND}'")
    print(f"  Срезы: p0({WEIGHT_P0}) p1({WEIGHT_P1}) pmask({WEIGHT_PMASK}) "
          f"p2({weight_p2})" +
          (f" p0_mask({weight_p0mask})" if weight_p0mask > 0 else ""))
    print(f"  Seeds: {seeds}")
    print(f"  tag = '{tag}'")
    print("=" * 78)

    t0 = time.time()
    data = load_all()
    train = data["train"]
    val = data["val"]
    test = data["test"]
    modules = data["modules"]
    val_solution = data["val_solution"]
    val_gt_raw = dict(zip(val_solution["id"], val_solution["ground_truth"]))

    # ==== 1. FeatureStats ====
    print(f"\n[1/8] FeatureStats (content='{CONTENT_BACKEND}')...")
    stats = FeatureStats.fit(train, modules, content_backend=CONTENT_BACKEND)
    stats.add_val_solution_popularity(val_solution)

    # ==== 2. Train-срезы ====
    print("\n[2/8] Train-срезы...")
    train_students_p1 = [
        s for s in train
        if any(h.get("period") == 2 for h in s.get("history", []))
    ]
    train_students_p0 = [
        s for s in train
        if any(h.get("period") == 1 for h in s.get("history", []))
    ]

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
    parts_log = (f"p1={len(train_df_p1):,}  p0={len(train_df_p0):,}  "
                 f"pmask={len(train_df_pmask):,}")

    train_df_p0mask = None
    if weight_p0mask > 0:
        train_df_p0mask = build_dataset(
            train_students_p0, stats,
            candidates_fn=make_candidates_train_period1(stats),
            target_fn=target_period_1,
            past_periods=[],
            mask_profile=True,
        )
        parts_log += f"  p0_mask={len(train_df_p0mask):,}"
    print(f"      {parts_log}")

    # ==== 3. Holdout + p2 ====
    print("\n[3/8] Holdout + p2-срез...")
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
        max_history_periods=2,
    )
    print(f"      val={len(val_df):,}  test={len(test_df):,}")

    # ==== 5. Сборка + кодирование ====
    print("\n[5/8] Сборка + единый словарь категорий...")
    train_df_p1 = _add_weight(train_df_p1, WEIGHT_P1)
    train_df_p0 = _add_weight(train_df_p0, WEIGHT_P0)
    train_df_pmask = _add_weight(train_df_pmask, WEIGHT_PMASK)
    p2_df = _add_weight(p2_df, weight_p2)
    holdout_df = _add_weight(holdout_df, 1.0)
    val_df = _add_weight(val_df, 1.0)
    test_df = _add_weight(test_df, 1.0)

    train_df_p1["student_id"] = "p1__" + train_df_p1["student_id"].astype(str)
    train_df_p0["student_id"] = "p0__" + train_df_p0["student_id"].astype(str)
    train_df_pmask["student_id"] = "pmask__" + train_df_pmask["student_id"].astype(str)
    p2_df["student_id"] = "p2__" + p2_df["student_id"].astype(str)
    holdout_df["student_id"] = "ho__" + holdout_df["student_id"].astype(str)

    parts = [train_df_p1, train_df_p0, train_df_pmask, p2_df]
    if train_df_p0mask is not None:
        train_df_p0mask = _add_weight(train_df_p0mask, weight_p0mask)
        train_df_p0mask["student_id"] = (
            "p0mask__" + train_df_p0mask["student_id"].astype(str)
        )
        parts.append(train_df_p0mask)

    train_df = pd.concat(parts, ignore_index=True)

    n_tr = len(train_df)
    n_ho = len(holdout_df)
    n_va = len(val_df)
    all_df = pd.concat([train_df, holdout_df, val_df, test_df], ignore_index=True)
    all_df, _ = encode_categoricals(all_df, all_df, CATEGORICAL_FEATS)

    train_df = all_df.iloc[:n_tr].reset_index(drop=True)
    holdout_df = all_df.iloc[n_tr:n_tr + n_ho].reset_index(drop=True)
    val_df = all_df.iloc[n_tr + n_ho:n_tr + n_ho + n_va].reset_index(drop=True)
    test_df = all_df.iloc[n_tr + n_ho + n_va:].reset_index(drop=True)

    # ==== 6. Матрицы ====
    print("\n[6/8] Матрицы X, y, g, w...")
    X_train, y_train, g_train, w_train, _ = prepare_xygw(
        train_df, feature_cols=feature_cols)
    X_ho, y_ho, g_ho, w_ho, ho_sorted = prepare_xygw(
        holdout_df, feature_cols=feature_cols)
    X_val, y_val, g_val, w_val, val_sorted = prepare_xygw(
        val_df, feature_cols=feature_cols)
    _, _, _, _, test_sorted = prepare_xygw(test_df, feature_cols=feature_cols)
    print(f"      Train X={X_train.shape}, групп={len(g_train)}, "
          f"w_mean={w_train.mean():.3f}")

    # ==== 7. Multi-seed ====
    print(f"\n[7/8] Обучение {n_seeds} моделей...")
    scores_ho_list: list[np.ndarray] = []
    scores_val_list: list[np.ndarray] = []
    scores_test_list: list[np.ndarray] = []
    best_iters: list[int] = []
    last_model: Any = None

    for i, seed in enumerate(seeds):
        print(f"\n      --- Модель {i + 1}/{n_seeds}, seed={seed} ---")
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
        print(f"      seed={seed} за {time.time() - t_train:.1f} сек, "
              f"best_iter={model.best_iteration}")

        scores_ho_list.append(
            predict_scores(model, ho_sorted, feature_cols=feature_cols))
        scores_val_list.append(
            predict_scores(model, val_sorted, feature_cols=feature_cols))
        scores_test_list.append(
            predict_scores(model, test_sorted, feature_cols=feature_cols))
        last_model = model

    scores_ho = np.mean(scores_ho_list, axis=0)
    scores_val = np.mean(scores_val_list, axis=0)
    scores_test = np.mean(scores_test_list, axis=0)
    print(f"\n      best_iter mean={np.mean(best_iters):.0f}, "
          f"min={min(best_iters)}, max={max(best_iters)}")

    model_name = f"lgbm_ranker_v42_{tag}.txt"
    save_model(last_model, MODELS_DIR / model_name)

    # ==== 8. Оценка ====
    print("\n[8/8] Оценка...")

    rec_ho = top_k_per_student(ho_sorted, scores_ho, k=5)
    rec_ho_clean = {sid.replace("ho__", ""): v for sid, v in rec_ho.items()}
    gt_ho = {sid: gt for sid, gt in val_gt_raw.items() if sid in holdout_ids}
    mean_ho, _ = evaluate_pndcg5(rec_ho_clean, gt_ho, k=5)
    print(f"      Holdout (n={len(holdout_records)}): {mean_ho:.4f}")

    cold_ids_val = identify_cold_students(val)
    rec_val_ltr = top_k_per_student(val_sorted, scores_val, k=5)
    mean_ltr, cold_ltr, warm_ltr = _segment_means(
        rec_val_ltr, val_gt_raw, cold_ids_val)
    print(f"      val raw: cold={cold_ltr:.4f}  warm={warm_ltr:.4f}")

    val_path = SUBMISSIONS_DIR / f"val_submission_ltr_v42_{tag}_raw.csv"
    write_submission(rec_val_ltr, val_path)

    # Test raw (БЕЗ fallback, как v41 champion)
    rec_test_ltr = top_k_per_student(test_sorted, scores_test, k=5)
    cold_ids_test = identify_cold_students(test)
    test_path = SUBMISSIONS_DIR / f"submission_v42_{tag}_raw.csv"
    write_submission(rec_test_ltr, test_path)
    print(f"\n      Cold в test: {len(cold_ids_test)}")
    print(f"      Сохранено: {test_path}")

    print("\n[Feature importance — top-15]")
    imp = feature_importance_report(last_model, feature_cols=feature_cols, top_n=15)
    for _, row in imp.iterrows():
        print(f"      {row['feature']:<30}  gain = {row['gain']:>12.0f}")

    # ==== Вердикт ====
    print("\n" + "=" * 78)
    print(f"ИТОГ: feature={feature_set}, p2={weight_p2}, p0mask={weight_p0mask}")
    print("=" * 78)
    print(f"  Holdout      = {mean_ho:.4f}  (v41: 0.5007)")
    print(f"  cold raw     = {cold_ltr:.4f}  (v41: 0.3372)")
    print(f"  warm raw     = {warm_ltr:.4f}  (v41: 0.4839)")
    print()

    if cold_ltr >= 0.337 and warm_ltr >= 0.487:
        print(f"  ✅ РЕКОМЕНДАЦИЯ: cold {cold_ltr:.4f}≥0.337, warm "
              f"{warm_ltr:.4f}≥0.487 — КАНДИДАТ на сабмит")
        print(f"     файл: {test_path.name}")
    elif cold_ltr >= 0.337 and warm_ltr >= 0.483:
        print(f"  ⚠️  ОСТОРОЖНО: cold {cold_ltr:.4f}≥0.337, warm "
              f"{warm_ltr:.4f} ∈ [0.483, 0.487) — на грани")
    else:
        print(f"  ❌ НЕ САБМИТИМ: cold {cold_ltr:.4f} или warm {warm_ltr:.4f} "
              f"не дотягивают")
    print("=" * 78)
    print(f"\n[OK] завершён за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()