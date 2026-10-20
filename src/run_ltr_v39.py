"""
run_ltr_v39.py — V11 + p0_mask + fallback, без p2-среза (honest).

ПОЧЕМУ V11:
    eval_clean (без p0mask):
        v4  composite = 0.3203  (cold=0.2239, p2=0.3733)
        v11 composite = 0.3275  (cold=0.2403, p2=0.3757)   ← +0.0072
    eval_clean_p0mask (с p0mask=3.0):
        v4  composite = 0.3241  (cold=0.2232, p2=0.3796)
        v11 composite = 0.3331  (cold=0.2409, p2=0.3841)   ← +0.0090

    V11 выигрывает у V4 и на cold, и на warm. Это единственная
    конфигурация за сессию с положительным Δ на обоих сегментах
    одновременно.

ПОЧЕМУ НЕ V12:
    V12 (без rank-3) на honest валится: cold=0.1194 (регресс −0.10).
    Rank-3 — реальный сигнал (v28 public: −0.025 при удалении).
    V11 = V4 + 3 глобальные rank, БЕЗ удаления rank-3.

ПОЧЕМУ БЕЗ p2:
    p2-срез = 80% val_p2 в обучении → in-sample bias на val.
    v37 (V12+p0mask+p2) дал honest=0.42, public=0.29 — классический
    overfit. v39 обучается честно, оценка совпадает с public.

ПРОГНОЗ PUBLIC:
    Оффсет honest → public на новой data ≈ 0 (v7: honest 0.3241 → public 0.3240).
    Композит honest для v39 ≈ 0.3331 + fallbackΔ ≈ 0.347.
    Если оффсет сохранится — public ≈ 0.347.
    Если оффсет сместится (как v38: −0.036) — public ≈ 0.311.
    Реалистичный диапазон: 0.325 – 0.350.

ЗАПУСК:
    python -m src.run_ltr_v39
    python -m src.run_ltr_v39 --weight-p0mask 5.0   # A/B
    python -m src.run_ltr_v39 --n-seeds 3            # быстрее
"""
from __future__ import annotations

import argparse
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
)
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
    FEATURE_COLS_V11,
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

TEST_SHARES = {"cold": 0.352, "p1": 0.020, "p2": 0.324, "p3": 0.304}

WEIGHT_P1 = 1.0
WEIGHT_PMASK = 1.0

ES_FRACTION = 0.10
ES_SEED = 42
BASE_SEED = 42

FEATURE_COLS = FEATURE_COLS_V11
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


def _add_weight(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = np.float32(weight)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="LTR v39: V11 + p0mask + fallback")
    parser.add_argument("--weight-p0mask", type=float, default=3.0)
    parser.add_argument("--weight-p0", type=float, default=0.3)
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()

    weight_p0mask = float(args.weight_p0mask)
    weight_p0 = float(args.weight_p0)
    n_seeds = max(1, int(args.n_seeds))
    seeds = [BASE_SEED + i for i in range(n_seeds)]

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    tag = f"v11_p0mask{weight_p0mask}_p0{weight_p0}".replace(".", "p")

    print("=" * 78)
    print(f"LTR v39: V11 + p0_mask({weight_p0mask}) + fallback (n_seeds={n_seeds})")
    print(f"  feature_set = V11 ({len(FEATURE_COLS)} признаков)")
    print(f"  content_backend = '{CONTENT_BACKEND}'")
    print(f"  Срезы: p0({weight_p0}) p1({WEIGHT_P1}) pmask({WEIGHT_PMASK}) "
          f"p0_mask({weight_p0mask})  [БЕЗ p2]")
    print(f"  Инференс: LTR + fallback на cold")
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

    # ==== 1. ES-split ====
    print(f"\n[1/8] ES-split по train (fraction={ES_FRACTION})...")
    rng = random.Random(ES_SEED)
    train_shuffled = list(train)
    rng.shuffle(train_shuffled)
    n_es = int(len(train_shuffled) * ES_FRACTION)
    es_students = train_shuffled[:n_es]
    core_students = train_shuffled[n_es:]
    print(f"      core={len(core_students)}, es={len(es_students)}")

    # ==== 2. FeatureStats ====
    print(f"\n[2/8] FeatureStats (content_backend='{CONTENT_BACKEND}')...")
    stats = FeatureStats.fit(train, modules, content_backend=CONTENT_BACKEND)

    # ==== 3. Train-срезы ====
    print("\n[3/8] Train-срезы p0, p1, pmask, p0_mask, ES...")
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
    train_df_p0mask = build_dataset(
        core_p0, stats,
        candidates_fn=make_candidates_train_period1(stats),
        target_fn=target_period_1,
        past_periods=[],
        mask_profile=True,
    )
    es_df = build_dataset(
        es_students, stats,
        candidates_fn=make_candidates_train_period2(stats),
        target_fn=target_period_2,
        past_periods=[1],
    )
    print(f"      p1={len(train_df_p1):,}  p0={len(train_df_p0):,}  "
          f"pmask={len(train_df_pmask):,}  p0_mask={len(train_df_p0mask):,}  "
          f"es={len(es_df):,}")

    # ==== 4. Val и test ====
    print("\n[4/8] Val и test датасеты...")
    val_df = build_dataset(
        val, stats,
        candidates_fn=candidates_available,
        target_fn=None,
        past_periods=None,
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
    train_df_p0 = _add_weight(train_df_p0, weight_p0)
    train_df_pmask = _add_weight(train_df_pmask, WEIGHT_PMASK)
    train_df_p0mask = _add_weight(train_df_p0mask, weight_p0mask)
    es_df = _add_weight(es_df, 1.0)
    val_df = _add_weight(val_df, 1.0)
    test_df = _add_weight(test_df, 1.0)

    train_df_p1["student_id"] = "p1__" + train_df_p1["student_id"].astype(str)
    train_df_p0["student_id"] = "p0__" + train_df_p0["student_id"].astype(str)
    train_df_pmask["student_id"] = "pmask__" + train_df_pmask["student_id"].astype(str)
    train_df_p0mask["student_id"] = "p0mask__" + train_df_p0mask["student_id"].astype(str)
    es_df["student_id"] = "es__" + es_df["student_id"].astype(str)

    train_df = pd.concat(
        [train_df_p1, train_df_p0, train_df_pmask, train_df_p0mask],
        ignore_index=True,
    )

    n_tr = len(train_df)
    n_es = len(es_df)
    n_va = len(val_df)
    all_df = pd.concat([train_df, es_df, val_df, test_df], ignore_index=True)
    all_df, _ = encode_categoricals(all_df, all_df, CATEGORICAL_FEATS)

    train_df = all_df.iloc[:n_tr].reset_index(drop=True)
    es_df = all_df.iloc[n_tr:n_tr + n_es].reset_index(drop=True)
    val_df = all_df.iloc[n_tr + n_es:n_tr + n_es + n_va].reset_index(drop=True)
    test_df = all_df.iloc[n_tr + n_es + n_va:].reset_index(drop=True)

    # ==== 6. Матрицы ====
    print("\n[6/8] Подготовка X, y, g, w...")
    X_train, y_train, g_train, w_train, _ = prepare_xygw(
        train_df, feature_cols=FEATURE_COLS)
    X_es, y_es, g_es, w_es, es_sorted = prepare_xygw(
        es_df, feature_cols=FEATURE_COLS)
    X_val, y_val, g_val, w_val, val_sorted = prepare_xygw(
        val_df, feature_cols=FEATURE_COLS)
    _, _, _, _, test_sorted = prepare_xygw(test_df, feature_cols=FEATURE_COLS)

    print(f"      Train: X={X_train.shape}, групп={len(g_train)}, "
          f"w_mean={w_train.mean():.3f}")
    print(f"      ES:    X={X_es.shape},   групп={len(g_es)}")

    # ==== 7. Multi-seed ====
    print(f"\n[7/8] Обучение {n_seeds} моделей...")
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
            X_es, y_es, g_es,
            params=params,
            num_boost_round=NUM_BOOST_ROUND,
            early_stopping_rounds=EARLY_STOPPING,
            log_every=500,
            sample_weight_train=w_train,
            sample_weight_val=w_es,
        )
        best_iters.append(model.best_iteration)
        print(f"      seed={seed} за {time.time() - t_train:.1f} сек, "
              f"best_iter={model.best_iteration}")

        scores_val_list.append(
            predict_scores(model, val_sorted, feature_cols=FEATURE_COLS))
        scores_test_list.append(
            predict_scores(model, test_sorted, feature_cols=FEATURE_COLS))
        last_model = model

    scores_val = np.mean(scores_val_list, axis=0)
    scores_test = np.mean(scores_test_list, axis=0)

    print(f"\n      best_iter: mean={np.mean(best_iters):.0f}, "
          f"min={min(best_iters)}, max={max(best_iters)}")

    save_model(last_model, MODELS_DIR / f"lgbm_ranker_v39_{tag}.txt")

    # ==== 8. Оценка val + инференс test ====
    print("\n[8/8] Оценка val + инференс test...")

    cold_ids_val = identify_cold_students(val)
    val_p1_ids = {s["id"] for s in val
                  if s["id"] not in cold_ids_val
                  and len(s.get("history") or []) == 1}
    val_p2_ids = {s["id"] for s in val
                  if s["id"] not in cold_ids_val
                  and len(s.get("history") or []) >= 2}

    rec_val_ltr = top_k_per_student(val_sorted, scores_val, k=5)
    mean_ltr, per_student = evaluate_pndcg5(rec_val_ltr, val_gt_raw, k=5)

    cold_scores = [per_student[sid] for sid in cold_ids_val if sid in per_student]
    warm_scores = [per_student[sid] for sid in per_student
                   if sid not in cold_ids_val]
    m_cold_ltr = float(np.mean(cold_scores)) if cold_scores else 0.0
    m_warm_ltr = float(np.mean(warm_scores)) if warm_scores else 0.0
    print(f"      val LTR raw: cold={m_cold_ltr:.4f}  warm={m_warm_ltr:.4f}")

    # Fallback
    pop_weighted, _ = compute_popularity(train)
    fb_val = build_global_recommendations(val, pop_weighted, cold_ids_val, k=5)
    rec_val_fb = apply_fallback(rec_val_ltr, fb_val, k=5)
    mean_fb, per_student_fb = evaluate_pndcg5(rec_val_fb, val_gt_raw, k=5)

    m_cold_fb = float(np.mean([per_student_fb[sid] for sid in cold_ids_val
                               if sid in per_student_fb]))
    m_p1_fb = float(np.mean([per_student_fb[sid] for sid in val_p1_ids
                             if sid in per_student_fb]))
    m_p2_fb = float(np.mean([per_student_fb[sid] for sid in val_p2_ids
                             if sid in per_student_fb]))

    composite_fb = (
        TEST_SHARES["cold"] * m_cold_fb
        + TEST_SHARES["p1"] * m_p1_fb
        + (TEST_SHARES["p2"] + TEST_SHARES["p3"]) * m_p2_fb
    )

    print(f"      val + fallback: cold={m_cold_fb:.4f}  p1={m_p1_fb:.4f}  "
          f"p2={m_p2_fb:.4f}")
    print(f"      mean (all) = {mean_fb:.4f}")
    print(f"\n      composite (0.352·cold + 0.020·p1 + 0.628·p2) = {composite_fb:.4f}")

    print(f"\n      Ориентиры:")
    print(f"        public v7 (факт, новая data)  = 0.3240")
    print(f"        public v38 (V12+p0mask)       = 0.3168")
    print(f"        honest v11+p0mask (без fb)    = 0.3331")
    print(f"        public v39 прогноз            ≈ {composite_fb + 0.000:.4f} ± 0.015")

    val_path = SUBMISSIONS_DIR / f"val_submission_ltr_v39_{tag}.csv"
    write_submission(rec_val_fb, val_path)

    # Test
    rec_test_ltr = top_k_per_student(test_sorted, scores_test, k=5)
    cold_ids_test = identify_cold_students(test)
    fb_test = build_global_recommendations(test, pop_weighted, cold_ids_test, k=5)
    rec_test_final = apply_fallback(rec_test_ltr, fb_test, k=5)

    test_path = SUBMISSIONS_DIR / f"submission_v39_{tag}.csv"
    write_submission(rec_test_final, test_path)
    print(f"\n      Cold в test: {len(cold_ids_test)}")
    print(f"      Сохранено: {test_path}  ({len(rec_test_final)} строк)")

    print("\n[Feature importance — top-15]")
    imp = feature_importance_report(last_model, feature_cols=FEATURE_COLS, top_n=15)
    for _, row in imp.iterrows():
        print(f"      {row['feature']:<28}  gain = {row['gain']:>12.0f}")

    print(f"\n[OK] v39 завершён за {time.time() - t0:.1f} сек.")

    # Вердикт
    print("\n" + "=" * 78)
    if composite_fb >= 0.345:
        print(f"РЕКОМЕНДАЦИЯ: composite={composite_fb:.4f} ≥ 0.345 — "
              f"сабмитим {test_path.name}")
    elif composite_fb >= 0.333:
        print(f"ОСТОРОЖНО: composite={composite_fb:.4f} ∈ [0.333, 0.345) — "
              f"сравнить с honest v11+p0mask (0.3331), может быть +0")
    else:
        print(f"ВНИМАНИЕ: composite={composite_fb:.4f} < 0.333 — "
              f"не сабмитим")
    print("=" * 78)


if __name__ == "__main__":
    main()