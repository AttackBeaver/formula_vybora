"""
run_ltr_v40.py — v7 с параметрическими весами p2 и p0mask.

КОНТЕКСТ:
    На новой data (train=5002) работают только модели с p2-срезом:
        v7  (p2=3.0, p0mask=0):   public = 0.3240  ← champion
        v38 (без p2, V12):        public = 0.3168  регресс
        v39 (без p2, V11):        public = 0.3142  регресс

    Значит p2-срез — реальный сигнал, а не утечка. Возвращаемся
    в схему v7 и крутим только веса.

ЧТО ПАРАМЕТРИЗОВАНО:
    --weight-p2 FLOAT       (default 3.0)  — вес p2-среза (80% val_p2)
    --weight-p0mask FLOAT   (default 0.0)  — вес p0_mask-среза (0 = выкл)

    При weight-p2=3.0, weight-p0mask=0.0 — точная репликация v7.

ОЦЕНКА (прямой сигнал на holdout):
    Holdout = 20% val_p2, на которых p2-срез НЕ обучался.
    Это proxy для test_p2/p3 (63% test).
    v7 holdout ≈ 0.51.
    Если p2-вес 5.0 даёт holdout > 0.52 — усиление p2 помогает.

ДОПОЛНИТЕЛЬНО:
    Холод оценивается отдельно (raw и с fallback).
    Если p0mask>0 — смотрим cold raw; fallback всё равно 0.2816.

ЗАПУСК (два прогона A/B):
    python -m src.run_ltr_v40 --weight-p2 3.0   # v7 baseline
    python -m src.run_ltr_v40 --weight-p2 5.0   # усиленный p2
    python -m src.run_ltr_v40 --weight-p2 1.5   # ослабленный p2

    python -m src.run_ltr_v40 --weight-p2 3.0 --weight-p0mask 1.0   # v7+p0mask

РЕШЕНИЕ О САБМИТЕ:
    holdout ≥ 0.520  (v7 + 0.007)  →  кандидат на завтрашний сабмит
    holdout 0.505–0.520            →  шум, не сабмитим
    holdout < 0.505                →  регресс, не сабмитим
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

TEST_SHARES = {"cold": 0.352, "p1": 0.020, "p2": 0.324, "p3": 0.304}

WEIGHT_P1 = 1.0
WEIGHT_P0 = 0.5
WEIGHT_PMASK = 1.0

HOLDOUT_FRACTION = 0.20
HOLDOUT_SEED = 42
BASE_SEED = 42

FEATURE_COLS = FEATURE_COLS_V4
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


def main() -> None:
    ap = argparse.ArgumentParser(description="LTR v40: v7 с весами p2/p0mask")
    ap.add_argument("--weight-p2", type=float, default=3.0)
    ap.add_argument("--weight-p0mask", type=float, default=0.0)
    ap.add_argument("--n-seeds", type=int, default=5)
    args = ap.parse_args()

    weight_p2 = float(args.weight_p2)
    weight_p0mask = float(args.weight_p0mask)
    n_seeds = max(1, int(args.n_seeds))
    seeds = [BASE_SEED + i for i in range(n_seeds)]

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    tag = f"p2w{weight_p2}_p0mask{weight_p0mask}".replace(".", "p")

    print("=" * 78)
    print(f"LTR v40: V4, p2={weight_p2}, p0_mask={weight_p0mask}")
    print(f"  feature_set = V4 ({len(FEATURE_COLS)} признаков)")
    print(f"  content_backend = '{CONTENT_BACKEND}'")
    print(f"  Срезы: p0({WEIGHT_P0}) p1({WEIGHT_P1}) pmask({WEIGHT_PMASK}) "
          f"p2({weight_p2})" + (f" p0_mask({weight_p0mask})"
                                if weight_p0mask > 0 else ""))
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
    print(f"\n[1/8] FeatureStats (content_backend='{CONTENT_BACKEND}')...")
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

    if weight_p0mask > 0:
        train_df_p0mask = build_dataset(
            train_students_p0, stats,
            candidates_fn=make_candidates_train_period1(stats),
            target_fn=target_period_1,
            past_periods=[],
            mask_profile=True,
        )
        parts_log += f"  p0_mask={len(train_df_p0mask):,}"
    else:
        train_df_p0mask = None

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

    # ==== 5. Сборка ====
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
    print("\n[6/8] Подготовка X, y, g, w...")
    X_train, y_train, g_train, w_train, _ = prepare_xygw(
        train_df, feature_cols=FEATURE_COLS)
    X_ho, y_ho, g_ho, w_ho, ho_sorted = prepare_xygw(
        holdout_df, feature_cols=FEATURE_COLS)
    X_val, y_val, g_val, w_val, val_sorted = prepare_xygw(
        val_df, feature_cols=FEATURE_COLS)
    _, _, _, _, test_sorted = prepare_xygw(test_df, feature_cols=FEATURE_COLS)

    print(f"      Train: X={X_train.shape}, групп={len(g_train)}, "
          f"w_mean={w_train.mean():.3f}")
    print(f"      Holdout: X={X_ho.shape}, групп={len(g_ho)}")

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
            predict_scores(model, ho_sorted, feature_cols=FEATURE_COLS))
        scores_val_list.append(
            predict_scores(model, val_sorted, feature_cols=FEATURE_COLS))
        scores_test_list.append(
            predict_scores(model, test_sorted, feature_cols=FEATURE_COLS))
        last_model = model

    scores_ho = np.mean(scores_ho_list, axis=0)
    scores_val = np.mean(scores_val_list, axis=0)
    scores_test = np.mean(scores_test_list, axis=0)

    print(f"\n      best_iter: mean={np.mean(best_iters):.0f}, "
          f"min={min(best_iters)}, max={max(best_iters)}")

    save_model(last_model, MODELS_DIR / f"lgbm_ranker_v40_{tag}.txt")

    # ==== 8. Оценка ====
    print("\n[8/8] Оценка holdout + val:")

    rec_ho = top_k_per_student(ho_sorted, scores_ho, k=5)
    rec_ho_clean = {sid.replace("ho__", ""): v for sid, v in rec_ho.items()}
    gt_ho = {sid: gt for sid, gt in val_gt_raw.items() if sid in holdout_ids}
    mean_ho, _ = evaluate_pndcg5(rec_ho_clean, gt_ho, k=5)
    print(f"      Holdout (n={len(holdout_records)}): {mean_ho:.4f}")
    print(f"      --- ориентир v7: 0.5134 (старая data), ≈0.51 (новая)")

    cold_ids_val = identify_cold_students(val)
    rec_val_ltr = top_k_per_student(val_sorted, scores_val, k=5)
    mean_ltr, cold_ltr, warm_ltr = _segment_means(
        rec_val_ltr, val_gt_raw, cold_ids_val)
    print(f"\n      val LTR raw: mean={mean_ltr:.4f}  cold={cold_ltr:.4f}  "
          f"warm={warm_ltr:.4f}")

    pop_weighted, _ = compute_popularity(train)
    fb_val = build_global_recommendations(val, pop_weighted, cold_ids_val, k=5)
    rec_val_fb = apply_fallback(rec_val_ltr, fb_val, k=5)
    mean_val, cold_val, warm_val = _segment_means(
        rec_val_fb, val_gt_raw, cold_ids_val)

    # Сегментный composite
    val_p1_ids = {s["id"] for s in val
                  if s["id"] not in cold_ids_val
                  and len(s.get("history") or []) == 1}
    val_p2_ids = {s["id"] for s in val
                  if s["id"] not in cold_ids_val
                  and len(s.get("history") or []) >= 2}
    _, per_student_fb = evaluate_pndcg5(rec_val_fb, val_gt_raw, k=5)
    m_p1_fb = float(np.mean([per_student_fb[sid] for sid in val_p1_ids
                             if sid in per_student_fb]))
    m_p2_fb = float(np.mean([per_student_fb[sid] for sid in val_p2_ids
                             if sid in per_student_fb]))
    composite_fb = (
        TEST_SHARES["cold"] * cold_val
        + TEST_SHARES["p1"] * m_p1_fb
        + (TEST_SHARES["p2"] + TEST_SHARES["p3"]) * m_p2_fb
    )

    print(f"      val + fallback: mean={mean_val:.4f}  cold={cold_val:.4f}  "
          f"p1={m_p1_fb:.4f}  p2={m_p2_fb:.4f}")
    print(f"      composite (0.352·cold + 0.020·p1 + 0.628·p2) = {composite_fb:.4f}")

    val_path = SUBMISSIONS_DIR / f"val_submission_ltr_v40_{tag}.csv"
    write_submission(rec_val_fb, val_path)

    rec_test_ltr = top_k_per_student(test_sorted, scores_test, k=5)
    cold_ids_test = identify_cold_students(test)
    fb_test = build_global_recommendations(test, pop_weighted, cold_ids_test, k=5)
    rec_test_final = apply_fallback(rec_test_ltr, fb_test, k=5)

    test_path = SUBMISSIONS_DIR / f"submission_v40_{tag}.csv"
    write_submission(rec_test_final, test_path)
    print(f"\n      Cold в test: {len(cold_ids_test)}")
    print(f"      Сохранено: {test_path}  ({len(rec_test_final)} строк)")

    print("\n[Feature importance — top-12]")
    imp = feature_importance_report(last_model, feature_cols=FEATURE_COLS, top_n=12)
    for _, row in imp.iterrows():
        print(f"      {row['feature']:<28}  gain = {row['gain']:>12.0f}")

    # ==== Вердикт ====
    print("\n" + "=" * 78)
    print(f"ИТОГ: p2={weight_p2}, p0mask={weight_p0mask}")
    print("=" * 78)
    print(f"  Holdout            = {mean_ho:.4f}  (v7 ориентир ≈ 0.51)")
    print(f"  val composite (fb) = {composite_fb:.4f}")
    print(f"  cold raw           = {cold_ltr:.4f}  (fallback = {cold_val:.4f})")
    print(f"  warm raw           = {warm_ltr:.4f}")
    print()
    if mean_ho >= 0.520:
        print(f"  РЕКОМЕНДАЦИЯ: holdout {mean_ho:.4f} ≥ 0.520 — кандидат на сабмит")
    elif mean_ho >= 0.505:
        print(f"  ОСТОРОЖНО: holdout {mean_ho:.4f} ∈ [0.505, 0.520) — шум")
    else:
        print(f"  НЕ САБМИТИМ: holdout {mean_ho:.4f} < 0.505 — регресс")
    print("=" * 78)
    print(f"\n[OK] v40 завершён за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()