"""
LTR v5: переход от transition diagnostic к корректному обучению.

Отличия от transition_diagnostic:
    * Transition-матрица строится из train + val_minus_holdout.
      Holdout-студенты исключены — иначе их собственный target попадает
      в transition-матрицу, и фичи transition_from_last/prev дают утечку.
    * Используется FEATURE_COLS_V5 (34 признака: 31 из v4 + 3 transition).
    * Финальный val/test инференс использует transition из train + полного
      val — это корректно, потому что для test val_solution нам известен.

Запуск:
    python -m src.run_ltr_v5
"""
from __future__ import annotations

import random
import time
from collections import Counter, defaultdict
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
from src.metric import evaluate_pndcg5, parse_ground_truth
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

# Итоговый набор для v5.
FEATURE_COLS_V5: list[str] = FEATURE_COLS_V4 + [
    "transition_from_last_sum",
    "transition_from_last_max",
    "transition_from_prev_sum",
]


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
# Transition-матрица
# ---------------------------------------------------------------------------
def build_transition_matrix(
    students: list[dict[str, Any]],
    val_gt: dict[str, str] | None = None,
) -> dict[str, Counter]:
    """
    Матрица переходов между соседними периодами.

    Для train-студента с историей [P1, P2]:
        пары (m1 ∈ P1, m2 ∈ P2).
    Для val-студента с историей [P1] и target (val_solution):
        пары (m1 ∈ P1, m2 ∈ target).
    Для val-студента с историей [P1, P2] и target:
        пары (m2 ∈ P2, m3 ∈ target).

    Возвращает trans[X][Y] = count.
    """
    trans: dict[str, Counter] = defaultdict(Counter)

    for s in students:
        history = s.get("history") or []
        if not history:
            continue

        last_mods = [c["module_id"] for c in history[-1].get("choices", [])]
        prev_mods: list[str] = []
        if len(history) >= 2:
            prev_mods = [c["module_id"] for c in history[-2].get("choices", [])]

        target_mods: list[str] = []
        if val_gt is not None:
            sid = s.get("id") or s.get("student_id")
            gt_str = val_gt.get(sid) if sid else None
            if gt_str:
                gt = parse_ground_truth(gt_str)
                target_mods = list(gt.keys())

        pairs: list[tuple[list[str], list[str]]] = []
        if prev_mods and last_mods:
            pairs.append((prev_mods, last_mods))
        if target_mods and last_mods:
            pairs.append((last_mods, target_mods))
        elif target_mods and prev_mods:
            pairs.append((prev_mods, target_mods))

        for sources, targets in pairs:
            for m1 in sources:
                for m2 in targets:
                    if m1 != m2:
                        trans[m1][m2] += 1

    return dict(trans)


def attach_transition_features(
    df: pd.DataFrame,
    students: list[dict[str, Any]],
    trans: dict[str, Counter],
) -> pd.DataFrame:
    """
    Добавляет transition-фичи к df. Префиксы p1__ / ho__ автоматически
    срезаются при поиске студента по id.
    """
    by_id: dict[str, tuple[list[str], list[str]]] = {}
    for s in students:
        sid = s.get("id") or s.get("student_id")
        if sid is None:
            continue
        history = s.get("history") or []
        last_mods = [c["module_id"] for c in history[-1].get("choices", [])] if history else []
        prev_mods = (
            [c["module_id"] for c in history[-2].get("choices", [])]
            if len(history) >= 2 else []
        )
        by_id[sid] = (last_mods, prev_mods)

    def clean_id(raw: str) -> str:
        return raw.split("__", 1)[1] if "__" in raw else raw

    student_ids = df["student_id"].values
    module_ids = df["module_id"].values
    n = len(df)

    tls = np.zeros(n, dtype=np.float32)
    tlm = np.zeros(n, dtype=np.float32)
    tps = np.zeros(n, dtype=np.float32)

    for i in range(n):
        sid = clean_id(str(student_ids[i]))
        mid = str(module_ids[i])
        last_mods, prev_mods = by_id.get(sid, ([], []))

        s_last = 0
        m_last = 0
        for h in last_mods:
            v = trans.get(h, Counter()).get(mid, 0)
            s_last += v
            if v > m_last:
                m_last = v
        s_prev = 0
        for h in prev_mods:
            s_prev += trans.get(h, Counter()).get(mid, 0)

        tls[i] = s_last
        tlm[i] = m_last
        tps[i] = s_prev

    out = df.copy()
    out["transition_from_last_sum"] = tls
    out["transition_from_last_max"] = tlm
    out["transition_from_prev_sum"] = tps
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print("LTR v5: transition-фичи (честный holdout, без утечки)")
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
    print("\n[1/8] Статистики + transition-матрица (для holdout — без утечки)...")
    stats = FeatureStats.fit(train, modules)
    stats.add_val_solution_popularity(val_solution)

    # Сначала определяем holdout, чтобы исключить его из transition.
    val_p2 = [s for s in val if _n_periods(s) == 2]
    rng = random.Random(HOLDOUT_SEED)
    shuffled = list(val_p2)
    rng.shuffle(shuffled)
    n_holdout = int(len(shuffled) * HOLDOUT_FRACTION)
    holdout_records = shuffled[:n_holdout]
    p2_records = shuffled[n_holdout:]
    holdout_ids = {s["id"] for s in holdout_records}

    # Transition для обучения: train (P1→P2) + val_minus_holdout (последний→target)
    val_for_trans = [s for s in val if s["id"] not in holdout_ids]
    trans_train = build_transition_matrix(train, val_gt=None)
    trans_val_not_ho = build_transition_matrix(val_for_trans, val_gt=val_gt_raw)

    trans_no_ho: dict[str, Counter] = defaultdict(Counter)
    for src, cnt in trans_train.items():
        for dst, c in cnt.items():
            trans_no_ho[src][dst] += c
    for src, cnt in trans_val_not_ho.items():
        for dst, c in cnt.items():
            trans_no_ho[src][dst] += c
    trans_no_ho = dict(trans_no_ho)

    # Transition для финального val/test: train + полный val (без исключений)
    trans_val_full = build_transition_matrix(val, val_gt=val_gt_raw)
    trans_full: dict[str, Counter] = defaultdict(Counter)
    for src, cnt in trans_train.items():
        for dst, c in cnt.items():
            trans_full[src][dst] += c
    for src, cnt in trans_val_full.items():
        for dst, c in cnt.items():
            trans_full[src][dst] += c
    trans_full = dict(trans_full)

    print(f"      Transition без holdout: {sum(len(v) for v in trans_no_ho.values())} рёбер")
    print(f"      Transition с полным val: {sum(len(v) for v in trans_full.values())} рёбер")

    # ==== 2. Train-срезы ====
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

    # ==== 3. p2-срез и holdout ====
    print("\n[3/8] p2-срез + holdout...")

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
    print(f"      p2={len(p2_df):,}  holdout={len(holdout_df):,}")

    # ==== 4. Применяем transition (без утечки) к обучающим срезам + holdout ====
    print("\n[4/8] Применяю transition-фичи (без holdout в матрице)...")

    train_df_p1 = attach_transition_features(train_df_p1, train_students_p1, trans_no_ho)
    train_df_p0 = attach_transition_features(
        train_df_p0,
        [s for s in train if any(h.get("period") == 1 for h in s.get("history", []))],
        trans_no_ho,
    )
    train_df_pmask = attach_transition_features(train_df_pmask, train_students_p1, trans_no_ho)
    p2_df = attach_transition_features(p2_df, p2_records, trans_no_ho)
    holdout_df = attach_transition_features(holdout_df, holdout_records, trans_no_ho)
    print(f"      p1 zeros: {int((train_df_p1['transition_from_last_sum'] == 0).sum()):,}")
    print(f"      holdout zeros: {int((holdout_df['transition_from_last_sum'] == 0).sum()):,}")

    # ==== 5. Val и test с transition из полного val ====
    print("\n[5/8] Val и test с transition из полного val...")

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
    val_df = attach_transition_features(val_df, val, trans_full)
    test_df = attach_transition_features(test_df, test, trans_full)
    print(f"      val={len(val_df):,}  test={len(test_df):,}")

    # ==== 6. Сборка + кодирование ====
    print("\n[6/8] Сборка + единый словарь категорий...")

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

    train_df = pd.concat([train_df_p1, train_df_p0, train_df_pmask, p2_df], ignore_index=True)

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

    print(f"      Train: {train_df['student_id'].nunique():,} групп, "
          f"{len(train_df):,} строк")

    # ==== 7. Обучение ====
    print("\n[7/8] Обучение LightGBM Ranker (FEATURE_COLS_V5)...")

    X_tr, y_tr, g_tr, w_tr, _ = prepare_xygw(train_df, feature_cols=FEATURE_COLS_V5)
    X_ho, y_ho, g_ho, w_ho, ho_sorted = prepare_xygw(holdout_df, feature_cols=FEATURE_COLS_V5)
    X_val, y_val, g_val, w_val, val_sorted = prepare_xygw(val_df, feature_cols=FEATURE_COLS_V5)
    _, _, _, _, test_sorted = prepare_xygw(test_df, feature_cols=FEATURE_COLS_V5)

    print(f"      Train: X={X_tr.shape}, групп={len(g_tr)}")

    t_train = time.time()
    model = train_lgbm_ranker(
        X_tr, y_tr, g_tr,
        X_ho, y_ho, g_ho,
        params=FAST_PARAMS,
        num_boost_round=NUM_BOOST_ROUND,
        early_stopping_rounds=EARLY_STOPPING,
        log_every=100,
        sample_weight_train=w_tr,
        sample_weight_val=w_ho,
    )
    print(f"      Обучено за {time.time() - t_train:.1f} сек, "
          f"best_iter={model.best_iteration}")

    save_model(model, MODELS_DIR / "lgbm_ranker_v5.txt")

    # ==== 8. Оценка + test ====
    print("\n[8/8] Оценка + test...")

    # Holdout — честная метрика
    scores_ho = predict_scores(model, ho_sorted, feature_cols=FEATURE_COLS_V5)
    rec_ho = top_k_per_student(ho_sorted, scores_ho, k=5)
    rec_ho_clean = {sid.replace("ho__", ""): v for sid, v in rec_ho.items()}
    gt_ho = {sid: gt for sid, gt in val_gt_raw.items() if sid in holdout_ids}
    mean_ho, _ = evaluate_pndcg5(rec_ho_clean, gt_ho, k=5)
    print(f"\n      Holdout (n={len(holdout_records)}): {mean_ho:.4f}   <- ЧЕСТНАЯ")
    print(f"      --- сравнение:")
    print(f"          v4 без transition (A): 0.4987")
    print(f"          v5 (с transition):     {mean_ho:.4f}")

    # Full val (с утечкой) — для справки
    scores_val = predict_scores(model, val_sorted, feature_cols=FEATURE_COLS_V5)
    rec_val_ltr = top_k_per_student(val_sorted, scores_val, k=5)
    mean_val_ltr, cold_ltr, warm_ltr = _segment_means(
        rec_val_ltr, val_gt_raw, identify_cold_students(val),
    )
    print(f"      Full val (LTR raw):     {mean_val_ltr:.4f}   "
          f"cold={cold_ltr:.4f}  warm={warm_ltr:.4f}")

    # Fallback для cold
    cold_ids_val = identify_cold_students(val)
    pop_weighted, _ = compute_popularity(train)
    fb_val = build_global_recommendations(val, pop_weighted, cold_ids_val, k=5)
    rec_val_fb = apply_fallback(rec_val_ltr, fb_val, k=5)

    mean_val, cold_val, warm_val = _segment_means(rec_val_fb, val_gt_raw, cold_ids_val)
    est_val = TEST_COLD_SHARE * cold_val + (1 - TEST_COLD_SHARE) * warm_val
    print(f"      Full val (+ fallback):  mean={mean_val:.4f}  "
          f"cold={cold_val:.4f}  warm={warm_val:.4f}")
    print(f"      test_estimate:          {est_val:.4f}")

    val_path = SUBMISSIONS_DIR / "val_submission_ltr_v5.csv"
    write_submission(rec_val_fb, val_path)

    # Test
    scores_test = predict_scores(model, test_sorted, feature_cols=FEATURE_COLS_V5)
    rec_test_ltr = top_k_per_student(test_sorted, scores_test, k=5)
    cold_ids_test = identify_cold_students(test)
    fb_test = build_global_recommendations(test, pop_weighted, cold_ids_test, k=5)
    rec_test_final = apply_fallback(rec_test_ltr, fb_test, k=5)

    test_path = SUBMISSIONS_DIR / "submission_v5.csv"
    write_submission(rec_test_final, test_path)
    print(f"\n      Cold-студентов в test: {len(cold_ids_test)}")
    print(f"      Сохранено: {test_path}  ({len(rec_test_final)} строк)")

    # Feature importance
    print("\n[Feature importance — топ-15]")
    imp = feature_importance_report(model, feature_cols=FEATURE_COLS_V5, top_n=15)
    for _, row in imp.iterrows():
        print(f"      {row['feature']:<32}  gain = {row['gain']:>12.0f}")

    print(f"\n[OK] Пайплайн завершён за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()