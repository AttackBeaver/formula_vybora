"""
LTR v24: чистая проверка fix периодов (без val_p1).

Изолированный эксперимент:
    v7 КАК ЕСТЬ + перезапись pop_last/prev/first_period значениями
    по истинным (до renaming) периодам студента.

    Для train и val/holdout (P1/P2) это no-op.
    Для test P3 — исправление: last=3, prev=2, first=1
    (вместо last=2, prev=1, first=1, которые даёт build_dataset
     после renaming).

    Отличие от v23: без среза val_p1 (w=0). Это позволяет
    изолировать эффект fix от эффекта +val_p1.

Ожидания:
    Holdout ≈ 0.5134 (v7-уровень; если больше — либо fix, либо шум).
    Cold ≈ 0.2474 (без val_p1 cold не растёт).
    test_estimate ≈ 0.4017.
    Test top-5 отличается от v7 для P3-студентов.

Запуск:
    python -m src.run_ltr_v24 --n-seeds 5
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
V22_HOLDOUT = 0.5067
V23_HOLDOUT = 0.5225
V23_COLD_VAL = 0.2959


def _n_periods(r: dict[str, Any]) -> int:
    return len(r.get("history") or [])


def _add_weight(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = np.float32(weight)
    return df


def _strip_prefix(sid_series: pd.Series) -> pd.Series:
    return sid_series.str.replace(r"^[a-z0-9]+__", "", regex=True)


def _mean_for_ids(
    rec: dict[str, list[str]],
    gt_raw: dict[str, str],
    ids: set[str],
    k: int = 5,
) -> float:
    _, per_student = evaluate_pndcg5(rec, gt_raw, k=k)
    vals = [s for sid, s in per_student.items() if sid in ids]
    return float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Коррекция периодов (копия из v23)
# ---------------------------------------------------------------------------
def add_correct_period_features(
    df: pd.DataFrame,
    students: list[dict],
    pop_by_period: dict[int, dict[str, float]],
    past_periods: list[int] | None,
) -> pd.DataFrame:
    """
    Перезаписывает pop_last_period, pop_prev_period, pop_first_period,
    используя ИСТИННЫЕ (до renaming) периоды студента, отфильтрованные
    по past_periods.

    Для train/val P1/P2 — no-op (то же, что build_dataset).
    Для test P3 — исправление (last=3, prev=2, first=1).
    """
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


def main() -> None:
    parser = argparse.ArgumentParser(description="LTR v24: чистый fix периодов")
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()
    n_seeds = max(1, int(args.n_seeds))
    seeds = [BASE_SEED + i for i in range(n_seeds)]

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print(f"LTR v24: чистый fix периодов (v7 + fix, без val_p1), n_seeds={n_seeds}")
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
    print(f"      pop_by_period[3]: {len(stats.pop_by_period.get(3, {}))} модулей")

    # ==== 2. Train-срезы ====
    print("\n[2/9] Train-срезы (как v7)...")
    train_students_p1 = [
        s for s in train
        if any(h.get("period") == 2 for h in s.get("history", []))
    ]
    train_students_p0 = [
        s for s in train if any(h.get("period") == 1 for h in s.get("history", []))
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
    print(f"      p1={len(train_df_p1):,}  p0={len(train_df_p0):,}  "
          f"pmask={len(train_df_pmask):,}")

    # ==== 3. Holdout + p2 ====
    print("\n[3/9] Holdout + p2...")
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

    # ==== 4. Val и test ====
    print("\n[4/9] Val и test...")
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

    # ==== 5. Коррекция периодов ====
    print("\n[5/9] Коррекция pop_last/prev/first_period...")
    train_df_p1 = add_correct_period_features(
        train_df_p1, train_students_p1, stats.pop_by_period, past_periods=[1],
    )
    train_df_p0 = add_correct_period_features(
        train_df_p0, train_students_p0, stats.pop_by_period, past_periods=[],
    )
    train_df_pmask = add_correct_period_features(
        train_df_pmask, train_students_p1, stats.pop_by_period, past_periods=[1],
    )
    p2_df = add_correct_period_features(
        p2_df, p2_records, stats.pop_by_period, past_periods=[1, 2],
    )
    holdout_df = add_correct_period_features(
        holdout_df, holdout_records, stats.pop_by_period, past_periods=[1, 2],
    )
    val_df = add_correct_period_features(
        val_df, val, stats.pop_by_period, past_periods=[1, 2],
    )
    test_df = add_correct_period_features(
        test_df, test, stats.pop_by_period, past_periods=None,
    )

    # Диагностика: проверяем, что train не изменился (no-op)
    print(f"      Проверка no-op для train: см. ниже (сравнение с v7 в конце)")

    test_p3_ids = {s["id"] for s in test if _n_periods(s) >= 3}
    test_p3_rows = test_df[test_df["student_id"].isin(test_p3_ids)]
    if len(test_p3_rows) > 0:
        n_nonzero = int((test_p3_rows["pop_last_period"].values > 0).sum())
        print(f"      test P3: {test_p3_rows['student_id'].nunique()} студентов, "
              f"pop_last_period!=0 у {n_nonzero:,}/{len(test_p3_rows):,}")

    # ==== 6. Веса, префиксы, категориальные ====
    print("\n[6/9] Веса, префиксы, категориальные...")
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
    val_df["student_id"] = "vl__" + val_df["student_id"].astype(str)
    test_df["student_id"] = "ts__" + test_df["student_id"].astype(str)

    train_df = pd.concat(
        [train_df_p1, train_df_p0, train_df_pmask, p2_df],
        ignore_index=True,
    )
    print(f"      train: 4 среза, {len(train_df):,} строк")

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

    for seed in seeds:
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

    scores_ho = np.mean(scores_ho_list, axis=0)
    scores_val = np.mean(scores_val_list, axis=0)
    scores_test = np.mean(scores_test_list, axis=0)
    print(f"\n      best_iter: mean={np.mean(best_iters):.0f}, "
          f"min={min(best_iters)}, max={max(best_iters)}")

    # ==== 8. Оценка ====
    print("\n[8/9] Оценка:")

    rec_ho = top_k_per_student(ho_sorted, scores_ho, k=5)
    rec_ho_clean = {sid.replace("ho__", ""): v for sid, v in rec_ho.items()}
    gt_ho = {sid: gt for sid, gt in val_gt_raw.items() if sid in holdout_ids}
    mean_ho, _ = evaluate_pndcg5(rec_ho_clean, gt_ho, k=5)

    print(f"      Holdout (n=284):  {mean_ho:.4f}")
    print(f"          v7   = {V7_HOLDOUT:.4f}")
    print(f"          v22  = {V22_HOLDOUT:.4f}")
    print(f"          v23  = {V23_HOLDOUT:.4f}  (с val_p1)")
    print(f"          v24  = {mean_ho:.4f}  (без val_p1)")

    cold_ids_val = identify_cold_students(val)
    rec_val = top_k_per_student(val_sorted, scores_val, k=5)
    rec_val_clean = {sid.replace("vl__", ""): v for sid, v in rec_val.items()}
    m_val_cold = _mean_for_ids(rec_val_clean, val_gt_raw, cold_ids_val)
    m_val_hold = _mean_for_ids(rec_val_clean, val_gt_raw, holdout_ids)

    print(f"\n      Cold val (n={len(cold_ids_val)}):  {m_val_cold:.4f}")
    print(f"          v7   = {V7_COLD_VAL:.4f}")
    print(f"          v23  = {V23_COLD_VAL:.4f}")
    print(f"          v24  = {m_val_cold:.4f}")

    est_val_clean = TEST_COLD_SHARE * m_val_cold + (1 - TEST_COLD_SHARE) * m_val_hold
    print(f"\n      test_estimate = {est_val_clean:.4f}")
    print(f"          v7 ≈ 0.4017  v22 = 0.4218  v23 = 0.4428  v24 = {est_val_clean:.4f}")

    val_path = SUBMISSIONS_DIR / "val_submission_ltr_v24.csv"
    write_submission(rec_val_clean, val_path)
    print(f"      Сохранено: {val_path}")

    # ==== 9. Test ====
    print("\n[9/9] Test submission...")
    rec_test = top_k_per_student(test_sorted, scores_test, k=5)
    rec_test_clean = {sid.replace("ts__", ""): v for sid, v in rec_test.items()}
    test_path = SUBMISSIONS_DIR / "submission_v24.csv"
    write_submission(rec_test_clean, test_path)
    cold_test = identify_cold_students(test)
    print(f"      Cold в test: {len(cold_test)}  Сохранено: {test_path}")

    # Сравнение с v7 / v22 / v23
    for cmp_name in ["submission_v7.csv", "submission_v22.csv", "submission_v23.csv"]:
        cmp_path = SUBMISSIONS_DIR / cmp_name
        if not cmp_path.exists():
            continue
        df_cmp = pd.read_csv(cmp_path)
        rec_cmp = {str(r["id"]): str(r["recommendations"]).split()
                   for _, r in df_cmp.iterrows()}
        n_diff = sum(1 for sid in rec_test_clean
                     if rec_test_clean[sid] != rec_cmp.get(sid, []))
        n_diff_p3 = sum(1 for sid in test_p3_ids
                        if rec_test_clean.get(sid, []) != rec_cmp.get(sid, []))
        print(f"\n      vs {cmp_name}: "
              f"top-5 отличается у {n_diff}/{len(rec_test_clean)} test-студентов; "
              f"из P3 — у {n_diff_p3}/{len(test_p3_ids)}")

    print("\n[Feature importance — top-15]")
    imp = feature_importance_report(
        # последняя модель = последний seed
        None if 'model' not in locals() else model,
        feature_cols=FEATURE_COLS_V4, top_n=15,
    )
    # model перезаписывается в цикле — берём последний
    if 'model' in locals():
        imp = feature_importance_report(model, feature_cols=FEATURE_COLS_V4, top_n=15)
        for _, row in imp.iterrows():
            print(f"      {row['feature']:<32}  gain = {row['gain']:>12.0f}")

    print(f"\n[OK] Пайплайн завершён за {time.time() - t0:.1f} сек.")

    print("\n" + "=" * 78)
    print("ИНТЕРПРЕТАЦИЯ:")
    print("=" * 78)
    print(f"  Holdout:  v7={V7_HOLDOUT:.4f}  v24={mean_ho:.4f}  "
          f"Δ={mean_ho - V7_HOLDOUT:+.4f}")
    if abs(mean_ho - V7_HOLDOUT) < 0.005:
        print(f"  [OK] Holdout ≈ v7 → fix не ломает обучение. Безопасно.")
    elif mean_ho > V7_HOLDOUT:
        print(f"  [+] Holdout ВЫШЕ v7 → fix полезен даже для warm. Хороший знак.")
    else:
        print(f"  [!] Holdout ниже v7 на > 0.005. Проверить no-op для train.")

    if m_val_cold > V7_COLD_VAL + 0.02:
        print(f"  [+] Cold вырос ({m_val_cold:.4f} vs {V7_COLD_VAL:.4f}).")
    elif abs(m_val_cold - V7_COLD_VAL) < 0.02:
        print(f"  [~] Cold ≈ v7. Fix не затрагивает cold (ожидаемо).")
    else:
        print(f"  [!] Cold упал. Странно — fix не должен влиять на cold.")

    print()
    print("  Решение о сабмите v24:")
    if mean_ho >= V7_HOLDOUT - 0.005:
        print("    → holdout в норме. v24 — безопасный кандидат.")
        print("    → ВЫБОР: v24 (чистый fix) или v23 (fix + val_p1, cold выше).")
        print("    → Если лимит 2 сабмита: отправить ОБА для сравнения.")
    else:
        print("    → holdout провалился. Откат к v7.")


if __name__ == "__main__":
    main()