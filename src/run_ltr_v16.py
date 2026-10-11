"""
LTR v16: абсолютные периоды для periods=3 test-студентов.

БАГ, который исправляем (найден при анализе features.py):
    build_dataset(max_history_periods=2) обрезает историю test-студентов
    с periods=3 (P1,P2,P3) до [P2,P3] и ПЕРЕИМЕНОВЫВАЕТ их в [1,2].
    После этого:
        last_p = 2 → pop_last_period = pop_by_period[2]  (неверно, нужно [3])
        prev_p = 1 → pop_prev_period = pop_by_period[1]  (неверно, нужно [2])
    
    pop_by_period[3] у нас есть — построен из val_solution (3043 студента,
    159 модулей). Он никогда не используется для test periods=3.
    
    30% test-студентов получают исторические признаки не по своему
    настоящему последнему периоду, а по train-P2. Это создаёт скрытый
    distribution shift.

РЕШЕНИЕ:
    Добавляем две АБСОЛЮТНЫЕ версии признаков, использующие оригинальный
    (до переименования) период:
        pop_last_period_abs = pop_by_period[actual_last_period]
        pop_prev_period_abs = pop_by_period[actual_prev_period]
    
    Для train (periods≤2): новые признаки == старым, модель видит дубликат.
    Для test periods=2:    новые == старым.
    Для test periods=3:    новые используют pop_by_period[3], [2] — правильные.
    Для test periods≤1:    prev_abs = 0.

Всё остальное — как в v7 (champion):
    FEATURE_COLS_V4 (31) + 2 новых = 33 признака.
    Те же срезы p0/p1/pmask/p2 с весами 0.5/1.0/1.0/3.0.
    max_history_periods=2 для test.
    truncation_level=15.
    multi-seed=5, seeds 42..46.
    Fallback отключён.

Ожидания:
    holdout      ~0.513  (не хуже v7, train не изменился содержательно)
    val cold     может вырасти (правильные pop значения для cold)
    public       если баг был значимым — выше 0.3348

Запуск:
    python -m src.run_ltr_v16 --n-seeds 5
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

ADD_TO_V4 = [
    "pop_last_period_abs",
    "pop_prev_period_abs",
]
FEATURE_COLS_V16: list[str] = FEATURE_COLS_V4 + ADD_TO_V4


def _student_id(s: dict[str, Any]) -> str | None:
    return s.get("id") or s.get("student_id")


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
# Оригинальные периоды по каждому студенту
# ---------------------------------------------------------------------------
def compute_abs_periods(
    students: list[dict],
) -> dict[str, tuple[int | None, int | None]]:
    """
    Возвращает {student_id: (last_period_abs, prev_period_abs)}.
    Периоды БЕЗ переименования — как в оригинальной истории.
    """
    result: dict[str, tuple[int | None, int | None]] = {}
    for s in students:
        sid = _student_id(s)
        if sid is None:
            continue
        periods = sorted({
            h.get("period")
            for h in (s.get("history") or [])
            if h.get("period") is not None
        })
        if not periods:
            result[sid] = (None, None)
        elif len(periods) == 1:
            result[sid] = (periods[0], None)
        else:
            result[sid] = (periods[-1], periods[-2])
    return result


def add_abs_period_features(
    df: pd.DataFrame,
    abs_periods: dict[str, tuple[int | None, int | None]],
    pop_by_period: dict[int, dict[str, float]],
) -> pd.DataFrame:
    """
    Добавляет pop_last_period_abs и pop_prev_period_abs к df.

    Ключи abs_periods — сырые student_id (без префиксов p1__/ho__/...).
    В df student_id может быть с префиксом — снимаем его.
    """
    df = df.copy()

    # Снимаем префикс, чтобы найти студента
    raw_ids = _strip_prefix(df["student_id"].astype(str)).to_numpy()

    last_abs_period = np.full(len(df), -1, dtype=np.int32)
    prev_abs_period = np.full(len(df), -1, dtype=np.int32)
    for i, sid in enumerate(raw_ids):
        last_p, prev_p = abs_periods.get(sid, (None, None))
        if last_p is not None:
            last_abs_period[i] = last_p
        if prev_p is not None:
            prev_abs_period[i] = prev_p

    module_ids = df["module_id"].to_numpy()

    # Предвычисляем max pop_weighted для нормализации (если надо). Пока raw.
    last_vals = np.zeros(len(df), dtype=np.float32)
    prev_vals = np.zeros(len(df), dtype=np.float32)
    for i in range(len(df)):
        lp = int(last_abs_period[i])
        pp = int(prev_abs_period[i])
        mid = module_ids[i]
        if lp > 0:
            last_vals[i] = float(pop_by_period.get(lp, {}).get(mid, 0.0))
        if pp > 0:
            prev_vals[i] = float(pop_by_period.get(pp, {}).get(mid, 0.0))

    df["pop_last_period_abs"] = last_vals
    df["pop_prev_period_abs"] = prev_vals
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="LTR v16: absolute-period фичи")
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()
    n_seeds = max(1, int(args.n_seeds))
    seeds = [BASE_SEED + i for i in range(n_seeds)]

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print(f"LTR v16: absolute-period фичи, n_seeds={n_seeds}")
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
    print("\n[2/9] Train-срезы...")
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
    print("\n[3/9] Holdout + p2...")
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

    # ==== 5. Новые фичи: absolute periods ====
    print("\n[5/9] Absolute-period фичи...")
    t_abs = time.time()

    # Для train используем полный train (у него periods ≤ 2 → abs == rel)
    abs_periods: dict[str, tuple[int | None, int | None]] = {}
    abs_periods.update(compute_abs_periods(train))
    abs_periods.update(compute_abs_periods(val))
    abs_periods.update(compute_abs_periods(test))
    print(f"      Студентов с известными периодами: {len(abs_periods)}")

    train_df_p1 = add_abs_period_features(train_df_p1, abs_periods, stats.pop_by_period)
    train_df_p0 = add_abs_period_features(train_df_p0, abs_periods, stats.pop_by_period)
    train_df_pmask = add_abs_period_features(train_df_pmask, abs_periods, stats.pop_by_period)
    p2_df = add_abs_period_features(p2_df, abs_periods, stats.pop_by_period)
    holdout_df = add_abs_period_features(holdout_df, abs_periods, stats.pop_by_period)
    val_df = add_abs_period_features(val_df, abs_periods, stats.pop_by_period)
    test_df = add_abs_period_features(test_df, abs_periods, stats.pop_by_period)
    print(f"      Время: {time.time() - t_abs:.1f} сек")

    # Диагностика: для test periods=3 abs vs rel должны РАЗЛИЧАТЬСЯ
    test_with_hist3 = {s["id"] for s in test if _n_periods(s) >= 3}
    test_p3_rows = test_df[test_df["student_id"].isin(test_with_hist3)]
    if len(test_p3_rows) > 0:
        n_uniq = test_p3_rows["student_id"].nunique()
        rel_mean = test_p3_rows["pop_last_period"].mean()
        abs_mean = test_p3_rows["pop_last_period_abs"].mean()
        diff = (test_p3_rows["pop_last_period"] - test_p3_rows["pop_last_period_abs"]).abs().mean()
        print(f"      test periods≥3: {n_uniq} студентов, "
              f"pop_last_period mean={rel_mean:.3f}, "
              f"pop_last_period_abs mean={abs_mean:.3f}, "
              f"|rel-abs| mean={diff:.3f}")

    # Для train (periods≤2) — должны СОВПАДАТЬ
    train_p2_rows = train_df_p1.head(2000)
    diff_train = (train_p2_rows["pop_last_period"] - train_p2_rows["pop_last_period_abs"]).abs().mean()
    print(f"      train periods=2: |pop_last_period - pop_last_period_abs| mean = {diff_train:.6f} "
          f"(должно быть ≈ 0)")

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

    print(f"      FEATURE_COLS_V16: {len(FEATURE_COLS_V16)} признаков "
          f"(v7 было {len(FEATURE_COLS_V4)})")
    print(f"      Добавлено: {ADD_TO_V4}")

    # ==== 7. Обучение ====
    print(f"\n[7/9] Обучение {n_seeds} моделей...")

    X_train, y_train, g_train, w_train, _ = prepare_xygw(
        train_df, feature_cols=FEATURE_COLS_V16,
    )
    X_ho, y_ho, g_ho, w_ho, ho_sorted = prepare_xygw(
        holdout_df, feature_cols=FEATURE_COLS_V16,
    )
    _, _, _, _, val_sorted = prepare_xygw(
        val_df, feature_cols=FEATURE_COLS_V16,
    )
    _, _, _, _, test_sorted = prepare_xygw(
        test_df, feature_cols=FEATURE_COLS_V16,
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
        scores_ho_list.append(predict_scores(model, ho_sorted, feature_cols=FEATURE_COLS_V16))
        scores_val_list.append(predict_scores(model, val_sorted, feature_cols=FEATURE_COLS_V16))
        scores_test_list.append(predict_scores(model, test_sorted, feature_cols=FEATURE_COLS_V16))
        last_model = model

    scores_ho = np.mean(scores_ho_list, axis=0)
    scores_val = np.mean(scores_val_list, axis=0)
    scores_test = np.mean(scores_test_list, axis=0)
    print(f"\n      best_iter: mean={np.mean(best_iters):.0f}, "
          f"min={min(best_iters)}, max={max(best_iters)}")

    save_model(last_model, MODELS_DIR / "lgbm_ranker_v16.txt")

    # ==== 8. Оценка ====
    print("\n[8/9] Оценка:")

    rec_ho = top_k_per_student(ho_sorted, scores_ho, k=5)
    rec_ho_clean = {sid.replace("ho__", ""): v for sid, v in rec_ho.items()}
    gt_ho = {sid: gt for sid, gt in val_gt_raw.items() if sid in holdout_ids}
    mean_ho, _ = evaluate_pndcg5(rec_ho_clean, gt_ho, k=5)

    print(f"      Holdout: {mean_ho:.4f}   (v7 = {V7_HOLDOUT:.4f}, "
          f"Δ = {mean_ho - V7_HOLDOUT:+.4f})")

    cold_ids_val = identify_cold_students(val)
    rec_val = top_k_per_student(val_sorted, scores_val, k=5)
    mean_val, cold_val, warm_val = _segment_means(rec_val, val_gt_raw, cold_ids_val)
    est_val = TEST_COLD_SHARE * cold_val + (1 - TEST_COLD_SHARE) * warm_val

    print(f"      Val: mean={mean_val:.4f}  cold={cold_val:.4f}  warm={warm_val:.4f}")
    print(f"      test_estimate = {est_val:.4f}")

    val_path = SUBMISSIONS_DIR / "val_submission_ltr_v16.csv"
    write_submission(rec_val, val_path)
    print(f"      Сохранено: {val_path}")

    rec_test = top_k_per_student(test_sorted, scores_test, k=5)
    test_path = SUBMISSIONS_DIR / "submission_v16.csv"
    write_submission(rec_test, test_path)
    print(f"      Сохранено: {test_path}")

    print("\n[Feature importance — top-15]")
    imp = feature_importance_report(last_model, feature_cols=FEATURE_COLS_V16, top_n=15)
    for _, row in imp.iterrows():
        print(f"      {row['feature']:<32}  gain = {row['gain']:>12.0f}")

    print(f"\n[OK] Пайплайн завершён за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()