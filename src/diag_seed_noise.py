"""
Диагностика: 10-seed holdout для v7 и v23.

ВОПРОС:
    Уровень шума holdout-метрики при multi-seed. Все наши эксперименты
    (v10-v25) показывали приросты +0.001..+0.010 на holdout. Это
    реальный сигнал или шум?

МЕТОД:
    Обучаем v7-конфигурацию (без fix периодов, без val_p1) на 10 seeds.
    Обучаем v23-конфигурацию (с fix периодов, с val_p1) на 10 seeds.
    Считаем holdout для каждой отдельной модели и статистику.
    Считаем holdout усреднённых ансамблей при N=1..10 seeds.

ВЫВОД:
    Сравниваем std между seeds vs Δ между mean(v7) и mean(v23).
    Если Δ < 2*std / sqrt(10) — разница статистически незначима.

Запуск:
    python -m src.diag_seed_noise
"""
from __future__ import annotations

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
    predict_scores,
    prepare_xygw,
    top_k_per_student,
    train_lgbm_ranker,
)
from src.metric import evaluate_pndcg5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

TEST_COLD_SHARE = 0.352
WEIGHT_P1 = 1.0
WEIGHT_P0 = 0.5
WEIGHT_PMASK = 1.0
WEIGHT_P2 = 3.0
WEIGHT_VALP1 = 1.0

HOLDOUT_FRACTION = 0.20
HOLDOUT_SEED = 42

# 10 разных seeds
SEEDS = list(range(42, 52))

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


def _score_ho(ho_sorted: pd.DataFrame, scores: np.ndarray,
              gt_ho: dict[str, str]) -> float:
    rec = top_k_per_student(ho_sorted, scores, k=5)
    rec_clean = {sid.replace("ho__", ""): v for sid, v in rec.items()}
    m, _ = evaluate_pndcg5(rec_clean, gt_ho, k=5)
    return m


def _prepare_one(
    name: str,
    stats: FeatureStats,
    train_students_p1: list[dict],
    train_students_p0: list[dict],
    p2_records: list[dict],
    val_p1_records: list[dict],
    holdout_records: list[dict],
    apply_fix: bool,
    include_valp1: bool,
    val_gt_raw: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Возвращает (X_train, X_ho, y_train, g_train, w_train).
    Один раз строит df — потом обучаем на 10 seeds.
    """
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
    p2_df = build_dataset(
        p2_records, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )
    valp1_df = None
    if include_valp1:
        valp1_df = build_dataset(
            val_p1_records, stats,
            candidates_fn=candidates_available,
            target_fn=make_target_from_val_solution(val_gt_raw),
            past_periods=[1],
        )

    holdout_df = build_dataset(
        holdout_records, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )

    if apply_fix:
        train_df_p1 = add_correct_period_features(
            train_df_p1, train_students_p1, stats.pop_by_period, past_periods=[1])
        train_df_p0 = add_correct_period_features(
            train_df_p0, train_students_p0, stats.pop_by_period, past_periods=[])
        train_df_pmask = add_correct_period_features(
            train_df_pmask, train_students_p1, stats.pop_by_period, past_periods=[1])
        p2_df = add_correct_period_features(
            p2_df, p2_records, stats.pop_by_period, past_periods=[1, 2])
        holdout_df = add_correct_period_features(
            holdout_df, holdout_records, stats.pop_by_period, past_periods=[1, 2])
        if valp1_df is not None:
            valp1_df = add_correct_period_features(
                valp1_df, val_p1_records, stats.pop_by_period, past_periods=[1])

    train_df_p1 = _add_weight(train_df_p1, WEIGHT_P1)
    train_df_p0 = _add_weight(train_df_p0, WEIGHT_P0)
    train_df_pmask = _add_weight(train_df_pmask, WEIGHT_PMASK)
    p2_df = _add_weight(p2_df, WEIGHT_P2)
    holdout_df = _add_weight(holdout_df, 1.0)
    if valp1_df is not None:
        valp1_df = _add_weight(valp1_df, WEIGHT_VALP1)

    train_df_p1["student_id"] = "p1__" + train_df_p1["student_id"].astype(str)
    train_df_p0["student_id"] = "p0__" + train_df_p0["student_id"].astype(str)
    train_df_pmask["student_id"] = "pmask__" + train_df_pmask["student_id"].astype(str)
    p2_df["student_id"] = "p2__" + p2_df["student_id"].astype(str)
    holdout_df["student_id"] = "ho__" + holdout_df["student_id"].astype(str)
    if valp1_df is not None:
        valp1_df["student_id"] = "vp1__" + valp1_df["student_id"].astype(str)

    parts = [train_df_p1, train_df_p0, train_df_pmask, p2_df]
    if valp1_df is not None:
        parts.append(valp1_df)
    train_df = pd.concat(parts, ignore_index=True)

    n_train = len(train_df)
    n_ho = len(holdout_df)

    all_df = pd.concat([train_df, holdout_df], ignore_index=True)
    all_df, _ = encode_categoricals(all_df, all_df, CATEGORICAL_FEATS)
    train_df = all_df.iloc[:n_train].reset_index(drop=True)
    holdout_df = all_df.iloc[n_train:].reset_index(drop=True)

    X_train, y_train, g_train, w_train, _ = prepare_xygw(
        train_df, feature_cols=FEATURE_COLS_V4)
    _, _, _, _, ho_sorted = prepare_xygw(holdout_df, feature_cols=FEATURE_COLS_V4)

    print(f"      [{name}] X_train={X_train.shape}, групп={len(g_train)}")

    return X_train, y_train, g_train, w_train, ho_sorted


def _train_seeds(
    name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    g_train: np.ndarray,
    w_train: np.ndarray,
    ho_sorted: pd.DataFrame,
    gt_ho: dict[str, str],
    seeds: list[int],
) -> tuple[list[float], list[np.ndarray]]:
    """
    Обучает модель на каждом seed. Возвращает (list_holdout_scores, list_ho_preds).
    """
    X_ho, y_ho, g_ho, w_ho, _ = prepare_xygw(
        ho_sorted, feature_cols=FEATURE_COLS_V4)
    scores_per_seed: list[float] = []
    preds_per_seed: list[np.ndarray] = []
    for seed in seeds:
        params = {**FAST_PARAMS, "seed": seed}
        t0 = time.time()
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
        preds = predict_scores(model, ho_sorted, feature_cols=FEATURE_COLS_V4)
        preds_per_seed.append(preds)
        s = _score_ho(ho_sorted, preds, gt_ho)
        scores_per_seed.append(s)
        print(f"      [{name}] seed={seed}: best_iter={model.best_iteration}, "
              f"holdout={s:.4f}, {time.time() - t0:.1f}сек")
    return scores_per_seed, preds_per_seed


def _ensemble_curve(preds_per_seed: list[np.ndarray],
                    ho_sorted: pd.DataFrame,
                    gt_ho: dict[str, str]) -> list[float]:
    """Holdout усреднённого ансамбля при N=1..len(preds)."""
    cum: list[float] = []
    for n in range(1, len(preds_per_seed) + 1):
        avg = np.mean(preds_per_seed[:n], axis=0)
        s = _score_ho(ho_sorted, avg, gt_ho)
        cum.append(s)
    return cum


def main() -> None:
    print("=" * 78)
    print(f"ДИАГНОСТИКА: 10-seed holdout для v7 и v23")
    print(f"Seeds: {SEEDS}")
    print("=" * 78)

    t0 = time.time()
    data = load_all()
    train = data["train"]
    val = data["val"]
    modules = data["modules"]
    val_solution = data["val_solution"]
    val_gt_raw = dict(zip(val_solution["id"], val_solution["ground_truth"]))

    # ==== Статистики (один раз) ====
    print("\n[1/4] Статистики...")
    stats = FeatureStats.fit(train, modules)
    stats.add_val_solution_popularity(val_solution)

    # ==== Подготовка списков ====
    train_students_p1 = [
        s for s in train
        if any(h.get("period") == 2 for h in s.get("history", []))
    ]
    train_students_p0 = [
        s for s in train if any(h.get("period") == 1 for h in s.get("history", []))
    ]
    val_p2 = [s for s in val if _n_periods(s) == 2]
    rng = random.Random(HOLDOUT_SEED)
    shuffled = list(val_p2)
    rng.shuffle(shuffled)
    n_holdout = int(len(shuffled) * HOLDOUT_FRACTION)
    holdout_records = shuffled[:n_holdout]
    p2_records = shuffled[n_holdout:]
    holdout_ids = {s["id"] for s in holdout_records}
    val_p1_records = [s for s in val if _n_periods(s) == 1]
    gt_ho = {sid: gt for sid, gt in val_gt_raw.items() if sid in holdout_ids}

    print(f"      p1={len(train_students_p1)}  p0={len(train_students_p0)}  "
          f"p2={len(p2_records)}  holdout={len(holdout_records)}  "
          f"val_p1={len(val_p1_records)}")

    # ==== v7 ====
    print("\n[2/4] v7 (без fix, без val_p1)...")
    X_tr, y_tr, g_tr, w_tr, ho_sorted = _prepare_one(
        name="v7",
        stats=stats,
        train_students_p1=train_students_p1,
        train_students_p0=train_students_p0,
        p2_records=p2_records,
        val_p1_records=val_p1_records,
        holdout_records=holdout_records,
        apply_fix=False,
        include_valp1=False,
        val_gt_raw=val_gt_raw,
    )
    v7_scores, v7_preds = _train_seeds(
        name="v7",
        X_train=X_tr, y_train=y_tr, g_train=g_tr, w_train=w_tr,
        ho_sorted=ho_sorted, gt_ho=gt_ho, seeds=SEEDS,
    )

    # ==== v23 ====
    print("\n[3/4] v23 (fix + val_p1)...")
    X_tr, y_tr, g_tr, w_tr, ho_sorted_v23 = _prepare_one(
        name="v23",
        stats=stats,
        train_students_p1=train_students_p1,
        train_students_p0=train_students_p0,
        p2_records=p2_records,
        val_p1_records=val_p1_records,
        holdout_records=holdout_records,
        apply_fix=True,
        include_valp1=True,
        val_gt_raw=val_gt_raw,
    )
    v23_scores, v23_preds = _train_seeds(
        name="v23",
        X_train=X_tr, y_train=y_tr, g_train=g_tr, w_train=w_tr,
        ho_sorted=ho_sorted_v23, gt_ho=gt_ho, seeds=SEEDS,
    )

    # ==== Статистика ====
    print("\n[4/4] Анализ:")
    print("\n" + "=" * 78)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 78)

    v7_arr = np.array(v7_scores)
    v23_arr = np.array(v23_scores)

    print(f"\n[Per-seed holdout, 10 seeds]")
    print(f"  v7:  mean={v7_arr.mean():.4f}  std={v7_arr.std(ddof=1):.4f}  "
          f"min={v7_arr.min():.4f}  max={v7_arr.max():.4f}")
    print(f"  v23: mean={v23_arr.mean():.4f}  std={v23_arr.std(ddof=1):.4f}  "
          f"min={v23_arr.min():.4f}  max={v23_arr.max():.4f}")

    diff = v23_arr.mean() - v7_arr.mean()
    # SE разности независимых средних
    se = np.sqrt(v7_arr.var(ddof=1)/len(v7_arr) + v23_arr.var(ddof=1)/len(v23_arr))
    t_stat = diff / se if se > 0 else 0.0
    print(f"\n[Разница mean(v23) - mean(v7)]")
    print(f"  Δ = {diff:+.4f}   SE(Δ) = {se:.4f}   t = {t_stat:.2f}")
    print(f"  |t| > 2.0 => статистически значимо (p<0.05)")
    print(f"  |t| <= 2.0 => в пределах шума")

    # ==== Кумулятивные ансамбли ====
    print(f"\n[Кумулятивный ансамбль N=1..10]")
    v7_curve = _ensemble_curve(v7_preds, ho_sorted, gt_ho)
    v23_curve = _ensemble_curve(v23_preds, ho_sorted_v23, gt_ho)

    print(f"  N   v7       v23")
    for i in range(len(v7_curve)):
        print(f"  {i+1:<3} {v7_curve[i]:.4f}   {v23_curve[i]:.4f}")

    # ==== Итог ====
    print("\n" + "=" * 78)
    print("ИНТЕРПРЕТАЦИЯ:")
    print("=" * 78)
    if v7_arr.std(ddof=1) > 0.008:
        print("[!] std(v7) > 0.008 => holdout ШУМИТ.")
        print("    Все наши приросты +0.005..+0.010 были в пределах 1σ.")
        print("    => Для сравнения конфигураций нужно 15+ seeds,")
        print("       либо тестировать на public.")
    else:
        print(f"[+] std(v7) = {v7_arr.std(ddof=1):.4f} <= 0.008 => holdout ОК.")
        print("    Наблюдаемые приросты можно интерпретировать как сигнал.")

    if abs(t_stat) > 2.0:
        print(f"\n[+] v23 vs v7 статистически значимо (t={t_stat:.2f}).")
        direction = "лучше" if diff > 0 else "хуже"
        print(f"    v23 {direction} на {abs(diff):.4f}.")
        print(f"    => Сабмитим v23 с уверенностью.")
    else:
        print(f"\n[~] v23 vs v7 НЕ значимо (t={t_stat:.2f}, |t|<2.0).")
        print(f"    Разница {diff:+.4f} в пределах шума 10 seeds.")
        print(f"    => Обе версии эквивалентны, выбор между ними — вкусовой.")

    print(f"\n[OK] Диагностика завершена за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()