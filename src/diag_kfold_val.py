"""
KFold-метрика: стабильная локальная оценка на всём val.

ПРОБЛЕМА:
    Текущий holdout (284 val_p2) шумит на ±0.0075 между seeds. Все
    эксперименты v10-v25 показывали приросты +0.001..+0.010 — в пределах шума.
    Мы не можем отличить сигнал от шума.

РЕШЕНИЕ:
    5-fold CV по warm-val студентам (val_p1 + val_p2 = 2671).
    На каждом фолде:
        - 80% val_p1 + val_p2 → добавляем в train как дополнительные срезы
        - 20% val_p1 + val_p2 → holdout этого фолда (OOF)
    Обучаем 5 моделей (по одной на фолд), объединяем OOF-предсказания.
    Итоговая метрика — pNDCG@5 на всех 2671 warm-val студентах.

    Cold-val (88) — отдельно, модель их не тренирует, просто eval.

    Итог: composite = 0.352 * cold + 0.648 * oof_warm.

ОЖИДАНИЕ:
    std(metric) снизится в 3 раза → ±0.0024.
    Все будущие эксперименты будут различимы с точностью ±0.003.

Запуск:
    python -m src.diag_kfold_val
    python -m src.diag_kfold_val --n-seeds 3  # несколько seeds на фолд
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_COLD_SHARE = 0.352
WEIGHT_P1 = 1.0
WEIGHT_P0 = 0.5
WEIGHT_PMASK = 1.0
WEIGHT_P2 = 3.0
WEIGHT_VALP1 = 1.0  # только для val-фолдов

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

N_FOLDS = 5


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


def main() -> None:
    parser = argparse.ArgumentParser(description="KFold-метрика по val")
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    args = parser.parse_args()
    n_seeds = max(1, int(args.n_seeds))
    n_folds = max(2, int(args.n_folds))
    seeds = [BASE_SEED + i for i in range(n_seeds)]

    print("=" * 78)
    print(f"KFold-метрика: {n_folds}-fold по warm-val, n_seeds={n_seeds}")
    print("=" * 78)

    t0 = time.time()
    data = load_all()
    train = data["train"]
    val = data["val"]
    modules = data["modules"]
    val_solution = data["val_solution"]
    val_gt_raw = dict(zip(val_solution["id"], val_solution["ground_truth"]))

    # ==== 1. Статистики на train ====
    print("\n[1/5] Статистики (только train, без val — чтобы избежать утечки)…")
    stats = FeatureStats.fit(train, modules)
    stats.add_val_solution_popularity(val_solution)

    # ==== 2. Все train-срезы ====
    print("\n[2/5] Train-срезы…")
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
    # fix периодов (для train — no-op)
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

    # ==== 3. Полная val-выборка (источники для фолдов) ====
    print("\n[3/5] Разбиваем val на фолды…")
    val_p1_records = [s for s in val if _n_periods(s) == 1]
    val_p2_records = [s for s in val if _n_periods(s) == 2]
    print(f"      val_p1 = {len(val_p1_records)}, val_p2 = {len(val_p2_records)}")

    all_warm_val = [(s, 'p1') for s in val_p1_records] + [(s, 'p2') for s in val_p2_records]
    rng = random.Random(BASE_SEED)
    rng.shuffle(all_warm_val)

    kf = KFold(n_splits=n_folds, shuffle=False)  # уже перемешано
    fold_splits = list(kf.split(all_warm_val))

    # ==== 4. Обучаем модель на каждом фолде ====
    print(f"\n[4/5] Обучение {n_folds} моделей (по фолду) × {n_seeds} seeds…")
    oof_rec: dict[str, list[str]] = {}  # student_id → top-5 (OOF)
    fold_info: list[dict[str, Any]] = []

    for fold_idx, (train_idx, hold_idx) in enumerate(fold_splits):
        fold_hold = [all_warm_val[i] for i in hold_idx]
        fold_train = [all_warm_val[i] for i in train_idx]
        hold_ids = {s["id"] for s, _ in fold_hold}

        # Источники для обучающих срезов этого фолда
        fold_valp1_train = [s for s, t in fold_train if t == 'p1']
        fold_valp2_train = [s for s, t in fold_train if t == 'p2']
        fold_valp1_hold = [s for s, t in fold_hold if t == 'p1']
        fold_valp2_hold = [s for s, t in fold_hold if t == 'p2']

        # Обучающие срезы: valp1_train, valp2_train
        train_df_valp1 = build_dataset(
            fold_valp1_train, stats,
            candidates_fn=candidates_available,
            target_fn=make_target_from_val_solution(val_gt_raw),
            past_periods=[1],
        ) if fold_valp1_train else None
        train_df_valp2 = build_dataset(
            fold_valp2_train, stats,
            candidates_fn=candidates_available,
            target_fn=make_target_from_val_solution(val_gt_raw),
            past_periods=[1, 2],
        ) if fold_valp2_train else None

        if train_df_valp1 is not None:
            train_df_valp1 = add_correct_period_features(
                train_df_valp1, fold_valp1_train, stats.pop_by_period, past_periods=[1])
            train_df_valp1 = _add_weight(train_df_valp1, WEIGHT_VALP1)
            train_df_valp1["student_id"] = "vp1__" + train_df_valp1["student_id"].astype(str)
        if train_df_valp2 is not None:
            train_df_valp2 = add_correct_period_features(
                train_df_valp2, fold_valp2_train, stats.pop_by_period, past_periods=[1, 2])
            train_df_valp2 = _add_weight(train_df_valp2, WEIGHT_P2)
            train_df_valp2["student_id"] = "vp2__" + train_df_valp2["student_id"].astype(str)

        # Holdout датасеты фолда (для OOF-оценки)
        hold_df_p1 = build_dataset(
            fold_valp1_hold, stats,
            candidates_fn=candidates_available,
            target_fn=make_target_from_val_solution(val_gt_raw),
            past_periods=[1],
        ) if fold_valp1_hold else None
        hold_df_p2 = build_dataset(
            fold_valp2_hold, stats,
            candidates_fn=candidates_available,
            target_fn=make_target_from_val_solution(val_gt_raw),
            past_periods=[1, 2],
        ) if fold_valp2_hold else None

        if hold_df_p1 is not None:
            hold_df_p1 = add_correct_period_features(
                hold_df_p1, fold_valp1_hold, stats.pop_by_period, past_periods=[1])
            hold_df_p1 = _add_weight(hold_df_p1, 1.0)
            hold_df_p1["student_id"] = "hp1__" + hold_df_p1["student_id"].astype(str)
        if hold_df_p2 is not None:
            hold_df_p2 = add_correct_period_features(
                hold_df_p2, fold_valp2_hold, stats.pop_by_period, past_periods=[1, 2])
            hold_df_p2 = _add_weight(hold_df_p2, 1.0)
            hold_df_p2["student_id"] = "hp2__" + hold_df_p2["student_id"].astype(str)

        # Собираем train_df
        train_parts = [train_df_p1, train_df_p0, train_df_pmask]
        if train_df_valp1 is not None:
            train_parts.append(train_df_valp1)
        if train_df_valp2 is not None:
            train_parts.append(train_df_valp2)
        train_df_fold = pd.concat(train_parts, ignore_index=True)

        # Собираем holdout_df
        hold_parts = []
        if hold_df_p1 is not None:
            hold_parts.append(hold_df_p1)
        if hold_df_p2 is not None:
            hold_parts.append(hold_df_p2)
        hold_df_fold = pd.concat(hold_parts, ignore_index=True) if hold_parts else None

        if hold_df_fold is None or len(hold_df_fold) == 0:
            print(f"      fold {fold_idx+1}: пустой holdout, пропускаем")
            continue

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

        # Несколько seeds на фолд
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

        # Усредняем по seeds
        hold_preds = np.mean(hold_preds_list, axis=0)
        rec_fold = top_k_per_student(hold_sorted, hold_preds, k=5)
        # Снимаем префикс
        for sid_pref, rec in rec_fold.items():
            sid_clean = sid_pref.replace("hp1__", "").replace("hp2__", "")
            oof_rec[sid_clean] = rec

        # Диагностика фолда
        fold_ho_p1 = _mean_for_ids(rec_fold, val_gt_raw,
                                    {s["id"] for s in fold_valp1_hold})
        fold_ho_p2 = _mean_for_ids(rec_fold, val_gt_raw,
                                    {s["id"] for s in fold_valp2_hold})
        fold_info.append({
            "fold": fold_idx + 1,
            "n_p1": len(fold_valp1_hold),
            "n_p2": len(fold_valp2_hold),
            "ho_p1": fold_ho_p1,
            "ho_p2": fold_ho_p2,
        })
        print(f"      fold {fold_idx+1}: p1={len(fold_valp1_hold)}, "
              f"p2={len(fold_valp2_hold)}, "
              f"ho_p1={fold_ho_p1:.4f}, ho_p2={fold_ho_p2:.4f}")

    # ==== 5. Итог ====
    print(f"\n[5/5] Итог:")

    p1_ids = {s["id"] for s in val_p1_records}
    p2_ids = {s["id"] for s in val_p2_records}
    cold_ids = identify_cold_students(val)

    # OOF-метрика на всём warm-val
    mean_warm_oof = _mean_for_ids(oof_rec, val_gt_raw, p1_ids | p2_ids)
    mean_p1 = _mean_for_ids(oof_rec, val_gt_raw, p1_ids)
    mean_p2 = _mean_for_ids(oof_rec, val_gt_raw, p2_ids)

    print(f"\n      OOF на warm-val:")
    print(f"          p1 (n={len(p1_ids)}):     {mean_p1:.4f}")
    print(f"          p2 (n={len(p2_ids)}):     {mean_p2:.4f}")
    print(f"          всего (n={len(p1_ids|p2_ids)}): {mean_warm_oof:.4f}")

    # Cold-val — просто оценим популярность
    from src.popularity import compute_popularity
    pop_w, _ = compute_popularity(train)
    rec_cold: dict[str, list[str]] = {}
    for s in val:
        if s["id"] not in cold_ids:
            continue
        avail = list(s.get("available_modules") or [])
        scored = [(m, pop_w.get(m, 0.0)) for m in avail]
        scored.sort(key=lambda x: (-x[1], x[0]))
        rec_cold[s["id"]] = [m for m, _ in scored[:5]]
    mean_cold = _mean_for_ids(rec_cold, val_gt_raw, cold_ids)
    print(f"\n      Cold-val (n={len(cold_ids)}): {mean_cold:.4f}")

    composite = TEST_COLD_SHARE * mean_cold + (1 - TEST_COLD_SHARE) * mean_warm_oof
    print(f"\n      Composite test-estimate:")
    print(f"          0.352 × {mean_cold:.4f} + 0.648 × {mean_warm_oof:.4f} = "
          f"{composite:.4f}")

    # Сравнение с holdout на 284
    print(f"\n      Сравнение с шумным holdout (284):")
    print(f"          старый holdout std: ±0.0075 (10 seeds)")
    print(f"          новый KFold std:    ±{0.0075 / np.sqrt(len(p1_ids|p2_ids)/284):.4f} "
          f"(оценочно, n={len(p1_ids|p2_ids)})")

    print(f"\n[OK] Диагностика завершена за {time.time() - t0:.1f} сек.")

    print("\n" + "=" * 78)
    print("ИНТЕРПРЕТАЦИЯ:")
    print("=" * 78)
    print(f"  Composite = {composite:.4f}")
    print(f"  Это БАЗОВАЯ метрика для будущих сравнений.")
    print(f"  Ожидаемый уровень шума: ±{0.0075 / np.sqrt(len(p1_ids|p2_ids)/284):.4f}")
    print(f"  Все будущие эксперименты сравнивать по этой метрике.")


if __name__ == "__main__":
    main()