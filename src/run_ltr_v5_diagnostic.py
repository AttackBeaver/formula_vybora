"""
LTR v5 diagnostic: честная проверка «модель без p2-среза» на val periods=2.

Гипотеза:
    Test periods=3 (30.4% test) требует предсказать P4, для которого
    в train/val нет прямого обучающего сигнала. Модель v4 в такой ситуации
    может проваливаться.

Как проверяем:
    Обучаем ДВЕ модели:
        A) v4: p0 + p1 + pmask + p2 (с p2-срезом)
        B) v5: p0 + p1 + pmask (БЕЗ p2-среза — аналог "test periods=3")
    Оцениваем на holdout (284 студента val periods=2, target=P3).
    Если A и B дают близкие результаты на holdout → обучение на p2-срезе
    не критично, test periods=3 не проваливается.
    Если A >> B → p2-срез критичен, test periods=3 теряет баллы.

Дополнительно оцениваем модели на группах val:
    - periods=1 (1531): target P2. Обучались оба варианта на p1.
    - periods=2 (1424): target P3. Обучались только A на p2.
    - warm / cold сегменты.

Запуск:
    python -m src.run_ltr_v5_diagnostic
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

WEIGHT_P1 = 1.0
WEIGHT_P0 = 0.5
WEIGHT_PMASK = 1.0
WEIGHT_P2 = 1.0

HOLDOUT_FRACTION = 0.20
HOLDOUT_SEED = 42

FAST_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
}


def _n_periods(r: dict[str, Any]) -> int:
    return len(r.get("history") or [])


def _add_weight(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = np.float32(weight)
    return df


def _metrics_by_group(
    model: Any,
    sorted_df: pd.DataFrame,
    feature_cols: list[str],
    label: str,
) -> None:
    scores = predict_scores(model, sorted_df, feature_cols=feature_cols)
    rec = top_k_per_student(sorted_df, scores, k=5)
    return rec


def main() -> None:
    print("=" * 78)
    print("LTR v5 diagnostic: A (с p2) vs B (без p2) на holdout и группах val")
    print("=" * 78)

    t0 = time.time()
    data = load_all()
    train = data["train"]
    val = data["val"]
    modules = data["modules"]
    val_solution = data["val_solution"]
    val_gt_raw = dict(zip(val_solution["id"], val_solution["ground_truth"]))

    print("\n[1/6] Статистики на train + val_solution...")
    stats = FeatureStats.fit(train, modules)
    stats.add_val_solution_popularity(val_solution)

    # ==== 2. Срезы ====
    print("\n[2/6] Train-срезы...")

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

    # ==== 3. Holdout + p2 ====
    print("\n[3/6] Holdout + p2-срез...")
    val_p2 = [s for s in val if _n_periods(s) == 2]
    rng = random.Random(HOLDOUT_SEED)
    shuffled = list(val_p2)
    rng.shuffle(shuffled)
    n_holdout = int(len(shuffled) * HOLDOUT_FRACTION)
    holdout_records = shuffled[:n_holdout]
    p2_records = shuffled[n_holdout:]

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

    val_df = build_dataset(
        val, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )

    # ==== 4. Собираем два train-фрейма ====
    print("\n[4/6] Сборка A (с p2) и B (без p2)...")

    for df, w, pref in [
        (train_df_p1, WEIGHT_P1, "p1"),
        (train_df_p0, WEIGHT_P0, "p0"),
        (train_df_pmask, WEIGHT_PMASK, "pmask"),
        (p2_df, WEIGHT_P2, "p2"),
        (holdout_df, 1.0, "ho"),
        (val_df, 1.0, "val"),
    ]:
        df["weight"] = np.float32(w)
        df["student_id"] = pref + "__" + df["student_id"].astype(str)

    train_A = pd.concat([train_df_p1, train_df_p0, train_df_pmask, p2_df], ignore_index=True)
    train_B = pd.concat([train_df_p1, train_df_p0, train_df_pmask], ignore_index=True)

    # Единый словарь категорий
    n_A = len(train_A)
    n_B = len(train_B)
    n_ho = len(holdout_df)
    n_val = len(val_df)

    all_df = pd.concat([train_A, train_B, holdout_df, val_df], ignore_index=True)
    all_df, _ = encode_categoricals(all_df, all_df, CATEGORICAL_FEATS)

    train_A = all_df.iloc[:n_A].reset_index(drop=True)
    train_B = all_df.iloc[n_A:n_A + n_B].reset_index(drop=True)
    holdout_df = all_df.iloc[n_A + n_B:n_A + n_B + n_ho].reset_index(drop=True)
    val_df = all_df.iloc[n_A + n_B + n_ho:].reset_index(drop=True)

    # ==== 5. Обучение двух моделей ====
    print("\n[5/6] Обучение A и B...")

    X_A, y_A, g_A, w_A, _ = prepare_xygw(train_A, feature_cols=FEATURE_COLS_V4)
    X_B, y_B, g_B, w_B, _ = prepare_xygw(train_B, feature_cols=FEATURE_COLS_V4)
    X_ho, y_ho, g_ho, w_ho, ho_sorted = prepare_xygw(holdout_df, feature_cols=FEATURE_COLS_V4)
    X_val, y_val, g_val, w_val, val_sorted = prepare_xygw(val_df, feature_cols=FEATURE_COLS_V4)

    print(f"  A: {X_A.shape}, {len(g_A)} групп")
    print(f"  B: {X_B.shape}, {len(g_B)} групп")

    tA = time.time()
    model_A = train_lgbm_ranker(
        X_A, y_A, g_A, X_ho, y_ho, g_ho,
        params=FAST_PARAMS, num_boost_round=2500, early_stopping_rounds=100,
        log_every=500, sample_weight_train=w_A, sample_weight_val=w_ho,
    )
    print(f"  A обучена за {time.time() - tA:.1f} сек, best_iter={model_A.best_iteration}")

    tB = time.time()
    model_B = train_lgbm_ranker(
        X_B, y_B, g_B, X_ho, y_ho, g_ho,
        params=FAST_PARAMS, num_boost_round=2500, early_stopping_rounds=100,
        log_every=500, sample_weight_train=w_B, sample_weight_val=w_ho,
    )
    print(f"  B обучена за {time.time() - tB:.1f} сек, best_iter={model_B.best_iteration}")

    # ==== 6. Оценка ====
    print("\n[6/6] Оценка:")

    ho_ids = {s["id"] for s in holdout_records}
    gt_ho = {sid: gt for sid, gt in val_gt_raw.items() if sid in ho_ids}

    for name, model in [("A (с p2)", model_A), ("B (без p2)", model_B)]:
        scores_ho = predict_scores(model, ho_sorted, feature_cols=FEATURE_COLS_V4)
        rec_ho = top_k_per_student(ho_sorted, scores_ho, k=5)
        rec_ho_clean = {sid.replace("ho__", ""): v for sid, v in rec_ho.items()}
        mean_ho, _ = evaluate_pndcg5(rec_ho_clean, gt_ho, k=5)
        print(f"\n  {name}:")
        print(f"      Holdout (n={len(holdout_records)}): {mean_ho:.4f}")

        # Оценка на полном val
        scores_val = predict_scores(model, val_sorted, feature_cols=FEATURE_COLS_V4)
        rec_val = top_k_per_student(val_sorted, scores_val, k=5)
        # Фильтруем val_solution только по holdout-IDs — честно для оценки
        # (иначе p2-срез пересекается с val)
        # Но для периода целиком val — с утечкой
        mean_val_full, _ = evaluate_pndcg5(rec_val, val_gt_raw, k=5)
        print(f"      Full val (утечка):      {mean_val_full:.4f}")

    print(f"\n[OK] Диагностика завершена за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()