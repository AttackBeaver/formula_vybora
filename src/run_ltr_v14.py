"""
LTR v14: CatBoost Ranker (YetiRank) вместо LightGBM lambdarank.

Гипотеза: за 8 итераций (v8-v13) мы только тюнили LightGBM и меняли
признаки. Все изменения сломались на public. Причина — признаки,
зависящие от состава available_modules (rank-фичи, content-pop), разные
в val и test. Пора менять сам алгоритм.

CatBoost YetiRank:
    - Стохастический pairwise-сэмплинг (по умолчанию 10k пар).
    - Симметричные (oblivious) деревья — более сильная регуляризация.
    - Устойчивая работа с категориальными признаками.
    - Другая bias-variance траектория, чем у LightGBM lambdarank.

План сравнения — честный:
    v14 = v7 по ВСЕМУ, кроме модели:
        - Те же срезы p0/p1/pmask/p2 с теми же весами (1.0 / 0.5 / 1.0 / 3.0).
        - Те же FEATURE_COLS_V4 (31 признак, без rank-фич и content-pop).
        - Тот же holdout (20% val P2, seed=42).
        - Тот же max_history_periods=2 для test.
        - Fallback отключён.
    Только модель другая.

Если holdout ≥ 0.515 → это чистый прирост от смены алгоритма.
Если < 0.5134 → CatBoost не подходит, откатываемся к v7.

Запуск:
    python -m src.run_ltr_v14 --n-seeds 5
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

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
    top_k_per_student,
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

# CatBoost hyperparameters — консервативные, как v7-подобные.
CATBOOST_PARAMS: dict[str, Any] = {
    "loss_function": "YetiRank",
    "iterations": 3000,
    "learning_rate": 0.05,
    "depth": 6,                        # ~ num_leaves=64, сопоставимо с v7's 31
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.8,
    "rsm": 0.7,                        # colsample_bylevel
    "border_count": 128,
    "verbose": 500,
    "allow_writing_files": False,
    "thread_count": -1,
}

V7_HOLDOUT = 0.5134


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


def _sort_and_group(df: pd.DataFrame) -> pd.DataFrame:
    """Сортирует df по student_id; для CatBoost группа должна быть непрерывной."""
    return df.sort_values("student_id", kind="stable").reset_index(drop=True)


def _make_pool(
    df: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    with_label: bool,
    with_weight: bool,
) -> Pool:
    df_sorted = _sort_and_group(df)
    X = df_sorted[feature_cols].copy()
    # CatBoost категориальные признаки должны быть строками
    for c in cat_cols:
        X[c] = X[c].astype(str)
    group_id = df_sorted["student_id"].astype(str).tolist()
    label = df_sorted["label"].to_numpy(dtype=np.float32) if with_label else None
    weight = df_sorted["weight"].to_numpy(dtype=np.float32) if with_weight else None
    return Pool(
        data=X,
        label=label,
        weight=weight,
        group_id=group_id,
        cat_features=cat_cols,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LTR v14: CatBoost Ranker")
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()
    n_seeds = max(1, int(args.n_seeds))
    seeds = [BASE_SEED + i for i in range(n_seeds)]

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print(f"LTR v14: CatBoost Ranker (YetiRank), n_seeds={n_seeds}")
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
    print("\n[1/8] Статистики...")
    stats = FeatureStats.fit(train, modules)
    stats.add_val_solution_popularity(val_solution)

    # ==== 2. Train-срезы ====
    print("\n[2/8] Train-срезы...")
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
    print("\n[3/8] Holdout + p2...")
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
    print("\n[4/8] Val и test...")
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

    # ==== 5. Веса, префиксы, категориальные ====
    print("\n[5/8] Веса, префиксы, категориальные...")
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
    # encode_categoricals даст общий словарь. CatBoost потом превратит в строки.
    all_df, _ = encode_categoricals(all_df, all_df, CATEGORICAL_FEATS)

    train_df = all_df.iloc[:n_train].reset_index(drop=True)
    holdout_df = all_df.iloc[n_train:n_train + n_ho].reset_index(drop=True)
    val_df = all_df.iloc[n_train + n_ho:n_train + n_ho + n_val].reset_index(drop=True)
    test_df = all_df.iloc[n_train + n_ho + n_val:].reset_index(drop=True)

    # ==== 6. Pool'ы ====
    print("\n[6/8] Pool'ы для CatBoost...")
    train_pool = _make_pool(train_df, FEATURE_COLS_V4, CATEGORICAL_FEATS,
                            with_label=True, with_weight=True)
    holdout_pool = _make_pool(holdout_df, FEATURE_COLS_V4, CATEGORICAL_FEATS,
                              with_label=True, with_weight=True)
    val_pool = _make_pool(val_df, FEATURE_COLS_V4, CATEGORICAL_FEATS,
                          with_label=True, with_weight=True)
    test_pool = _make_pool(test_df, FEATURE_COLS_V4, CATEGORICAL_FEATS,
                           with_label=False, with_weight=False)
    print(f"      train_pool: {train_pool.num_row()} строк")

    # ==== 7. Обучение ====
    print(f"\n[7/8] Обучение {n_seeds} CatBoost-моделей...")

    scores_ho_list: list[np.ndarray] = []
    scores_val_list: list[np.ndarray] = []
    scores_test_list: list[np.ndarray] = []
    best_iters: list[int] = []
    last_model: CatBoostRanker | None = None

    for i, seed in enumerate(seeds):
        print(f"\n      --- Модель {i + 1}/{n_seeds}, seed={seed} ---")
        params = {**CATBOOST_PARAMS, "random_seed": seed}
        t_train = time.time()
        model = CatBoostRanker(**params)
        model.fit(
            train_pool,
            eval_set=holdout_pool,
            use_best_model=True,
            early_stopping_rounds=100,
        )
        best_iters.append(model.get_best_iteration())
        print(f"      seed={seed}: best_iter={model.get_best_iteration()}, "
              f"{time.time() - t_train:.1f} сек")

        # predict возвращает сырые скоры; порядок — как в pool (sorted by student_id).
        ho_preds = model.predict(holdout_pool)
        val_preds = model.predict(val_pool)
        test_preds = model.predict(test_pool)
        scores_ho_list.append(np.asarray(ho_preds).ravel())
        scores_val_list.append(np.asarray(val_preds).ravel())
        scores_test_list.append(np.asarray(test_preds).ravel())
        last_model = model

    scores_ho = np.mean(scores_ho_list, axis=0)
    scores_val = np.mean(scores_val_list, axis=0)
    scores_test = np.mean(scores_test_list, axis=0)
    print(f"\n      best_iter: mean={np.mean(best_iters):.0f}, "
          f"min={min(best_iters)}, max={max(best_iters)}")

    # Сохраняем последнюю
    if last_model is not None:
        save_path = MODELS_DIR / "catboost_ranker_v14.cbm"
        last_model.save_model(str(save_path))
        print(f"      Модель сохранена: {save_path}")

    # ==== 8. Оценка ====
    print("\n[8/8] Оценка:")

    # df_sorted для top_k: CatBoost Pool сортирует по student_id
    ho_sorted = _sort_and_group(holdout_df)
    val_sorted = _sort_and_group(val_df)
    test_sorted = _sort_and_group(test_df)

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

    val_path = SUBMISSIONS_DIR / "val_submission_ltr_v14.csv"
    write_submission(rec_val, val_path)
    print(f"      Сохранено: {val_path}")

    rec_test = top_k_per_student(test_sorted, scores_test, k=5)
    test_path = SUBMISSIONS_DIR / "submission_v14.csv"
    write_submission(rec_test, test_path)
    print(f"      Сохранено: {test_path}")

    # Feature importance
    if last_model is not None:
        print("\n[Feature importance — top-15 (CatBoost PredictedValuesChange)]")
        try:
            imp_vals = last_model.get_feature_importance()
            imp_df = pd.DataFrame({
                "feature": FEATURE_COLS_V4,
                "importance": imp_vals,
            }).sort_values("importance", ascending=False).head(15)
            for _, row in imp_df.iterrows():
                print(f"      {row['feature']:<32}  {row['importance']:>8.2f}")
        except Exception as e:
            print(f"      Не удалось получить feature importance: {e}")

    print(f"\n[OK] Пайплайн завершён за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()