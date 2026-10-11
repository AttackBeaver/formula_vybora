"""
LTR v10b: чиним p1_prob (GroupKFold + average-of-folds) + ablation.

Что чиним относительно v10:
    1. GroupKFold по «голому» student_id (без префикса p1__/pmask__/...).
       Устраняет утечку: один и тот же студент в разных train-срезах
       больше не разъезжается по разным фолдам.
    2. Для holdout/val/test считаем p1_prob как СРЕДНЕЕ предсказаний
       5 фолд-моделей (а не full-fit на 90%). Калибровка train OOF
       (модель на 80%) ↔ eval (усреднение 5 моделей на 80%) — совпадают.
    3. Ablation: за один прогон обучаем ДВЕ LTR-модели:
         A — с p1_prob (32 признака, truncation_level=5)
         B — без p1_prob (31 признак, truncation_level=5) — эквивалент
             «v7 + trunc=5».
       По holdout сразу видно: p1_prob помогает или вредит.
    4. Всё остальное — срезы, веса, multi-seed, max_history_periods=2 —
       идентично v7/v10.

Запуск:
    python -m src.run_ltr_v10b
    python -m src.run_ltr_v10b --n-seeds 3
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

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
    "lambdarank_truncation_level": 5,
}

NUM_BOOST_ROUND = 2500
EARLY_STOPPING = 100

P1_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
    "num_threads": 0,
}
P1_N_FOLDS = 5
P1_NUM_BOOST = 800
P1_EARLY_STOP = 50

# v7 baseline для сравнения
V7_HOLDOUT = 0.5134


def _n_periods(r: dict[str, Any]) -> int:
    return len(r.get("history") or [])


def _add_weight(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = np.float32(weight)
    return df


def _strip_prefix(sid_series: pd.Series) -> pd.Series:
    """Убирает префикс вида 'p1__', 'pmask__', 'p2__', 'p0__', 'ho__'."""
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
# p1_prob: GroupKFold, OOF для train, average-of-folds для остальных
# ---------------------------------------------------------------------------
def compute_p1_prob_grouped(
    df_train: pd.DataFrame,
    df_others: list[pd.DataFrame],
    feature_cols: list[str],
    seed: int,
    n_folds: int = P1_N_FOLDS,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    p1_prob с учётом групп студентов.

    Группа = student_id без префикса среза. Один студент в разных train-срезах
    (p1__X, pmask__X, p2__X) попадает в одну группу → нет утечки в OOF.

    Для train: OOF (каждый фолд-модель предсказывает только свои out-of-fold
    строки).
    Для df_others: среднее предсказаний всех n_folds моделей — калибровка
    совпадает с OOF.

    Returns:
        (oof_train, [probs_per_other_df])
    """
    y_train = (df_train["priority"].to_numpy() == 1).astype(np.int32)
    X_train = df_train[feature_cols].copy()
    groups = _strip_prefix(df_train["student_id"].astype(str)).to_numpy()

    oof = np.zeros(len(df_train), dtype=np.float32)
    fold_models: list[lgb.Booster] = []

    gkf = GroupKFold(n_splits=n_folds)
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups), start=1):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y_train[va_idx]
        params = {**P1_PARAMS, "seed": seed + fold}
        dtrain = lgb.Dataset(
            X_tr, label=y_tr,
            categorical_feature=CATEGORICAL_FEATS,
            free_raw_data=False,
        )
        dval = lgb.Dataset(
            X_va, label=y_va,
            categorical_feature=CATEGORICAL_FEATS,
            reference=dtrain,
            free_raw_data=False,
        )
        booster = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=P1_NUM_BOOST,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(stopping_rounds=P1_EARLY_STOP, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        oof[va_idx] = booster.predict(X_va, num_iteration=booster.best_iteration)
        fold_models.append(booster)

    other_probs: list[np.ndarray] = []
    for df in df_others:
        X = df[feature_cols].copy()
        probs = np.zeros(len(X), dtype=np.float32)
        for booster in fold_models:
            probs += booster.predict(X, num_iteration=booster.best_iteration).astype(np.float32)
        probs /= float(n_folds)
        other_probs.append(probs)

    return oof, other_probs


# ---------------------------------------------------------------------------
# Прогон одного LTR-варианта (multi-seed) с оценкой
# ---------------------------------------------------------------------------
def run_ltr_variant(
    name: str,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    print(f"\n      ===== Вариант {name}: {len(feature_cols)} признаков =====")

    X_train, y_train, g_train, w_train, _ = prepare_xygw(
        train_df, feature_cols=feature_cols,
    )
    X_ho, y_ho, g_ho, w_ho, ho_sorted = prepare_xygw(
        holdout_df, feature_cols=feature_cols,
    )
    _, _, _, _, val_sorted = prepare_xygw(
        val_df, feature_cols=feature_cols,
    )
    _, _, _, _, test_sorted = prepare_xygw(
        test_df, feature_cols=feature_cols,
    )

    print(f"      Train: X={X_train.shape}, групп={len(g_train)}")

    scores_ho_list: list[np.ndarray] = []
    scores_val_list: list[np.ndarray] = []
    scores_test_list: list[np.ndarray] = []
    best_iters: list[int] = []
    last_model: Any = None

    for i, seed in enumerate(seeds):
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
        best_iters.append(model.best_iteration)
        print(f"        seed={seed}: best_iter={model.best_iteration}, "
              f"{time.time() - t0:.1f} сек")
        scores_ho_list.append(predict_scores(model, ho_sorted, feature_cols=feature_cols))
        scores_val_list.append(predict_scores(model, val_sorted, feature_cols=feature_cols))
        scores_test_list.append(predict_scores(model, test_sorted, feature_cols=feature_cols))
        last_model = model

    scores_ho = np.mean(scores_ho_list, axis=0)
    scores_val = np.mean(scores_val_list, axis=0)
    scores_test = np.mean(scores_test_list, axis=0)

    print(f"      best_iter: mean={np.mean(best_iters):.0f}, "
          f"min={min(best_iters)}, max={max(best_iters)}")

    return {
        "name": name,
        "scores_ho": scores_ho,
        "scores_val": scores_val,
        "scores_test": scores_test,
        "ho_sorted": ho_sorted,
        "val_sorted": val_sorted,
        "test_sorted": test_sorted,
        "last_model": last_model,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LTR v10b: p1_prob fix + ablation")
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()
    n_seeds = max(1, int(args.n_seeds))
    seeds = [BASE_SEED + i for i in range(n_seeds)]

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print(f"LTR v10b: p1_prob (GroupKFold + avg-of-folds) + ablation")
    print(f"n_seeds={n_seeds}, seeds={seeds}")
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

    # ==== 5. Веса, префиксы, категориальные ====
    print("\n[5/9] Веса, префиксы, категориальные...")
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

    # Категориальные кодируем ДО p1_prob — p1_prob фича числовая.
    all_df = pd.concat([train_df, holdout_df, val_df, test_df], ignore_index=True)
    all_df, _ = encode_categoricals(all_df, all_df, CATEGORICAL_FEATS)

    train_df = all_df.iloc[:n_train].reset_index(drop=True)
    holdout_df = all_df.iloc[n_train:n_train + n_ho].reset_index(drop=True)
    val_df = all_df.iloc[n_train + n_ho:n_train + n_ho + n_val].reset_index(drop=True)
    test_df = all_df.iloc[n_train + n_ho + n_val:].reset_index(drop=True)

    # ==== 6. p1_prob: GroupKFold + avg-of-folds ====
    print("\n[6/9] p1_prob (GroupKFold + average-of-folds)...")
    t_p1 = time.time()
    oof_p1, other_p1 = compute_p1_prob_grouped(
        df_train=train_df,
        df_others=[holdout_df, val_df, test_df],
        feature_cols=FEATURE_COLS_V4,
        seed=BASE_SEED,
        n_folds=P1_N_FOLDS,
    )
    train_df = train_df.copy()
    holdout_df = holdout_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["p1_prob"] = oof_p1
    holdout_df["p1_prob"] = other_p1[0]
    val_df["p1_prob"] = other_p1[1]
    test_df["p1_prob"] = other_p1[2]

    pos_rate = float((train_df["priority"] == 1).mean())
    print(f"      OOF p1_prob (train):  mean={oof_p1.mean():.4f}, "
          f"max={oof_p1.max():.4f}, pos_rate={pos_rate:.4f}")
    print(f"      p1_prob (holdout):    mean={other_p1[0].mean():.4f}, "
          f"max={other_p1[0].max():.4f}")
    print(f"      p1_prob (val):        mean={other_p1[1].mean():.4f}, "
          f"max={other_p1[1].max():.4f}")
    print(f"      p1_prob (test):       mean={other_p1[2].mean():.4f}, "
          f"max={other_p1[2].max():.4f}")
    print(f"      Время: {time.time() - t_p1:.1f} сек")

    FEATURE_COLS_V10: list[str] = FEATURE_COLS_V4 + ["p1_prob"]

    # ==== 7. Ablation: A (с p1_prob) vs B (без) ====
    print(f"\n[7/9] Ablation A vs B (по {n_seeds} сидов каждая)...")

    res_A = run_ltr_variant(
        name="A_with_p1_prob",
        train_df=train_df, holdout_df=holdout_df,
        val_df=val_df, test_df=test_df,
        feature_cols=FEATURE_COLS_V10,
        seeds=seeds,
    )
    res_B = run_ltr_variant(
        name="B_no_p1_prob",
        train_df=train_df, holdout_df=holdout_df,
        val_df=val_df, test_df=test_df,
        feature_cols=FEATURE_COLS_V4,
        seeds=seeds,
    )

    # ==== 8. Оценка holdout для A, B, энс. (0.5A+0.5B) ====
    print("\n[8/9] Оценка на holdout:")
    gt_ho = {sid: gt for sid, gt in val_gt_raw.items() if sid in holdout_ids}

    def _score_ho(scores: np.ndarray, df_sorted: pd.DataFrame) -> float:
        rec = top_k_per_student(df_sorted, scores, k=5)
        rec_clean = {sid.replace("ho__", ""): v for sid, v in rec.items()}
        m, _ = evaluate_pndcg5(rec_clean, gt_ho, k=5)
        return m

    m_A = _score_ho(res_A["scores_ho"], res_A["ho_sorted"])
    m_B = _score_ho(res_B["scores_ho"], res_B["ho_sorted"])

    # Нормализуем скоры перед усреднением: перевод в z-score по группе.
    def _z_norm(scores: np.ndarray, df_sorted: pd.DataFrame) -> np.ndarray:
        tmp = pd.DataFrame({
            "sid": df_sorted["student_id"].values,
            "s": scores,
        })
        means = tmp.groupby("sid")["s"].transform("mean").values
        stds = tmp.groupby("sid")["s"].transform("std").fillna(1.0).values
        stds = np.where(stds < 1e-9, 1.0, stds)
        return (scores - means) / stds

    zA = _z_norm(res_A["scores_ho"], res_A["ho_sorted"])
    zB = _z_norm(res_B["scores_ho"], res_B["ho_sorted"])
    m_AB = _score_ho(0.5 * zA + 0.5 * zB, res_A["ho_sorted"])

    print(f"      v7 (baseline):              {V7_HOLDOUT:.4f}")
    print(f"      A  (с p1_prob, trunc=5):    {m_A:.4f}   (Δ vs v7 = {m_A - V7_HOLDOUT:+.4f})")
    print(f"      B  (без p1_prob, trunc=5):  {m_B:.4f}   (Δ vs v7 = {m_B - V7_HOLDOUT:+.4f})")
    print(f"      0.5A + 0.5B (z-score):      {m_AB:.4f}   (Δ vs v7 = {m_AB - V7_HOLDOUT:+.4f})")

    # ==== 9. Сегментный анализ победителя + val/test submissions ====
    print("\n[9/9] Сегментный анализ и сохранение:")

    cold_ids_val = identify_cold_students(val)
    pop_weighted, _ = compute_popularity(train)

    candidates = [
        ("A", m_A, res_A),
        ("B", m_B, res_B),
    ]
    winner_name, winner_ho, winner = max(candidates, key=lambda x: x[1])
    print(f"      Победитель по holdout: {winner_name}  (holdout={winner_ho:.4f})")

    # val
    rec_val_ltr = top_k_per_student(winner["val_sorted"], winner["scores_val"], k=5)
    mean_ltr, cold_ltr, warm_ltr = _segment_means(
        rec_val_ltr, val_gt_raw, cold_ids_val,
    )
    fb_val = build_global_recommendations(val, pop_weighted, cold_ids_val, k=5)
    rec_val_fb = apply_fallback(rec_val_ltr, fb_val, k=5)
    mean_val, cold_val, warm_val = _segment_means(rec_val_fb, val_gt_raw, cold_ids_val)
    est_val = TEST_COLD_SHARE * cold_val + (1 - TEST_COLD_SHARE) * warm_val
    print(f"      Val LTR raw (winner): mean={mean_ltr:.4f}  cold={cold_ltr:.4f}  "
          f"warm={warm_ltr:.4f}")
    print(f"      Val + fallback:       mean={mean_val:.4f}  cold={cold_val:.4f}  "
          f"warm={warm_val:.4f}")
    print(f"      test_estimate:        {est_val:.4f}")

    val_path = SUBMISSIONS_DIR / "val_submission_ltr_v10b.csv"
    write_submission(rec_val_fb, val_path)
    print(f"      Сохранено: {val_path}")

    # test
    rec_test_ltr = top_k_per_student(winner["test_sorted"], winner["scores_test"], k=5)
    cold_ids_test = identify_cold_students(test)
    fb_test = build_global_recommendations(test, pop_weighted, cold_ids_test, k=5)
    rec_test_final = apply_fallback(rec_test_ltr, fb_test, k=5)
    test_path = SUBMISSIONS_DIR / "submission_v10b.csv"
    write_submission(rec_test_final, test_path)
    print(f"      Сохранено: {test_path}  (cold в test: {len(cold_ids_test)})")

    save_model(winner["last_model"], MODELS_DIR / f"lgbm_ranker_v10b_{winner_name}.txt")

    print("\n[Feature importance — top-15 победителя]")
    imp_cols = FEATURE_COLS_V10 if winner_name == "A" else FEATURE_COLS_V4
    imp = feature_importance_report(
        winner["last_model"], feature_cols=imp_cols, top_n=15,
    )
    for _, row in imp.iterrows():
        print(f"      {row['feature']:<28}  gain = {row['gain']:>12.0f}")

    print(f"\n[OK] Пайплайн завершён за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()