"""
run_ltr_v45_xgb.py — XGBoost + LightGBM ансамбль на V11-фичах.

ГИПОТЕЗА:
    v41 (LightGBM, V11) = 0.3604 public.
    XGBoost — другой алгоритм: другая реализация деревьев, другой
    rank:pairwise objective. Ошибки частично декореллируют.
    Ансамбль (z-score per student, среднее) должен дать +0.003..+0.008.

ПОЧЕМУ МОЖЕТ СРАБОТАТЬ:
    * LightGBM — leaf-wise, gradient-based one-side sampling.
    * XGBoost — level-wise, exact/hist, rank:pairwise.
    * Разные деревья → разные ошибки → ансамбль стабилизирует.

ПОЧЕМУ МОЖЕТ НЕ СРАБОТАТЬ:
    * Оба на одних V11-фичах → корреляция ошибок высокая (60-80%).
    * XGBoost slower на 2.7M строк.

ВАЖНО (XGBoost rank):
    XGBoost rank:pairwise требует веса per-group (длины n_groups), а не
    per-row. У нас weight одинаков внутри каждой группы (каждый срез
    имеет свой постоянный вес), поэтому агрегируем в xgb_to_dmatrix.

ПАРАМЕТРЫ:
    --n-seeds INT     (default 5)
    --xgb-rounds INT  (default 1000)

ЧТО СМОТРИМ:
    cold raw / warm raw для LGBM / XGB / ансамбля отдельно.
    Holdout — proxy warm, но на p2-срезе токсичен (см. README §6).

РЕШЕНИЕ:
    Ансамбль cold ≥ 0.337 И warm ≥ 0.487 → кандидат
    Иначе — XGBoost отбрасываем или пробуем без него.

УСТАНОВКА XGBoost:
    pip install xgboost
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

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
    FEATURE_COLS_V11,
    WEIGHTS_BY_LABEL,
    encode_categoricals,
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

WEIGHT_P1 = 1.0
WEIGHT_P0 = 0.5
WEIGHT_PMASK = 1.0
WEIGHT_P2 = 3.0

HOLDOUT_FRACTION = 0.20
HOLDOUT_SEED = 42
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
NUM_BOOST_ROUND_LGB = 2500
EARLY_STOPPING_LGB = 100

XGB_PARAMS: dict[str, Any] = {
    "objective": "rank:pairwise",
    "eval_metric": "ndcg@5",
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "nthread": -1,
    "verbosity": 1,
}
XGB_EARLY_STOP = 50


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


def _zscore_per_student(
    df: pd.DataFrame,
    scores: np.ndarray,
) -> np.ndarray:
    """Z-score внутри каждого студента."""
    out = np.zeros_like(scores, dtype=np.float64)
    work = pd.DataFrame({
        "student_id": df["student_id"].values,
        "score": scores,
    })
    for sid, grp in work.groupby("student_id", sort=False):
        idx = grp.index.to_numpy()
        s = grp["score"].to_numpy(dtype=np.float64)
        if len(s) <= 1:
            out[idx] = 0.0
            continue
        mu = s.mean()
        sd = s.std(ddof=0)
        if sd < 1e-12:
            out[idx] = 0.0
        else:
            out[idx] = (s - mu) / sd
    return out


# ---------------------------------------------------------------------------
# XGBoost custom eval (наш pNDCG@5)
# ---------------------------------------------------------------------------
def make_xgb_pndcg5_eval(group_sizes: np.ndarray):
    """Замыкание для custom eval XGBoost с нашей метрикой."""
    group_sizes = np.asarray(group_sizes, dtype=np.int64)

    def eval_fn(preds: np.ndarray, dtrain: xgb.DMatrix) -> tuple[str, float]:
        labels = dtrain.get_label().astype(np.int32)
        total = 0.0
        start = 0
        for size in group_sizes:
            end = start + int(size)
            p = preds[start:end]
            y = labels[start:end]
            order = np.argsort(-p, kind="stable")[:5]
            dcg = 0.0
            for pos, idx in enumerate(order, start=1):
                lab = int(y[idx])
                if lab > 0:
                    dcg += WEIGHTS_BY_LABEL[lab] / math.log2(pos + 1)
            total += min(1.0, dcg)
            start = end
        return "pndcg@5", float(total / max(1, len(group_sizes)))

    return eval_fn


def _aggregate_group_weights(g: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Агрегирует per-row weights в per-group weights.

    XGBoost rank:* требует weight длины n_groups. У нас weight одинаков
    внутри каждой группы (каждый срез имеет свой постоянный вес),
    поэтому берём первое значение в группе.
    """
    g_arr = np.asarray(g, dtype=np.int64)
    w_arr = np.asarray(w, dtype=np.float32)
    group_weights = np.zeros(len(g_arr), dtype=np.float32)
    start = 0
    for i, size in enumerate(g_arr):
        end = start + int(size)
        group_weights[i] = float(w_arr[start]) if end > start else 1.0
        start = end
    return group_weights


def xgb_to_dmatrix(
    X: pd.DataFrame,
    y: np.ndarray,
    g: np.ndarray,
    w: np.ndarray,
) -> xgb.DMatrix:
    """Создаёт DMatrix с поддержкой категориальных фич, группы и per-group весами."""
    group_weights = _aggregate_group_weights(g, w)
    d = xgb.DMatrix(
        X, label=y, weight=group_weights, enable_categorical=True,
    )
    d.set_group(np.asarray(g, dtype=np.int64))
    return d


def train_xgb_ranker(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    g_train: np.ndarray,
    w_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    g_val: np.ndarray,
    w_val: np.ndarray,
    seed: int,
    num_rounds: int = 1000,
) -> tuple[xgb.Booster, int]:
    """Обучает XGBoost ranker с early stopping на нашей pNDCG@5."""
    dtrain = xgb_to_dmatrix(X_train, y_train, g_train, w_train)
    dval = xgb_to_dmatrix(X_val, y_val, g_val, w_val)

    params = dict(XGB_PARAMS)
    params["seed"] = seed

    feval = make_xgb_pndcg5_eval(g_val)
    evals_result: dict[str, dict[str, list]] = {}

    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_rounds,
        evals=[(dval, "val")],
        custom_metric=feval,
        early_stopping_rounds=XGB_EARLY_STOP,
        verbose_eval=False,
        evals_result=evals_result,
    )
    best_iter = (model.best_iteration + 1
                 if model.best_iteration is not None else num_rounds)

    if "val" in evals_result and "pndcg@5" in evals_result["val"]:
        best_score = max(evals_result["val"]["pndcg@5"])
        print(f"      xgb seed={seed}: best_iter={best_iter}, "
              f"best pndcg@5={best_score:.4f}")

    return model, best_iter


def xgb_predict_scores(model: xgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """Предсказание скоров XGBoost (до best_iteration)."""
    d = xgb.DMatrix(X, enable_categorical=True)
    return model.predict(d, iteration_range=(0, model.best_iteration + 1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not HAS_XGB:
        print("=" * 78)
        print("ERROR: xgboost не установлен.")
        print("Установить: pip install xgboost")
        print("=" * 78)
        return

    ap = argparse.ArgumentParser(description="v45: XGBoost + LightGBM ансамбль")
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--xgb-rounds", type=int, default=1000)
    args = ap.parse_args()

    n_seeds = max(1, int(args.n_seeds))
    seeds = [BASE_SEED + i for i in range(n_seeds)]
    xgb_rounds = int(args.xgb_rounds)

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print(f"LTR v45: XGBoost + LightGBM ансамбль (n_seeds={n_seeds})")
    print(f"  feature_set = V11 ({len(FEATURE_COLS)} признаков)")
    print(f"  content = '{CONTENT_BACKEND}', p2={WEIGHT_P2}")
    print(f"  Seeds: {seeds}")
    print(f"  XGB rounds: {xgb_rounds}")
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
    print(f"\n[1/9] FeatureStats (content='{CONTENT_BACKEND}')...")
    stats = FeatureStats.fit(train, modules, content_backend=CONTENT_BACKEND)
    stats.add_val_solution_popularity(val_solution)

    # ==== 2. Train-срезы ====
    print("\n[2/9] Train-срезы p0/p1/pmask/p2...")
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
    print("\n[4/9] Val и test датасеты...")
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
    print("\n[5/9] Сборка + кодирование...")
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
    print("\n[6/9] Матрицы X, y, g, w...")
    X_train, y_train, g_train, w_train, _ = prepare_xygw(
        train_df, feature_cols=FEATURE_COLS)
    X_ho, y_ho, g_ho, w_ho, ho_sorted = prepare_xygw(
        holdout_df, feature_cols=FEATURE_COLS)
    X_val, y_val, g_val, w_val, val_sorted = prepare_xygw(
        val_df, feature_cols=FEATURE_COLS)
    _, _, _, _, test_sorted = prepare_xygw(test_df, feature_cols=FEATURE_COLS)

    print(f"      Train X={X_train.shape}, групп={len(g_train)}")

    # ==== 7. LightGBM (n seeds) ====
    print(f"\n[7/9] LightGBM {n_seeds} seeds...")
    lgb_scores_ho_list: list[np.ndarray] = []
    lgb_scores_val_list: list[np.ndarray] = []
    lgb_scores_test_list: list[np.ndarray] = []
    last_lgb_model = None

    for i, seed in enumerate(seeds):
        print(f"\n      --- LGBM seed={seed} ---")
        params = {**FAST_PARAMS, "seed": seed}
        t = time.time()
        model = train_lgbm_ranker(
            X_train, y_train, g_train,
            X_ho, y_ho, g_ho,
            params=params,
            num_boost_round=NUM_BOOST_ROUND_LGB,
            early_stopping_rounds=EARLY_STOPPING_LGB,
            log_every=1000,
            sample_weight_train=w_train,
            sample_weight_val=w_ho,
        )
        print(f"      lgb seed={seed}: {time.time() - t:.1f} сек, "
              f"best_iter={model.best_iteration}")
        lgb_scores_ho_list.append(
            predict_scores(model, ho_sorted, feature_cols=FEATURE_COLS))
        lgb_scores_val_list.append(
            predict_scores(model, val_sorted, feature_cols=FEATURE_COLS))
        lgb_scores_test_list.append(
            predict_scores(model, test_sorted, feature_cols=FEATURE_COLS))
        last_lgb_model = model

    lgb_scores_ho = np.mean(lgb_scores_ho_list, axis=0)
    lgb_scores_val = np.mean(lgb_scores_val_list, axis=0)
    lgb_scores_test = np.mean(lgb_scores_test_list, axis=0)

    if last_lgb_model is not None:
        save_model(last_lgb_model, MODELS_DIR / "lgbm_ranker_v45_xgb_lgb.txt")

    # ==== 8. XGBoost (n seeds) ====
    print(f"\n[8/9] XGBoost {n_seeds} seeds...")
    xgb_scores_ho_list: list[np.ndarray] = []
    xgb_scores_val_list: list[np.ndarray] = []
    xgb_scores_test_list: list[np.ndarray] = []
    last_xgb_model = None

    for i, seed in enumerate(seeds):
        print(f"\n      --- XGB seed={seed} ---")
        t = time.time()
        try:
            model, best_iter = train_xgb_ranker(
                X_train, y_train, g_train, w_train,
                X_ho, y_ho, g_ho, w_ho,
                seed=seed,
                num_rounds=xgb_rounds,
            )
            print(f"      xgb seed={seed}: {time.time() - t:.1f} сек, "
                  f"best_iter={best_iter}")

            xgb_scores_ho_list.append(xgb_predict_scores(model, X_ho))
            xgb_scores_val_list.append(xgb_predict_scores(model, X_val))
            xgb_scores_test_list.append(
                xgb_predict_scores(model, test_sorted[FEATURE_COLS])
            )
            last_xgb_model = model
        except Exception as e:
            print(f"      ! ОШИБКА xgb seed={seed}: {e}")
            continue

    if not xgb_scores_ho_list:
        print("\n!!! XGBoost не обучился ни разу. Ансамбль невозможен.")
        return

    xgb_scores_ho = np.mean(xgb_scores_ho_list, axis=0)
    xgb_scores_val = np.mean(xgb_scores_val_list, axis=0)
    xgb_scores_test = np.mean(xgb_scores_test_list, axis=0)

    if last_xgb_model is not None:
        last_xgb_model.save_model(str(MODELS_DIR / "xgb_ranker_v45.json"))
        print(f"[xgb] Модель сохранена: {MODELS_DIR / 'xgb_ranker_v45.json'}")

    # ==== 9. Оценка + ансамбль ====
    print("\n[9/9] Оценка + ансамбль...")

    cold_ids_val = identify_cold_students(val)

    # --- LGBM отдельно ---
    rec_lgb_ho = top_k_per_student(ho_sorted, lgb_scores_ho, k=5)
    rec_lgb_ho_clean = {sid.replace("ho__", ""): v for sid, v in rec_lgb_ho.items()}
    gt_ho = {sid: gt for sid, gt in val_gt_raw.items() if sid in holdout_ids}
    m_lgb_ho, _ = evaluate_pndcg5(rec_lgb_ho_clean, gt_ho, k=5)

    rec_lgb_val = top_k_per_student(val_sorted, lgb_scores_val, k=5)
    _, m_lgb_cold, m_lgb_warm = _segment_means(rec_lgb_val, val_gt_raw, cold_ids_val)

    print(f"\n  LightGBM:")
    print(f"    holdout = {m_lgb_ho:.4f}  cold raw = {m_lgb_cold:.4f}  "
          f"warm raw = {m_lgb_warm:.4f}")

    # --- XGB отдельно ---
    rec_xgb_ho = top_k_per_student(ho_sorted, xgb_scores_ho, k=5)
    rec_xgb_ho_clean = {sid.replace("ho__", ""): v for sid, v in rec_xgb_ho.items()}
    m_xgb_ho, _ = evaluate_pndcg5(rec_xgb_ho_clean, gt_ho, k=5)

    rec_xgb_val = top_k_per_student(val_sorted, xgb_scores_val, k=5)
    _, m_xgb_cold, m_xgb_warm = _segment_means(rec_xgb_val, val_gt_raw, cold_ids_val)

    print(f"\n  XGBoost:")
    print(f"    holdout = {m_xgb_ho:.4f}  cold raw = {m_xgb_cold:.4f}  "
          f"warm raw = {m_xgb_warm:.4f}")

    # --- Ансамбль (z-score avg) ---
    print(f"\n  Ансамбль (z-score avg):")
    lgb_z_ho = _zscore_per_student(ho_sorted, lgb_scores_ho)
    xgb_z_ho = _zscore_per_student(ho_sorted, xgb_scores_ho)
    ens_ho = 0.5 * lgb_z_ho + 0.5 * xgb_z_ho
    rec_ens_ho = top_k_per_student(ho_sorted, ens_ho, k=5)
    rec_ens_ho_clean = {sid.replace("ho__", ""): v for sid, v in rec_ens_ho.items()}
    m_ens_ho, _ = evaluate_pndcg5(rec_ens_ho_clean, gt_ho, k=5)

    lgb_z_val = _zscore_per_student(val_sorted, lgb_scores_val)
    xgb_z_val = _zscore_per_student(val_sorted, xgb_scores_val)
    ens_val = 0.5 * lgb_z_val + 0.5 * xgb_z_val
    rec_ens_val = top_k_per_student(val_sorted, ens_val, k=5)
    _, m_ens_cold, m_ens_warm = _segment_means(rec_ens_val, val_gt_raw, cold_ids_val)

    print(f"    holdout = {m_ens_ho:.4f}  cold raw = {m_ens_cold:.4f}  "
          f"warm raw = {m_ens_warm:.4f}")

    # Сохраняем test-сабмиты (без fallback — как v41 champion)
    lgb_z_test = _zscore_per_student(test_sorted, lgb_scores_test)
    xgb_z_test = _zscore_per_student(test_sorted, xgb_scores_test)
    ens_test = 0.5 * lgb_z_test + 0.5 * xgb_z_test

    rec_lgb_test = top_k_per_student(test_sorted, lgb_scores_test, k=5)
    rec_xgb_test = top_k_per_student(test_sorted, xgb_scores_test, k=5)
    rec_ens_test = top_k_per_student(test_sorted, ens_test, k=5)

    write_submission(rec_lgb_test, SUBMISSIONS_DIR / "submission_v45_lgb_raw.csv")
    write_submission(rec_xgb_test, SUBMISSIONS_DIR / "submission_v45_xgb_raw.csv")
    write_submission(rec_ens_test, SUBMISSIONS_DIR / "submission_v45_ens_raw.csv")

    # ==== Вердикт ====
    print("\n" + "=" * 78)
    print("ИТОГ: v45 (LightGBM + XGBoost + ансамбль)")
    print("=" * 78)
    print(f"  v41 (champion):          cold=0.3372  warm=0.4839  public=0.3604")
    print(f"  LightGBM (v45):          cold={m_lgb_cold:.4f}  warm={m_lgb_warm:.4f}")
    print(f"  XGBoost (v45):           cold={m_xgb_cold:.4f}  warm={m_xgb_warm:.4f}")
    print(f"  Ансамбль (v45):          cold={m_ens_cold:.4f}  warm={m_ens_warm:.4f}")
    print()

    cold_best = max(m_lgb_cold, m_xgb_cold, m_ens_cold)
    warm_best = max(m_lgb_warm, m_xgb_warm, m_ens_warm)

    if m_ens_cold >= 0.337 and m_ens_warm >= 0.487:
        print(f"  ✅ РЕКОМЕНДАЦИЯ: ансамбль cold={m_ens_cold:.4f}≥0.337, "
              f"warm={m_ens_warm:.4f}≥0.487 — КАНДИДАТ")
        print(f"     файл: submission_v45_ens_raw.csv")
    elif m_ens_cold >= 0.337 and m_ens_warm >= 0.483:
        print(f"  ⚠️  ОСТОРОЖНО: ансамбль warm={m_ens_warm:.4f} ∈ [0.483, 0.487) — "
              f"на грани")
        print(f"     файл: submission_v45_ens_raw.csv")
    elif cold_best >= 0.337 and warm_best >= 0.487:
        if m_xgb_warm > m_lgb_warm:
            print(f"  ⚠️  XGBoost лучше ансамбля на warm — попробовать только XGB")
        else:
            print(f"  ⚠️  LGBM лучше — ансамбль не помогает, отбрасываем")
    else:
        print(f"  ❌ НЕ САБМИТИМ: ни одна конфигурация не дотягивает")
        print(f"     cold_best={cold_best:.4f}, warm_best={warm_best:.4f}")
    print("=" * 78)
    print(f"\n[OK] v45 завершён за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()