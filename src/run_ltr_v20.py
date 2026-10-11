"""
LTR v20: ансамбль трёх v7-семейных конфигураций.

Три конфигурации:
    M1 = v7 (31 признак FEATURE_COLS_V4, WEIGHT_P2=3.0)
    M2 = без rank-фич (29 признаков, WEIGHT_P2=3.0)
    M3 = v7-style (31 признак, WEIGHT_P2=1.0)

Агрегация:
    z-score per student для каждой конфигурации;
    финальный score = w1*z(M1) + w2*z(M2) + w3*z(M3).

Архитектура:
    Для каждой конфигурации train/holdout/val/test кодируются ОДНИМ
    вызовом encode_categoricals — иначе не совпадают dtype у
    categorical features и LightGBM predict падает.

Запуск:
    python -m src.run_ltr_v20 --n-seeds 5
    python -m src.run_ltr_v20 --n-seeds 5 --weights 0.5 0.3 0.2
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

REMOVE_FOR_M2 = ["pop_rank_in_available", "pop_rank_frac"]
FEATURE_COLS_M2: list[str] = [
    c for c in FEATURE_COLS_V4 if c not in REMOVE_FOR_M2
]


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


def _z_normalize_by_student(
    df_sorted: pd.DataFrame,
    scores: np.ndarray,
) -> np.ndarray:
    """Z-нормализация скоров внутри каждого student_id."""
    tmp = pd.DataFrame({
        "sid": df_sorted["student_id"].values,
        "s": scores.astype(np.float64),
    })
    means = tmp.groupby("sid")["s"].transform("mean").values
    stds = tmp.groupby("sid")["s"].transform("std").fillna(1.0).values
    stds = np.where(stds < 1e-9, 1.0, stds)
    return ((scores - means) / stds).astype(np.float64)


# ---------------------------------------------------------------------------
# Сборка train_df для одной конфигурации
# ---------------------------------------------------------------------------
def _build_train_df_for_config(
    stats: FeatureStats,
    train_students_p1: list[dict],
    train_df_p0: pd.DataFrame,
    p2_df: pd.DataFrame,
    weight_p2: float,
) -> pd.DataFrame:
    """
    Собирает train_df для одной конфигурации (с заданным WEIGHT_P2),
    кодирует его СОБСТВЕННЫМ вызовом encode_categoricals (внутри
    main'а не получится, потому что у M2 другой набор фич).
    """
    train_df_p1 = build_dataset(
        train_students_p1, stats,
        candidates_fn=make_candidates_train_period2(stats),
        target_fn=target_period_2,
        past_periods=[1],
    )
    train_df_pmask = build_dataset(
        train_students_p1, stats,
        candidates_fn=make_candidates_train_period2(stats),
        target_fn=target_period_2,
        past_periods=[1],
        mask_profile=True,
    )
    train_df_p0_local = train_df_p0.copy()
    p2_local = p2_df.copy()

    train_df_p1 = _add_weight(train_df_p1, WEIGHT_P1)
    train_df_p0_local = _add_weight(train_df_p0_local, WEIGHT_P0)
    train_df_pmask = _add_weight(train_df_pmask, WEIGHT_PMASK)
    p2_local = _add_weight(p2_local, weight_p2)

    train_df_p1["student_id"] = "p1__" + train_df_p1["student_id"].astype(str)
    train_df_p0_local["student_id"] = "p0__" + train_df_p0_local["student_id"].astype(str)
    train_df_pmask["student_id"] = "pmask__" + train_df_pmask["student_id"].astype(str)
    p2_local["student_id"] = "p2__" + p2_local["student_id"].astype(str)

    parts = [train_df_p1, train_df_p0_local, train_df_pmask, p2_local]
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Обучение одной конфигурации
# ---------------------------------------------------------------------------
def _train_one_config(
    name: str,
    feature_cols: list[str],
    weight_p2: float,
    stats: FeatureStats,
    train_students_p1: list[dict],
    train_df_p0: pd.DataFrame,
    p2_df: pd.DataFrame,
    holdout_records: list[dict],
    val_students: list[dict],
    test_students: list[dict],
    val_gt_raw: dict[str, str],
    seeds: list[int],
) -> dict[str, Any]:
    """
    Обучает ансамбль моделей для одной конфигурации.

    Внутри: строит train/holdout/val/test, кодирует категориальные
    одним вызовом (чтобы dtype совпадал), обучает, возвращает
    скоры на каждом датасете + последний booster.
    """
    # --- train ---
    train_df = _build_train_df_for_config(
        stats, train_students_p1, train_df_p0, p2_df, weight_p2=weight_p2,
    )

    # --- holdout / val / test (с префиксами, чтобы общее кодирование не путалось) ---
    holdout_df = build_dataset(
        holdout_records, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )
    holdout_df = _add_weight(holdout_df, 1.0)
    holdout_df["student_id"] = "ho__" + holdout_df["student_id"].astype(str)

    val_df = build_dataset(
        val_students, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )
    val_df = _add_weight(val_df, 1.0)
    val_df["student_id"] = "vl__" + val_df["student_id"].astype(str)

    test_df = build_dataset(
        test_students, stats,
        candidates_fn=candidates_available,
        target_fn=None,
        past_periods=None,
        max_history_periods=2,
    )
    test_df = _add_weight(test_df, 1.0)
    test_df["student_id"] = "ts__" + test_df["student_id"].astype(str)

    # --- ЕДИНОЕ кодирование категориальных ---
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

    # --- Матрицы ---
    X_train, y_train, g_train, w_train, _ = prepare_xygw(
        train_df, feature_cols=feature_cols,
    )
    X_ho, y_ho, g_ho, w_ho, holdout_sorted = prepare_xygw(
        holdout_df, feature_cols=feature_cols,
    )
    _, _, _, _, val_sorted = prepare_xygw(
        val_df, feature_cols=feature_cols,
    )
    _, _, _, _, test_sorted = prepare_xygw(
        test_df, feature_cols=feature_cols,
    )

    print(f"      [{name}] train X={X_train.shape}, групп={len(g_train)}, "
          f"w_p2={weight_p2}")

    ho_scores_list: list[np.ndarray] = []
    val_scores_list: list[np.ndarray] = []
    test_scores_list: list[np.ndarray] = []
    best_iters: list[int] = []
    last_model: Any = None

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
        print(f"      [{name}] seed={seed}: best_iter={model.best_iteration}, "
              f"{time.time() - t_train:.1f} сек")
        ho_scores_list.append(predict_scores(model, holdout_sorted, feature_cols=feature_cols))
        val_scores_list.append(predict_scores(model, val_sorted, feature_cols=feature_cols))
        test_scores_list.append(predict_scores(model, test_sorted, feature_cols=feature_cols))
        last_model = model

    return {
        "name": name,
        "ho_scores": np.mean(ho_scores_list, axis=0),
        "val_scores": np.mean(val_scores_list, axis=0),
        "test_scores": np.mean(test_scores_list, axis=0),
        "holdout_sorted": holdout_sorted,
        "val_sorted": val_sorted,
        "test_sorted": test_sorted,
        "last_model": last_model,
        "best_iters": best_iters,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="LTR v20: ensemble of 3 configs")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument(
        "--weights", type=float, nargs=3, default=[0.5, 0.3, 0.2],
        help="Веса для (M1, M2, M3)",
    )
    args = parser.parse_args()
    n_seeds = max(1, int(args.n_seeds))
    seeds = [BASE_SEED + i for i in range(n_seeds)]
    w1, w2, w3 = args.weights
    total_w = w1 + w2 + w3
    w1, w2, w3 = w1 / total_w, w2 / total_w, w3 / total_w

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print(f"LTR v20: ensemble of 3 configs, n_seeds={n_seeds}")
    print(f"Веса: M1={w1:.2f} (v7-style), M2={w2:.2f} (no rank), M3={w3:.2f} (w_p2=1)")
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

    # ==== 2. Базовые срезы ====
    print("\n[2/8] Базовые train-срезы (p0 и p2 — один раз)...")
    train_students_p1 = [
        s for s in train
        if any(h.get("period") == 2 for h in s.get("history", []))
    ]
    train_df_p0_raw = build_dataset(
        [s for s in train if any(h.get("period") == 1 for h in s.get("history", []))],
        stats,
        candidates_fn=make_candidates_train_period1(stats),
        target_fn=target_period_1,
        past_periods=[],
    )
    print(f"      p0={len(train_df_p0_raw):,}")

    # holdout + p2
    val_p2 = [s for s in val if _n_periods(s) == 2]
    rng = random.Random(HOLDOUT_SEED)
    shuffled = list(val_p2)
    rng.shuffle(shuffled)
    n_holdout = int(len(shuffled) * HOLDOUT_FRACTION)
    holdout_records = shuffled[:n_holdout]
    p2_records = shuffled[n_holdout:]
    holdout_ids = {s["id"] for s in holdout_records}
    print(f"      p2={len(p2_records)}  holdout={len(holdout_records)}")

    p2_df_raw = build_dataset(
        p2_records, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )

    # ==== 3. Обучение M1 ====
    print("\n[3/8] Обучение M1 (31 признак, w_p2=3.0)...")
    res_M1 = _train_one_config(
        name="M1",
        feature_cols=FEATURE_COLS_V4,
        weight_p2=3.0,
        stats=stats,
        train_students_p1=train_students_p1,
        train_df_p0=train_df_p0_raw,
        p2_df=p2_df_raw,
        holdout_records=holdout_records,
        val_students=val,
        test_students=test,
        val_gt_raw=val_gt_raw,
        seeds=seeds,
    )
    save_model(res_M1["last_model"], MODELS_DIR / "lgbm_v20_M1.txt")

    # ==== 4. Обучение M2 ====
    print("\n[4/8] Обучение M2 (29 признаков, w_p2=3.0)...")
    res_M2 = _train_one_config(
        name="M2",
        feature_cols=FEATURE_COLS_M2,
        weight_p2=3.0,
        stats=stats,
        train_students_p1=train_students_p1,
        train_df_p0=train_df_p0_raw,
        p2_df=p2_df_raw,
        holdout_records=holdout_records,
        val_students=val,
        test_students=test,
        val_gt_raw=val_gt_raw,
        seeds=seeds,
    )
    save_model(res_M2["last_model"], MODELS_DIR / "lgbm_v20_M2.txt")

    # ==== 5. Обучение M3 ====
    print("\n[5/8] Обучение M3 (31 признак, w_p2=1.0)...")
    res_M3 = _train_one_config(
        name="M3",
        feature_cols=FEATURE_COLS_V4,
        weight_p2=1.0,
        stats=stats,
        train_students_p1=train_students_p1,
        train_df_p0=train_df_p0_raw,
        p2_df=p2_df_raw,
        holdout_records=holdout_records,
        val_students=val,
        test_students=test,
        val_gt_raw=val_gt_raw,
        seeds=seeds,
    )
    save_model(res_M3["last_model"], MODELS_DIR / "lgbm_ranker_v20_M3.txt")

    # ==== 6. Оценка на holdout ====
    print("\n[6/8] Оценка на holdout:")

    holdout_sorted = res_M1["holdout_sorted"]
    gt_ho = {sid: gt for sid, gt in val_gt_raw.items() if sid in holdout_ids}

    def _score_ho(scores: np.ndarray) -> float:
        rec = top_k_per_student(holdout_sorted, scores, k=5)
        rec_clean = {sid.replace("ho__", ""): v for sid, v in rec.items()}
        m, _ = evaluate_pndcg5(rec_clean, gt_ho, k=5)
        return m

    s_M1 = _score_ho(res_M1["ho_scores"])
    s_M2 = _score_ho(res_M2["ho_scores"])
    s_M3 = _score_ho(res_M3["ho_scores"])

    # Ансамбль
    z1 = _z_normalize_by_student(holdout_sorted, res_M1["ho_scores"])
    z2 = _z_normalize_by_student(holdout_sorted, res_M2["ho_scores"])
    z3 = _z_normalize_by_student(holdout_sorted, res_M3["ho_scores"])

    ens_raw = w1 * res_M1["ho_scores"] + w2 * res_M2["ho_scores"] + w3 * res_M3["ho_scores"]
    s_ens_raw = _score_ho(ens_raw)

    ens_z = w1 * z1 + w2 * z2 + w3 * z3
    s_ens_z = _score_ho(ens_z)

    ens_z_eq = (z1 + z2 + z3) / 3.0
    s_ens_z_eq = _score_ho(ens_z_eq)

    print(f"\n      M1 (v7-style):        {s_M1:.4f}")
    print(f"      M2 (no rank):         {s_M2:.4f}")
    print(f"      M3 (w_p2=1):          {s_M3:.4f}")
    print(f"      --- ансамбли:")
    print(f"      raw weighted:         {s_ens_raw:.4f}  "
          f"(Δ vs v7 = {s_ens_raw - V7_HOLDOUT:+.4f})")
    print(f"      z-weighted {w1:.2f}/{w2:.2f}/{w3:.2f}:  {s_ens_z:.4f}  "
          f"(Δ vs v7 = {s_ens_z - V7_HOLDOUT:+.4f})")
    print(f"      z-equal:              {s_ens_z_eq:.4f}  "
          f"(Δ vs v7 = {s_ens_z_eq - V7_HOLDOUT:+.4f})")

    candidates = [
        ("raw_weighted", s_ens_raw, "raw"),
        (f"z_weighted_{w1:.2f}_{w2:.2f}_{w3:.2f}", s_ens_z, "z_weighted"),
        ("z_equal", s_ens_z_eq, "z_equal"),
    ]
    best_name, best_ho, best_mode = max(candidates, key=lambda x: x[1])
    print(f"\n      Лучший вариант: {best_name} (holdout={best_ho:.4f})")

    # ==== 7. Применяем лучший вариант к val и test ====
    print("\n[7/8] Валидация + test submission:")

    # У каждой конфигурации свой holdout_sorted/val_sorted/test_sorted
    # (dtype категориальных разный). Согласуем через общий порядок строк:
    # сортировка по (student_id, module_id) одна и та же во всех конфигурациях
    # для одного и того же датасета (val/test) — используем val_sorted из M1.

    val_sorted_M1 = res_M1["val_sorted"]
    val_sorted_M2 = res_M2["val_sorted"]
    val_sorted_M3 = res_M3["val_sorted"]

    # sanity check: они должны совпадать по (student_id, module_id)
    def _key(df: pd.DataFrame) -> list[tuple[str, str]]:
        return list(zip(df["student_id"].astype(str), df["module_id"].astype(str)))

    assert _key(val_sorted_M1) == _key(val_sorted_M2) == _key(val_sorted_M3), \
        "val_sorted разошёлся между конфигурациями"
    assert _key(res_M1["test_sorted"]) == _key(res_M2["test_sorted"]) == _key(res_M3["test_sorted"]), \
        "test_sorted разошёлся между конфигурациями"

    val_sorted = val_sorted_M1
    test_sorted = res_M1["test_sorted"]

    if best_mode == "raw":
        val_ens = w1 * res_M1["val_scores"] + w2 * res_M2["val_scores"] + w3 * res_M3["val_scores"]
        test_ens = w1 * res_M1["test_scores"] + w2 * res_M2["test_scores"] + w3 * res_M3["test_scores"]
    elif best_mode == "z_weighted":
        zv1 = _z_normalize_by_student(val_sorted, res_M1["val_scores"])
        zv2 = _z_normalize_by_student(val_sorted, res_M2["val_scores"])
        zv3 = _z_normalize_by_student(val_sorted, res_M3["val_scores"])
        val_ens = w1 * zv1 + w2 * zv2 + w3 * zv3

        zt1 = _z_normalize_by_student(test_sorted, res_M1["test_scores"])
        zt2 = _z_normalize_by_student(test_sorted, res_M2["test_scores"])
        zt3 = _z_normalize_by_student(test_sorted, res_M3["test_scores"])
        test_ens = w1 * zt1 + w2 * zt2 + w3 * zt3
    else:  # z_equal
        zv1 = _z_normalize_by_student(val_sorted, res_M1["val_scores"])
        zv2 = _z_normalize_by_student(val_sorted, res_M2["val_scores"])
        zv3 = _z_normalize_by_student(val_sorted, res_M3["val_scores"])
        val_ens = (zv1 + zv2 + zv3) / 3.0

        zt1 = _z_normalize_by_student(test_sorted, res_M1["test_scores"])
        zt2 = _z_normalize_by_student(test_sorted, res_M2["test_scores"])
        zt3 = _z_normalize_by_student(test_sorted, res_M3["test_scores"])
        test_ens = (zt1 + zt2 + zt3) / 3.0

    cold_ids_val = identify_cold_students(val)
    rec_val = top_k_per_student(val_sorted, val_ens, k=5)
    # Снимаем префикс vl__
    rec_val_clean = {sid.replace("vl__", ""): v for sid, v in rec_val.items()}
    mean_val, cold_val, warm_val = _segment_means(rec_val_clean, val_gt_raw, cold_ids_val)
    est_val = TEST_COLD_SHARE * cold_val + (1 - TEST_COLD_SHARE) * warm_val
    print(f"\n      Val (ансамбль): mean={mean_val:.4f}  cold={cold_val:.4f}  "
          f"warm={warm_val:.4f}")
    print(f"      test_estimate = {est_val:.4f}")

    val_path = SUBMISSIONS_DIR / "val_submission_ltr_v20.csv"
    write_submission(rec_val_clean, val_path)
    print(f"      Сохранено: {val_path}")

    rec_test = top_k_per_student(test_sorted, test_ens, k=5)
    rec_test_clean = {sid.replace("ts__", ""): v for sid, v in rec_test.items()}
    test_path = SUBMISSIONS_DIR / "submission_v20.csv"
    write_submission(rec_test_clean, test_path)
    print(f"      Сохранено: {test_path}")

    # ==== 8. Итог ====
    print("\n[8/8] Итог:")
    print(f"      holdout M1={s_M1:.4f}  M2={s_M2:.4f}  M3={s_M3:.4f}")
    print(f"      holdout лучший ансамбль={best_ho:.4f} ({best_name})")
    print(f"      Δ vs v7={best_ho - V7_HOLDOUT:+.4f}")
    print(f"      Val ансамбля: mean={mean_val:.4f} cold={cold_val:.4f} warm={warm_val:.4f}")
    print(f"      test_estimate ансамбля = {est_val:.4f}")

    print(f"\n[OK] Пайплайн завершён за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()