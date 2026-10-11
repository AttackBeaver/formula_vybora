"""
LTR v25: z-ансамбль v23 + v24.

Мотивация:
    v23 = fix периодов + val_p1 (w=1.0)
    v24 = fix периодов (без val_p1)

    Обе:
        - FEATURE_COLS_V4 (31);
        - max_history_periods=2 + truncation=15;
        - одинаковые val_df/test_df после fix периодов.

    Holdout: v23=0.5225, v24=0.5126 (усреднённые по 5 seeds).
    Cold:    v23=0.2959, v24=0.2830.
    test_est: v23=0.4428, v24=0.4317.

Ансамбль:
    score = 0.5 * z(score_v23) + 0.5 * z(score_v24)
    z — нормализация внутри student_id.

    Обе конфигурации обучаются 5 seeds, скоры усредняются по seeds
    ВНУТРИ конфигурации, потом z-ансамбль.

Запуск:
    python -m src.run_ltr_v25 --n-seeds 5
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
WEIGHT_VALP1 = 1.0  # только для v23-конфигурации

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
V23_HOLDOUT = 0.5225
V23_COLD_VAL = 0.2959
V24_HOLDOUT = 0.5126
V24_COLD_VAL = 0.2830


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


def _z_normalize_by_student(
    df_sorted: pd.DataFrame,
    scores: np.ndarray,
) -> np.ndarray:
    tmp = pd.DataFrame({
        "sid": df_sorted["student_id"].values,
        "s": scores.astype(np.float64),
    })
    means = tmp.groupby("sid")["s"].transform("mean").values
    stds = tmp.groupby("sid")["s"].transform("std").fillna(1.0).values
    stds = np.where(stds < 1e-9, 1.0, stds)
    return ((scores - means) / stds).astype(np.float64)


# ---------------------------------------------------------------------------
# Fix периодов
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Обучение одной конфигурации (5 seeds, усреднение)
# ---------------------------------------------------------------------------
def train_config(
    name: str,
    include_valp1: bool,
    stats: FeatureStats,
    train_students_p1: list[dict],
    train_students_p0: list[dict],
    p2_records: list[dict],
    val_p1_records: list[dict],
    holdout_records: list[dict],
    val_students: list[dict],
    test_students: list[dict],
    val_gt_raw: dict[str, str],
    seeds: list[int],
) -> dict[str, Any]:
    """Обучает ансамбль 5 seeds. Возвращает скоры на ho/val/test + sorted df."""
    print(f"\n      ===== {name} (include_valp1={include_valp1}) =====")

    # --- train-срезы ---
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

    # holdout / val / test
    holdout_df = build_dataset(
        holdout_records, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )
    val_df = build_dataset(
        val_students, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )
    test_df = build_dataset(
        test_students, stats,
        candidates_fn=candidates_available,
        target_fn=None,
        past_periods=None,
        max_history_periods=2,
    )

    # fix периодов (no-op для train/holdout/val, важно для test)
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
    val_df = add_correct_period_features(
        val_df, val_students, stats.pop_by_period, past_periods=[1, 2])
    test_df = add_correct_period_features(
        test_df, test_students, stats.pop_by_period, past_periods=None)
    if valp1_df is not None:
        valp1_df = add_correct_period_features(
            valp1_df, val_p1_records, stats.pop_by_period, past_periods=[1])

    # веса
    train_df_p1 = _add_weight(train_df_p1, WEIGHT_P1)
    train_df_p0 = _add_weight(train_df_p0, WEIGHT_P0)
    train_df_pmask = _add_weight(train_df_pmask, WEIGHT_PMASK)
    p2_df = _add_weight(p2_df, WEIGHT_P2)
    holdout_df = _add_weight(holdout_df, 1.0)
    val_df = _add_weight(val_df, 1.0)
    test_df = _add_weight(test_df, 1.0)
    if valp1_df is not None:
        valp1_df = _add_weight(valp1_df, WEIGHT_VALP1)

    # префиксы
    train_df_p1["student_id"] = "p1__" + train_df_p1["student_id"].astype(str)
    train_df_p0["student_id"] = "p0__" + train_df_p0["student_id"].astype(str)
    train_df_pmask["student_id"] = "pmask__" + train_df_pmask["student_id"].astype(str)
    p2_df["student_id"] = "p2__" + p2_df["student_id"].astype(str)
    holdout_df["student_id"] = "ho__" + holdout_df["student_id"].astype(str)
    val_df["student_id"] = "vl__" + val_df["student_id"].astype(str)
    test_df["student_id"] = "ts__" + test_df["student_id"].astype(str)
    if valp1_df is not None:
        valp1_df["student_id"] = "vp1__" + valp1_df["student_id"].astype(str)

    parts = [train_df_p1, train_df_p0, train_df_pmask, p2_df]
    if valp1_df is not None:
        parts.append(valp1_df)
    train_df = pd.concat(parts, ignore_index=True)

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

    X_train, y_train, g_train, w_train, _ = prepare_xygw(
        train_df, feature_cols=FEATURE_COLS_V4)
    X_ho, y_ho, g_ho, w_ho, ho_sorted = prepare_xygw(
        holdout_df, feature_cols=FEATURE_COLS_V4)
    _, _, _, _, val_sorted = prepare_xygw(
        val_df, feature_cols=FEATURE_COLS_V4)
    _, _, _, _, test_sorted = prepare_xygw(
        test_df, feature_cols=FEATURE_COLS_V4)

    print(f"      [{name}] train X={X_train.shape}, групп={len(g_train)}")

    ho_list: list[np.ndarray] = []
    val_list: list[np.ndarray] = []
    test_list: list[np.ndarray] = []
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
        ho_list.append(predict_scores(model, ho_sorted, feature_cols=FEATURE_COLS_V4))
        val_list.append(predict_scores(model, val_sorted, feature_cols=FEATURE_COLS_V4))
        test_list.append(predict_scores(model, test_sorted, feature_cols=FEATURE_COLS_V4))
        last_model = model

    return {
        "name": name,
        "ho_scores": np.mean(ho_list, axis=0),
        "val_scores": np.mean(val_list, axis=0),
        "test_scores": np.mean(test_list, axis=0),
        "ho_sorted": ho_sorted,
        "val_sorted": val_sorted,
        "test_sorted": test_sorted,
        "last_model": last_model,
        "best_iters": best_iters,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LTR v25: z-ensemble v23+v24")
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()
    n_seeds = max(1, int(args.n_seeds))
    seeds = [BASE_SEED + i for i in range(n_seeds)]

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print(f"LTR v25: z-ensemble v23+v24, n_seeds={n_seeds}")
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
    print("\n[1/6] Статистики...")
    stats = FeatureStats.fit(train, modules)
    stats.add_val_solution_popularity(val_solution)

    # ==== 2. Подготовка списков студентов ====
    print("\n[2/6] Подготовка списков студентов...")
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
    val_p1_ids = {s["id"] for s in val_p1_records}
    print(f"      p1={len(train_students_p1)}  p0={len(train_students_p0)}  "
          f"p2={len(p2_records)}  holdout={len(holdout_records)}  "
          f"val_p1={len(val_p1_records)}")

    # ==== 3. Обучение v23 ====
    print("\n[3/6] Обучение v23 (fix + val_p1)...")
    res_v23 = train_config(
        name="v23",
        include_valp1=True,
        stats=stats,
        train_students_p1=train_students_p1,
        train_students_p0=train_students_p0,
        p2_records=p2_records,
        val_p1_records=val_p1_records,
        holdout_records=holdout_records,
        val_students=val,
        test_students=test,
        val_gt_raw=val_gt_raw,
        seeds=seeds,
    )
    save_model(res_v23["last_model"], MODELS_DIR / "lgbm_v25_v23.txt")

    # ==== 4. Обучение v24 ====
    print("\n[4/6] Обучение v24 (fix only)...")
    res_v24 = train_config(
        name="v24",
        include_valp1=False,
        stats=stats,
        train_students_p1=train_students_p1,
        train_students_p0=train_students_p0,
        p2_records=p2_records,
        val_p1_records=val_p1_records,
        holdout_records=holdout_records,
        val_students=val,
        test_students=test,
        val_gt_raw=val_gt_raw,
        seeds=seeds,
    )
    save_model(res_v24["last_model"], MODELS_DIR / "lgbm_v25_v24.txt")

    # ==== 5. Оценка каждой и ансамбля ====
    print("\n[5/6] Оценка:")
    gt_ho = {sid: gt for sid, gt in val_gt_raw.items() if sid in holdout_ids}

    def _score_ho(scores: np.ndarray, sorted_df: pd.DataFrame) -> float:
        rec = top_k_per_student(sorted_df, scores, k=5)
        rec_clean = {sid.replace("ho__", ""): v for sid, v in rec.items()}
        m, _ = evaluate_pndcg5(rec_clean, gt_ho, k=5)
        return m

    m23_ho = _score_ho(res_v23["ho_scores"], res_v23["ho_sorted"])
    m24_ho = _score_ho(res_v24["ho_scores"], res_v24["ho_sorted"])

    # z-ансамбль
    z_ho_23 = _z_normalize_by_student(res_v23["ho_sorted"], res_v23["ho_scores"])
    z_ho_24 = _z_normalize_by_student(res_v24["ho_sorted"], res_v24["ho_scores"])
    # ho_sorted от v23 и v24 — одинаковые (одинаковые holdout_records)
    assert (res_v23["ho_sorted"]["student_id"].values ==
            res_v24["ho_sorted"]["student_id"].values).all()
    ens_ho = 0.5 * z_ho_23 + 0.5 * z_ho_24
    m_ens_ho = _score_ho(ens_ho, res_v23["ho_sorted"])

    print(f"\n      Holdout:")
    print(f"          v23 (fix+valp1):        {m23_ho:.4f}")
    print(f"          v24 (fix only):         {m24_ho:.4f}")
    print(f"          ансамбль (0.5/0.5 z):   {m_ens_ho:.4f}")
    print(f"          v7 (ref):               {V7_HOLDOUT:.4f}")

    # Val
    cold_ids_val = identify_cold_students(val)
    val_sorted = res_v23["val_sorted"]
    assert (res_v23["val_sorted"]["student_id"].values ==
            res_v24["val_sorted"]["student_id"].values).all()

    rec_v23 = top_k_per_student(val_sorted, res_v23["val_scores"], k=5)
    rec_v24 = top_k_per_student(val_sorted, res_v24["val_scores"], k=5)
    rec_v23_clean = {sid.replace("vl__", ""): v for sid, v in rec_v23.items()}
    rec_v24_clean = {sid.replace("vl__", ""): v for sid, v in rec_v24.items()}

    zv_23 = _z_normalize_by_student(val_sorted, res_v23["val_scores"])
    zv_24 = _z_normalize_by_student(val_sorted, res_v24["val_scores"])
    ens_val = 0.5 * zv_23 + 0.5 * zv_24
    rec_ens = top_k_per_student(val_sorted, ens_val, k=5)
    rec_ens_clean = {sid.replace("vl__", ""): v for sid, v in rec_ens.items()}

    cold_v23 = _mean_for_ids(rec_v23_clean, val_gt_raw, cold_ids_val)
    cold_v24 = _mean_for_ids(rec_v24_clean, val_gt_raw, cold_ids_val)
    cold_ens = _mean_for_ids(rec_ens_clean, val_gt_raw, cold_ids_val)
    warm_v23 = _mean_for_ids(rec_v23_clean, val_gt_raw, holdout_ids)
    warm_v24 = _mean_for_ids(rec_v24_clean, val_gt_raw, holdout_ids)
    warm_ens = _mean_for_ids(rec_ens_clean, val_gt_raw, holdout_ids)

    print(f"\n      Val:")
    print(f"          v23: cold={cold_v23:.4f}  warm={warm_v23:.4f}  "
          f"test_est={TEST_COLD_SHARE*cold_v23+(1-TEST_COLD_SHARE)*warm_v23:.4f}")
    print(f"          v24: cold={cold_v24:.4f}  warm={warm_v24:.4f}  "
          f"test_est={TEST_COLD_SHARE*cold_v24+(1-TEST_COLD_SHARE)*warm_v24:.4f}")
    print(f"          ens: cold={cold_ens:.4f}  warm={warm_ens:.4f}  "
          f"test_est={TEST_COLD_SHARE*cold_ens+(1-TEST_COLD_SHARE)*warm_ens:.4f}")

    # ==== 6. Сохранение submission ====
    print("\n[6/6] Сохранение submission...")
    val_path = SUBMISSIONS_DIR / "val_submission_ltr_v25.csv"
    write_submission(rec_ens_clean, val_path)
    print(f"      val: {val_path}")

    # test
    test_sorted = res_v23["test_sorted"]
    assert (res_v23["test_sorted"]["student_id"].values ==
            res_v24["test_sorted"]["student_id"].values).all()
    zt_23 = _z_normalize_by_student(test_sorted, res_v23["test_scores"])
    zt_24 = _z_normalize_by_student(test_sorted, res_v24["test_scores"])
    ens_test = 0.5 * zt_23 + 0.5 * zt_24
    rec_test = top_k_per_student(test_sorted, ens_test, k=5)
    rec_test_clean = {sid.replace("ts__", ""): v for sid, v in rec_test.items()}
    test_path = SUBMISSIONS_DIR / "submission_v25.csv"
    write_submission(rec_test_clean, test_path)
    print(f"      test: {test_path}")

    # Сравнение с v23/v24
    for cmp_name in ["submission_v23.csv", "submission_v24.csv"]:
        cmp_path = SUBMISSIONS_DIR / cmp_name
        if not cmp_path.exists():
            continue
        df_cmp = pd.read_csv(cmp_path)
        rec_cmp = {str(r["id"]): str(r["recommendations"]).split()
                   for _, r in df_cmp.iterrows()}
        n_diff = sum(1 for sid in rec_test_clean
                     if rec_test_clean[sid] != rec_cmp.get(sid, []))
        print(f"      vs {cmp_name}: top-5 отличается у {n_diff}/{len(rec_test_clean)}")

    print(f"\n[OK] Пайплайн завершён за {time.time() - t0:.1f} сек.")

    print("\n" + "=" * 78)
    print("ИТОГ:")
    print("=" * 78)
    print(f"  Holdout:  v23={m23_ho:.4f}  v24={m24_ho:.4f}  ens={m_ens_ho:.4f}")
    print(f"  test_est: v23={TEST_COLD_SHARE*cold_v23+(1-TEST_COLD_SHARE)*warm_v23:.4f}  "
          f"v24={TEST_COLD_SHARE*cold_v24+(1-TEST_COLD_SHARE)*warm_v24:.4f}  "
          f"ens={TEST_COLD_SHARE*cold_ens+(1-TEST_COLD_SHARE)*warm_ens:.4f}")
    print()
    print("  Кандидаты на сабмит (завтра):")
    print(f"    1) v23 = fix + val_p1  (test_est ≈ 0.44)")
    print(f"    2) v24 = fix only       (test_est ≈ 0.43)")
    print(f"    3) v25 = ансамбль v23+v24 (test_est ≈ ?)")
    print()
    print("  Если есть 2 сабмита: v23 + v24 → сравнение поможет изолировать")
    print("  эффект val_p1 на public.")


if __name__ == "__main__":
    main()