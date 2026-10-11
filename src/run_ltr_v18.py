"""
LTR v18: v7 (champion) + post-hoc фильтр already_chosen.

Восстанавливаем p2-срез с WEIGHT_P2=3.0 — v17 доказал, что без него
модель не может обучаться на periods=2 (holdout упал до 0.40).

Единственное изменение относительно v7 — post-hoc фильтр:
    Убираем из top-5 модули, которые студент уже выбирал в своей
    истории. Это семантически корректно (нельзя повторно выбрать
    пройденный модуль) и не зависит от состава available, поэтому
    не пострадает от Jaccard(val, test) = 0.47.

Диагностика:
    Печатаем, сколько хитов already_chosen в top-5 у warm-студентов
    (у cold их нет by definition).
    Печатаем Δ метрики от применения фильтра на holdout/val/test.

Всё остальное — как v7:
    - FEATURE_COLS_V4 (31).
    - Срезы p0/p1/pmask/p2 с весами 0.5/1.0/1.0/3.0.
    - max_history_periods=2 для test.
    - truncation_level=15.
    - multi-seed=5, seeds 42..46.
    - Fallback отключён.

Запуск:
    python -m src.run_ltr_v18 --n-seeds 5
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


def _student_id(s: dict[str, Any]) -> str | None:
    return s.get("id") or s.get("student_id")


def _n_periods(r: dict[str, Any]) -> int:
    return len(r.get("history") or [])


def _add_weight(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = np.float32(weight)
    return df


def _history_set_for_student(student: dict) -> set[str]:
    out: set[str] = set()
    for h in student.get("history") or []:
        for c in h.get("choices") or []:
            out.add(c["module_id"])
    return out


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


def top_k_with_filter(
    df: pd.DataFrame,
    scores: np.ndarray,
    k: int = 5,
    exclude_by_student: dict[str, set[str]] | None = None,
) -> dict[str, list[str]]:
    """
    Аналог top_k_per_student, но исключает модули из exclude_by_student[sid].
    Если после исключения осталось < k, добиваем из исключённых
    (теоретически невозможно при |available| >= k+1).
    """
    assert len(df) == len(scores)
    work = pd.DataFrame({
        "student_id": df["student_id"].values,
        "module_id": df["module_id"].values,
        "score": scores,
    })
    work = work.sort_values(
        ["student_id", "score", "module_id"],
        ascending=[True, False, True],
        kind="stable",
    )

    result: dict[str, list[str]] = {}
    for sid, group in work.groupby("student_id", sort=False):
        blacklist = set()
        if exclude_by_student is not None:
            blacklist = exclude_by_student.get(sid, set())
        chosen: list[str] = []
        seen: set[str] = set()
        for m in group["module_id"]:
            if m in blacklist or m in seen:
                continue
            chosen.append(m)
            seen.add(m)
            if len(chosen) == k:
                break
        # Fallback (не должен срабатывать)
        if len(chosen) < k:
            for m in group["module_id"]:
                if m in seen:
                    continue
                chosen.append(m)
                seen.add(m)
                if len(chosen) == k:
                    break
        result[sid] = chosen
    return result


def count_blacklisted_in_topk(
    rec: dict[str, list[str]],
    exclude_by_student: dict[str, set[str]],
    k: int = 5,
) -> tuple[int, int]:
    hits = 0
    students_with_hits = 0
    for sid, mods in rec.items():
        blacklist = exclude_by_student.get(sid, set())
        this_hits = sum(1 for m in mods[:k] if m in blacklist)
        if this_hits > 0:
            hits += this_hits
            students_with_hits += 1
    return hits, students_with_hits


def main() -> None:
    parser = argparse.ArgumentParser(description="LTR v18: v7 + фильтр already_chosen")
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()
    n_seeds = max(1, int(args.n_seeds))
    seeds = [BASE_SEED + i for i in range(n_seeds)]

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print(f"LTR v18: v7 + фильтр already_chosen, n_seeds={n_seeds}")
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

    # ==== 2. Train-срезы (как v7) ====
    print("\n[2/8] Train-срезы (как v7)...")
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
    all_df, _ = encode_categoricals(all_df, all_df, CATEGORICAL_FEATS)

    train_df = all_df.iloc[:n_train].reset_index(drop=True)
    holdout_df = all_df.iloc[n_train:n_train + n_ho].reset_index(drop=True)
    val_df = all_df.iloc[n_train + n_ho:n_train + n_ho + n_val].reset_index(drop=True)
    test_df = all_df.iloc[n_train + n_ho + n_val:].reset_index(drop=True)

    print(f"      Признаков: {len(FEATURE_COLS_V4)}")

    # ==== 6. Обучение ====
    print(f"\n[6/8] Обучение {n_seeds} моделей...")

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
        scores_ho_list.append(predict_scores(model, ho_sorted, feature_cols=FEATURE_COLS_V4))
        scores_val_list.append(predict_scores(model, val_sorted, feature_cols=FEATURE_COLS_V4))
        scores_test_list.append(predict_scores(model, test_sorted, feature_cols=FEATURE_COLS_V4))
        last_model = model

    scores_ho = np.mean(scores_ho_list, axis=0)
    scores_val = np.mean(scores_val_list, axis=0)
    scores_test = np.mean(scores_test_list, axis=0)
    print(f"\n      best_iter: mean={np.mean(best_iters):.0f}, "
          f"min={min(best_iters)}, max={max(best_iters)}")

    save_model(last_model, MODELS_DIR / "lgbm_ranker_v18.txt")

    # ==== 7. Оценка: без фильтра и с фильтром ====
    print("\n[7/8] Оценка (без фильтра / с фильтром):")

    # Карты истории: {raw_student_id → set(module_ids)}
    val_hist_map: dict[str, set[str]] = {}
    for s in val:
        sid = _student_id(s)
        if sid is not None:
            val_hist_map[sid] = _history_set_for_student(s)
    test_hist_map: dict[str, set[str]] = {}
    for s in test:
        sid = _student_id(s)
        if sid is not None:
            test_hist_map[sid] = _history_set_for_student(s)

    gt_ho = {sid: gt for sid, gt in val_gt_raw.items() if sid in holdout_ids}

    # --- Holdout ---
    rec_ho_nf = top_k_per_student(ho_sorted, scores_ho, k=5)
    rec_ho_nf_clean = {sid.replace("ho__", ""): v for sid, v in rec_ho_nf.items()}
    mean_ho_nf, _ = evaluate_pndcg5(rec_ho_nf_clean, gt_ho, k=5)

    # exclude map для holdout — по ключам с префиксом ho__
    ho_exclude = {
        sid_prefixed: val_hist_map.get(sid_prefixed.replace("ho__", ""), set())
        for sid_prefixed in rec_ho_nf.keys()
    }
    rec_ho_f = top_k_with_filter(ho_sorted, scores_ho, k=5, exclude_by_student=ho_exclude)
    rec_ho_f_clean = {sid.replace("ho__", ""): v for sid, v in rec_ho_f.items()}
    mean_ho_f, _ = evaluate_pndcg5(rec_ho_f_clean, gt_ho, k=5)

    # диагностика: сколько хитов already_chosen было в rec_ho_nf
    ho_exclude_clean = {
        sid.replace("ho__", ""): s for sid, s in ho_exclude.items()
    }
    n_bl_ho, n_st_ho = count_blacklisted_in_topk(rec_ho_nf_clean, ho_exclude_clean)

    print(f"\n      Holdout (n={len(holdout_records)}):")
    print(f"          без фильтра:   {mean_ho_nf:.4f}")
    print(f"          с фильтром:    {mean_ho_f:.4f}   "
          f"(Δ = {mean_ho_f - mean_ho_nf:+.4f})")
    print(f"          already_chosen в top-5 (без фильтра): "
          f"{n_bl_ho} хитов у {n_st_ho} студентов")
    print(f"          v7 (ref):      {V7_HOLDOUT:.4f}")

    # --- Val ---
    cold_ids_val = identify_cold_students(val)
    rec_val_nf = top_k_per_student(val_sorted, scores_val, k=5)
    val_exclude = {
        sid: val_hist_map.get(sid, set()) for sid in rec_val_nf.keys()
    }
    rec_val_f = top_k_with_filter(
        val_sorted, scores_val, k=5, exclude_by_student=val_exclude,
    )
    n_bl_val, n_st_val = count_blacklisted_in_topk(rec_val_nf, val_exclude)

    mean_val_nf, cold_val_nf, warm_val_nf = _segment_means(
        rec_val_nf, val_gt_raw, cold_ids_val,
    )
    mean_val_f, cold_val_f, warm_val_f = _segment_means(
        rec_val_f, val_gt_raw, cold_ids_val,
    )
    est_val_nf = TEST_COLD_SHARE * cold_val_nf + (1 - TEST_COLD_SHARE) * warm_val_nf
    est_val_f = TEST_COLD_SHARE * cold_val_f + (1 - TEST_COLD_SHARE) * warm_val_f

    print(f"\n      Val (n={len(val)}):")
    print(f"          без фильтра: mean={mean_val_nf:.4f}  "
          f"cold={cold_val_nf:.4f}  warm={warm_val_nf:.4f}  "
          f"estimate={est_val_nf:.4f}")
    print(f"          с фильтром:  mean={mean_val_f:.4f}  "
          f"cold={cold_val_f:.4f}  warm={warm_val_f:.4f}  "
          f"estimate={est_val_f:.4f}")
    print(f"          already_chosen в top-5: "
          f"{n_bl_val} хитов у {n_st_val} студентов (из {len(rec_val_nf)})")

    # --- Сохраняем val ---
    val_path = SUBMISSIONS_DIR / "val_submission_ltr_v18.csv"
    write_submission(rec_val_f, val_path)
    print(f"      Сохранено: {val_path}")

    # --- Test ---
    rec_test_nf = top_k_per_student(test_sorted, scores_test, k=5)
    test_exclude = {
        sid: test_hist_map.get(sid, set()) for sid in rec_test_nf.keys()
    }
    rec_test_f = top_k_with_filter(
        test_sorted, scores_test, k=5, exclude_by_student=test_exclude,
    )
    n_bl_test, n_st_test = count_blacklisted_in_topk(rec_test_nf, test_exclude)
    print(f"\n      Test (n={len(test)}):")
    print(f"          already_chosen в top-5 (без фильтра): "
          f"{n_bl_test} хитов у {n_st_test} студентов "
          f"({100 * n_st_test / len(rec_test_nf):.1f}%)")

    test_path = SUBMISSIONS_DIR / "submission_v18.csv"
    write_submission(rec_test_f, test_path)
    print(f"      Сохранено: {test_path}")

    print("\n[Feature importance — top-15]")
    imp = feature_importance_report(last_model, feature_cols=FEATURE_COLS_V4, top_n=15)
    for _, row in imp.iterrows():
        print(f"      {row['feature']:<32}  gain = {row['gain']:>12.0f}")

    print(f"\n[OK] Пайплайн завершён за {time.time() - t0:.1f} сек.")

    print("\n" + "=" * 78)
    print("ИНТЕРПРЕТАЦИЯ:")
    print("=" * 78)
    print(f"  Holdout, Δ от фильтра:  {mean_ho_f - mean_ho_nf:+.4f}")
    print(f"  Val,     Δ от фильтра:  {mean_val_f - mean_val_nf:+.4f}")
    print(f"  (фильтр семантически не зависит от available →")
    print(f"   если есть Δ на val, должен перенестись на public)")


if __name__ == "__main__":
    main()