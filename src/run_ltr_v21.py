"""
LTR v21: v7 + новый срез val_p1 (используем всех warm-val P1 студентов).

Мотивация:
    В v7 обучающие срезы используют "правильную структуру" (student's
    real available_modules + real target) только через p2-срез
    (val_p2, 1140 студентов). Но в val есть ещё 1531 warm-студент
    с 1-периодной историей (val_p1). Их val_solution — это их реальные
    P2-таргеты. Их available_modules — тот же 159-модульный набор.

    Это ДОПОЛНИТЕЛЬНЫЙ сигнал с правильной структурой, аналогичный
    p2-срезу по формату, но с более короткой историей.

Что делаем:
    Добавляем срез valp1:
        val_p1 students (1531)
        candidates_fn = candidates_available  (их реальный 159-набор)
        target_fn = make_target_from_val_solution  (их P2-выбор)
        past_periods = [1]
        weight = 2.5

    Всё остальное — точно как v7:
        p1 (w=1.0), p0 (w=0.5), pmask (w=1.0), p2 (w=3.0).
        FEATURE_COLS_V4 (31).
        max_history_periods=2 для test.
        truncation_level=15.
        multi-seed=5, seeds 42..46.
        Fallback отключён.

Метрики:
    - Holdout (284 val_p2) — чистая (val_p1 не пересекается с holdout).
    - Val_warm_p2 (1140) — чистая.
    - Val_p1 — "грязная" (мы её тренировали), НЕ сравнивать с v7.

Ожидания:
    holdout ~0.513 (не должно упасть, val_p1 не пересекается).
    Если val_p1 помог → holdout должен ВЫРАСТИ (модель видит больше
    правильных примеров) → шанс на public > 0.335.

Запуск:
    python -m src.run_ltr_v21 --n-seeds 5
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
WEIGHT_VALP1 = 2.5  # новый срез

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


def _n_periods(r: dict[str, Any]) -> int:
    return len(r.get("history") or [])


def _add_weight(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = np.float32(weight)
    return df


def _segment_means(
    rec: dict[str, list[str]],
    gt_raw: dict[str, str],
    ids: set[str],
    k: int = 5,
) -> float:
    """Средний pNDCG только по указанным студентам."""
    mean_all, per_student = evaluate_pndcg5(rec, gt_raw, k=k)
    vals = [s for sid, s in per_student.items() if sid in ids]
    return float(np.mean(vals)) if vals else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="LTR v21: v7 + val_p1 slice")
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()
    n_seeds = max(1, int(args.n_seeds))
    seeds = [BASE_SEED + i for i in range(n_seeds)]

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print(f"LTR v21: v7 + val_p1 slice (w={WEIGHT_VALP1}), n_seeds={n_seeds}")
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

    # ==== 2. Train-срезы (p0/p1/pmask) ====
    print("\n[2/9] Train-срезы (p0/p1/pmask)...")
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

    # ==== 3. Holdout + p2 (из val_p2) ====
    print("\n[3/9] Holdout + p2 (из val_p2)...")
    val_p2 = [s for s in val if _n_periods(s) == 2]
    rng = random.Random(HOLDOUT_SEED)
    shuffled = list(val_p2)
    rng.shuffle(shuffled)
    n_holdout = int(len(shuffled) * HOLDOUT_FRACTION)
    holdout_records = shuffled[:n_holdout]
    p2_records = shuffled[n_holdout:]
    holdout_ids = {s["id"] for s in holdout_records}
    p2_ids = {s["id"] for s in p2_records}
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

    # ==== 4. НОВЫЙ СРЕЗ: val_p1 ====
    print("\n[4/9] НОВЫЙ срез val_p1 (warm-val с 1-периодной историей)...")
    val_p1_records = [s for s in val if _n_periods(s) == 1]
    val_p1_ids = {s["id"] for s in val_p1_records}
    print(f"      val_p1={len(val_p1_records)} студентов, "
          f"их available_modules — тот же набор, что у val_p2")

    valp1_df = build_dataset(
        val_p1_records, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1],
    )
    print(f"      valp1_df={len(valp1_df):,} строк")

    # ==== 5. Val и test ====
    print("\n[5/9] Val и test (для инференса)...")
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

    # ==== 6. Веса, префиксы, категориальные ====
    print("\n[6/9] Веса, префиксы, категориальные...")
    train_df_p1 = _add_weight(train_df_p1, WEIGHT_P1)
    train_df_p0 = _add_weight(train_df_p0, WEIGHT_P0)
    train_df_pmask = _add_weight(train_df_pmask, WEIGHT_PMASK)
    p2_df = _add_weight(p2_df, WEIGHT_P2)
    valp1_df = _add_weight(valp1_df, WEIGHT_VALP1)
    holdout_df = _add_weight(holdout_df, 1.0)
    val_df = _add_weight(val_df, 1.0)
    test_df = _add_weight(test_df, 1.0)

    train_df_p1["student_id"] = "p1__" + train_df_p1["student_id"].astype(str)
    train_df_p0["student_id"] = "p0__" + train_df_p0["student_id"].astype(str)
    train_df_pmask["student_id"] = "pmask__" + train_df_pmask["student_id"].astype(str)
    p2_df["student_id"] = "p2__" + p2_df["student_id"].astype(str)
    valp1_df["student_id"] = "vp1__" + valp1_df["student_id"].astype(str)
    holdout_df["student_id"] = "ho__" + holdout_df["student_id"].astype(str)

    train_df = pd.concat(
        [train_df_p1, train_df_p0, train_df_pmask, p2_df, valp1_df],
        ignore_index=True,
    )
    print(f"      train: 5 срезов, всего {len(train_df):,} строк")

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

    # ==== 7. Обучение ====
    print(f"\n[7/9] Обучение {n_seeds} моделей...")

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

    save_model(last_model, MODELS_DIR / "lgbm_ranker_v21.txt")

    # ==== 8. Оценка ====
    print("\n[8/9] Оценка:")

    # Holdout (чистая метрика)
    rec_ho = top_k_per_student(ho_sorted, scores_ho, k=5)
    rec_ho_clean = {sid.replace("ho__", ""): v for sid, v in rec_ho.items()}
    gt_ho = {sid: gt for sid, gt in val_gt_raw.items() if sid in holdout_ids}
    mean_ho, _ = evaluate_pndcg5(rec_ho_clean, gt_ho, k=5)

    print(f"      Holdout (n=284, чистая):  {mean_ho:.4f}   "
          f"(v7 = {V7_HOLDOUT:.4f}, Δ = {mean_ho - V7_HOLDOUT:+.4f})")

    # Val разбитый
    cold_ids_val = identify_cold_students(val)
    rec_val = top_k_per_student(val_sorted, scores_val, k=5)
    # Чистые подмножества: holdout-ids + val_p2-ids (не тренировались)
    val_clean_ids = holdout_ids | p2_ids
    m_val_p2clean = _segment_means(rec_val, val_gt_raw, val_clean_ids)
    m_val_cold = _segment_means(rec_val, val_gt_raw, cold_ids_val)
    # p1-val теперь в тренировке — не сравнивать!
    m_val_p1 = _segment_means(rec_val, val_gt_raw, val_p1_ids)

    print(f"\n      Val разбивка:")
    print(f"          val_p2∪holdout (чист., n={len(val_clean_ids)}): {m_val_p2clean:.4f}")
    print(f"          cold           (n={len(cold_ids_val)}):       {m_val_cold:.4f}")
    print(f"          val_p1 (трениров., для справки):             {m_val_p1:.4f}")

    # test_estimate на чистых
    warm_clean_ids = val_clean_ids - cold_ids_val
    warm_clean_scores = [s for sid, s in evaluate_pndcg5(rec_val, val_gt_raw)[1].items()
                         if sid in warm_clean_ids]
    warm_clean_mean = float(np.mean(warm_clean_scores)) if warm_clean_scores else 0.0
    est_val = TEST_COLD_SHARE * m_val_cold + (1 - TEST_COLD_SHARE) * warm_clean_mean
    print(f"      test_estimate (чист. warm) = {est_val:.4f}")

    val_path = SUBMISSIONS_DIR / "val_submission_ltr_v21.csv"
    write_submission(rec_val, val_path)
    print(f"      Сохранено: {val_path}")

    # ==== 9. Test submission ====
    print("\n[9/9] Test submission...")
    rec_test = top_k_per_student(test_sorted, scores_test, k=5)
    test_path = SUBMISSIONS_DIR / "submission_v21.csv"
    write_submission(rec_test, test_path)
    cold_test = identify_cold_students(test)
    print(f"      Cold в test: {len(cold_test)}  Сохранено: {test_path}")

    print("\n[Feature importance — top-15]")
    imp = feature_importance_report(last_model, feature_cols=FEATURE_COLS_V4, top_n=15)
    for _, row in imp.iterrows():
        print(f"      {row['feature']:<32}  gain = {row['gain']:>12.0f}")

    print(f"\n[OK] Пайплайн завершён за {time.time() - t0:.1f} сек.")

    print("\n" + "=" * 78)
    print("ИНТЕРПРЕТАЦИЯ:")
    print("=" * 78)
    if mean_ho >= V7_HOLDOUT + 0.002:
        print(f"[+] holdout ВЫРОС ({mean_ho:.4f} vs {V7_HOLDOUT:.4f}).")
        print("    val_p1 как обучающий срез помогает переносу.")
        print("    -> Сабмит v21 обоснован.")
    elif mean_ho >= V7_HOLDOUT - 0.003:
        print(f"[~] holdout сопоставим ({mean_ho:.4f} vs {V7_HOLDOUT:.4f}).")
        print("    val_p1 не вредит, но и не улучшает holdout.")
        print("    -> Есть смысл сабмитить: возможно, улучшение проявится")
        print("       на test-only сегментах, которых holdout не покрывает.")
    else:
        print(f"[-] holdout УПАЛ ({mean_ho:.4f} vs {V7_HOLDOUT:.4f}).")
        print("    val_p1 с w=2.5 разбавляет сигнал p2.")
        print("    -> Откатываемся к v7.")


if __name__ == "__main__":
    main()