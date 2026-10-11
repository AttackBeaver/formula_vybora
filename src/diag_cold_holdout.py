"""
Диагностика: чистый cold-holdout из train.

Цель — получить честную локальную оценку для cold-сегмента без
использования val (там 88 cold-студентов — слишком мало для
статистической значимости).

Логика:
    1. Берём train-студентов с историей P1 (train_students_p1).
    2. Случайно выбираем 20% — это cold_holdout.
    3. Исключаем ЭТИХ студентов из p1, pmask, p0 срезов.
    4. Обучаем v7-подобную модель на урезанном train.
    5. Оцениваем на cold_holdout: маскируем профиль и историю,
       кандидаты = модули, доступные в P2; target = P2.
    6. Параллельно оцениваем warm-holdout из val (sanity-check).

ВАЖНО: train-студенты НЕ имеют поля available_modules (оно только
у val/test). Поэтому для симуляции cold используем полный пул модулей
периода 2 — тот же подход, что в pmask-срезе v7.

Запуск:
    python -m src.diag_cold_holdout
"""
from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
from src.popularity import compute_popularity

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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


def _student_id(s: dict[str, Any]) -> str | None:
    """Безопасно достаёт id: train — student_id, val/test — id."""
    return s.get("id") or s.get("student_id")


def _n_periods(r: dict[str, Any]) -> int:
    return len(r.get("history") or [])


def _add_weight(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = np.float32(weight)
    return df


def _topk_global_pop(
    student_ids: list[str],
    candidate_modules: list[str],
    pop_weighted: dict[str, float],
    k: int = 5,
) -> dict[str, list[str]]:
    """
    Top-k по глобальной популярности среди фиксированного пула модулей.
    Для cold_holdout пул — все модули периода 2.
    """
    # Один и тот же top-k для всех студентов (все имеют одинаковый пул)
    scored = [(m, pop_weighted.get(m, 0.0)) for m in candidate_modules]
    scored.sort(key=lambda x: (-x[1], x[0]))
    top_k = [m for m, _ in scored[:k]]
    return {sid: list(top_k) for sid in student_ids}


def main() -> None:
    t0 = time.time()

    print("=" * 78)
    print("ДИАГНОСТИКА: чистый cold-holdout из train")
    print("=" * 78)

    data = load_all()
    train = data["train"]
    val = data["val"]
    modules = data["modules"]
    val_solution = data["val_solution"]
    val_gt_raw = dict(zip(val_solution["id"], val_solution["ground_truth"]))

    # ==== 1. Выбираем cold_holdout ====
    print("\n[1/6] Формируем cold_holdout из train_students_p1...")

    train_students_p1_all = [
        s for s in train
        if any(h.get("period") == 2 for h in s.get("history", []))
    ]
    print(f"      train_students_p1 всего: {len(train_students_p1_all)}")

    rng = random.Random(HOLDOUT_SEED)
    shuffled = list(train_students_p1_all)
    rng.shuffle(shuffled)
    n_cold_holdout = int(len(shuffled) * HOLDOUT_FRACTION)
    cold_holdout_students = shuffled[:n_cold_holdout]
    cold_holdout_ids = {
        sid for sid in (_student_id(s) for s in cold_holdout_students)
        if sid is not None
    }
    train_students_p1 = [s for s in shuffled[n_cold_holdout:]]

    print(f"      cold_holdout: {len(cold_holdout_students)} студентов "
          f"({len(cold_holdout_ids)} уникальных id)")
    print(f"      train (осталось): {len(train_students_p1)} студентов")

    # ==== 2. FeatureStats на урезанном train ====
    print("\n[2/6] FeatureStats на урезанном train...")
    stats = FeatureStats.fit(train_students_p1, modules)
    stats.add_val_solution_popularity(val_solution)

    # Пул кандидатов для cold_holdout и p1-среза — модули периода 2
    candidates_p2_fn = make_candidates_train_period2(stats)
    candidate_p2_modules = candidates_p2_fn(train_students_p1[0])
    print(f"      Пул кандидатов P2: {len(candidate_p2_modules)} модулей")

    # ==== 3. Train-срезы (без cold_holdout) ====
    print("\n[3/6] Train-срезы...")
    train_df_p1 = build_dataset(
        train_students_p1, stats,
        candidates_fn=candidates_p2_fn,
        target_fn=target_period_2,
        past_periods=[1],
    )

    train_students_p0_all = [
        s for s in train
        if any(h.get("period") == 1 for h in s.get("history", []))
    ]
    train_students_p0 = [
        s for s in train_students_p0_all
        if _student_id(s) not in cold_holdout_ids
    ]
    train_df_p0 = build_dataset(
        train_students_p0, stats,
        candidates_fn=make_candidates_train_period1(stats),
        target_fn=target_period_1,
        past_periods=[],
    )
    train_df_pmask = build_dataset(
        train_students_p1, stats,
        candidates_fn=candidates_p2_fn,
        target_fn=target_period_2,
        past_periods=[1],
        mask_profile=True,
    )
    print(f"      p1={len(train_df_p1):,}  p0={len(train_df_p0):,}  "
          f"pmask={len(train_df_pmask):,}")

    # ==== 4. Warm-holdout из val ====
    print("\n[4/6] Warm-holdout из val (periods=2)...")
    val_p2 = [s for s in val if _n_periods(s) == 2]
    rng2 = random.Random(HOLDOUT_SEED)
    shuffled_val = list(val_p2)
    rng2.shuffle(shuffled_val)
    n_warm_holdout = int(len(shuffled_val) * HOLDOUT_FRACTION)
    warm_holdout_records = shuffled_val[:n_warm_holdout]
    p2_records = shuffled_val[n_warm_holdout:]
    warm_holdout_ids = {s["id"] for s in warm_holdout_records}
    print(f"      warm_holdout={len(warm_holdout_records)}  p2={len(p2_records)}")

    warm_holdout_df = build_dataset(
        warm_holdout_records, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )
    p2_df = build_dataset(
        p2_records, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )

    # ==== 5. Cold-holdout датасет ====
    # ВАЖНО: candidates_fn — фиксированный пул модулей P2,
    # т.к. у train-студентов нет поля available_modules.
    print("\n[5/6] Cold-holdout датасет (fixed P2-pool)...")
    cold_holdout_df = build_dataset(
        cold_holdout_students, stats,
        candidates_fn=candidates_p2_fn,
        target_fn=target_period_2,
        past_periods=[1],
        mask_profile=True,
    )
    print(f"      cold_holdout_df: {len(cold_holdout_df):,} строк")
    if len(cold_holdout_df) == 0:
        print("[!] ОШИБКА: cold_holdout_df пуст. Прерываю.")
        return
    n_unique = cold_holdout_df["student_id"].nunique()
    print(f"      Уникальных студентов: {n_unique}")

    # Ground truth для cold_holdout
    cold_holdout_gt: dict[str, str] = {}
    for s in cold_holdout_students:
        sid = _student_id(s)
        if sid is None:
            continue
        tgt = target_period_2(s)
        cold_holdout_gt[sid] = " ".join(f"{m}:{p}" for m, p in tgt.items())

    # ==== 6. Обучение и оценка ====
    print("\n[6/6] Обучение v7-подобной модели (1 seed)...")

    train_df_p1 = _add_weight(train_df_p1, WEIGHT_P1)
    train_df_p0 = _add_weight(train_df_p0, WEIGHT_P0)
    train_df_pmask = _add_weight(train_df_pmask, WEIGHT_PMASK)
    p2_df = _add_weight(p2_df, WEIGHT_P2)
    warm_holdout_df = _add_weight(warm_holdout_df, 1.0)
    cold_holdout_df = _add_weight(cold_holdout_df, 1.0)

    train_df_p1["student_id"] = "p1__" + train_df_p1["student_id"].astype(str)
    train_df_p0["student_id"] = "p0__" + train_df_p0["student_id"].astype(str)
    train_df_pmask["student_id"] = "pmask__" + train_df_pmask["student_id"].astype(str)
    p2_df["student_id"] = "p2__" + p2_df["student_id"].astype(str)
    warm_holdout_df["student_id"] = "who__" + warm_holdout_df["student_id"].astype(str)
    cold_holdout_df["student_id"] = "cho__" + cold_holdout_df["student_id"].astype(str)

    train_df = pd.concat(
        [train_df_p1, train_df_p0, train_df_pmask, p2_df],
        ignore_index=True,
    )

    n_train = len(train_df)
    n_who = len(warm_holdout_df)
    n_cho = len(cold_holdout_df)

    all_df = pd.concat(
        [train_df, warm_holdout_df, cold_holdout_df], ignore_index=True,
    )
    all_df, _ = encode_categoricals(all_df, all_df, CATEGORICAL_FEATS)

    train_df = all_df.iloc[:n_train].reset_index(drop=True)
    warm_holdout_df = all_df.iloc[n_train:n_train + n_who].reset_index(drop=True)
    cold_holdout_df = all_df.iloc[n_train + n_who:].reset_index(drop=True)

    X_train, y_train, g_train, w_train, _ = prepare_xygw(
        train_df, feature_cols=FEATURE_COLS_V4,
    )
    _, _, _, _, who_sorted = prepare_xygw(
        warm_holdout_df, feature_cols=FEATURE_COLS_V4,
    )
    _, _, _, _, cho_sorted = prepare_xygw(
        cold_holdout_df, feature_cols=FEATURE_COLS_V4,
    )
    X_who, y_who, g_who, w_who, _ = prepare_xygw(
        warm_holdout_df, feature_cols=FEATURE_COLS_V4,
    )

    print(f"      Train: X={X_train.shape}, групп={len(g_train)}")

    model = train_lgbm_ranker(
        X_train, y_train, g_train,
        X_who, y_who, g_who,
        params={**FAST_PARAMS, "seed": BASE_SEED},
        num_boost_round=NUM_BOOST_ROUND,
        early_stopping_rounds=EARLY_STOPPING,
        log_every=500,
        sample_weight_train=w_train,
        sample_weight_val=w_who,
    )
    print(f"      best_iter={model.best_iteration}")

    # --- Warm-holdout ---
    scores_who = predict_scores(model, who_sorted, feature_cols=FEATURE_COLS_V4)
    rec_who = top_k_per_student(who_sorted, scores_who, k=5)
    rec_who_clean = {sid.replace("who__", ""): v for sid, v in rec_who.items()}
    gt_who = {sid: gt for sid, gt in val_gt_raw.items() if sid in warm_holdout_ids}
    mean_who, _ = evaluate_pndcg5(rec_who_clean, gt_who, k=5)

    # --- Cold-holdout (LTR) ---
    scores_cho = predict_scores(model, cho_sorted, feature_cols=FEATURE_COLS_V4)
    rec_cho = top_k_per_student(cho_sorted, scores_cho, k=5)
    rec_cho_clean = {sid.replace("cho__", ""): v for sid, v in rec_cho.items()}
    mean_cho_ltr, _ = evaluate_pndcg5(rec_cho_clean, cold_holdout_gt, k=5)

    # --- Cold-holdout (Global popularity baseline) ---
    pop_weighted, _ = compute_popularity(train_students_p1)
    rec_cho_pop = _topk_global_pop(
        list(cold_holdout_gt.keys()),
        candidate_p2_modules,
        pop_weighted,
        k=5,
    )
    mean_cho_pop, _ = evaluate_pndcg5(rec_cho_pop, cold_holdout_gt, k=5)

    # --- Разбивка по длине истории ---
    n_choices_per_student: dict[str, int] = {}
    for s in cold_holdout_students:
        sid = _student_id(s)
        if sid is None:
            continue
        n_choices_per_student[sid] = sum(
            len(h.get("choices", [])) for h in s.get("history", [])
        )
    n_ch_arr = np.array(list(n_choices_per_student.values()))
    median_n = float(np.median(n_ch_arr))
    _, per_student_ltr = evaluate_pndcg5(rec_cho_clean, cold_holdout_gt, k=5)
    long_ids = {sid for sid, nc in n_choices_per_student.items() if nc > median_n}
    short_ids = {sid for sid, nc in n_choices_per_student.items() if nc <= median_n}
    long_scores = [s for sid, s in per_student_ltr.items() if sid in long_ids]
    short_scores = [s for sid, s in per_student_ltr.items() if sid in short_ids]
    long_mean = float(np.mean(long_scores)) if long_scores else 0.0
    short_mean = float(np.mean(short_scores)) if short_scores else 0.0

    # ==== Отчёт ====
    print("\n" + "=" * 78)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 78)

    print(f"\n[Warm-holdout (val periods=2, n={len(warm_holdout_records)})]")
    print(f"    LTR: {mean_who:.4f}   (ожидание: ≈ v7 0.5134)")

    print(f"\n[Cold-holdout (train p1-students masked, n={len(cold_holdout_students)})]")
    print(f"    LTR (v7-like, 1 seed):   {mean_cho_ltr:.4f}")
    print(f"    Global popularity:       {mean_cho_pop:.4f}")
    print(f"    Δ (LTR vs pop):          {mean_cho_ltr - mean_cho_pop:+.4f}")

    print(f"\n[Разбивка cold_holdout по длине истории студента]")
    print(f"    median n_choices = {median_n:.0f}")
    print(f"    длинная история (>{median_n:.0f}, n={len(long_ids)}):  "
          f"LTR = {long_mean:.4f}")
    print(f"    короткая (≤{median_n:.0f}, n={len(short_ids)}):  "
          f"LTR = {short_mean:.4f}")

    print("\n" + "=" * 78)
    print("ИНТЕРПРЕТАЦИЯ:")
    print("=" * 78)
    if mean_cho_ltr >= 0.25:
        print("[+] v7-подобная модель ХОРОША на cold (≥0.25 на чистом holdout).")
        print("    => public 0.335 — реалистичный потолок v7.")
        print("    => для пробития нужно менять архитектуру, а не признаки.")
    elif mean_cho_ltr >= 0.20:
        print("[~] v7-подобная модель приемлема на cold (0.20-0.25).")
        print("    => val cold 0.2474 правдоподобен, но чуть завышен.")
        print("    => public 0.335 объясним сдвигом и шумом.")
    else:
        print("[-] v7-подобная модель ПЛОХА на cold (<0.20).")
        print("    => val cold 0.2474 был случайностью, настоящая cold-оценка ниже.")
        print("    => нужна принципиально другая cold-стратегия.")

    if mean_cho_ltr < mean_cho_pop:
        print("\n[!] Внимание: LTR ХУЖЕ глобальной популярности на cold-holdout!")
        print("    Это значит, что наши 4-срезные срезы не помогают на cold.")
    else:
        print(f"\n[OK] LTR лучше глобальной популярности на cold "
              f"на {mean_cho_ltr - mean_cho_pop:+.4f}.")

    print(f"\n[OK] Диагностика завершена за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()