"""
LTR v13: отказ от rank-фич + content-pop сигнал для test_only модулей.

Обоснование (по результатам diag_val_vs_test):
    - available_modules в val = 159 модулей, в test = 158, но
      Jaccard(val, test) = 0.47. Состав НАБОРОВ разный: 58 модулей
      только в val, 57 только в test.
    - Внутри-available ранги (pop_rank_in_available, pop_rank_frac,
      pop_rank_global_in_available из v12) чувствительны к составу
      available. Один и тот же модуль в val и test получает разные ранги,
      если наборы разные. v12 из-за этого переобучился.
    - test_only модули (57 шт) в 63% случаев имеют pop_weighted=0 в train.
      Модель для них видит только title_len, desc_len и tfidf_cos — но
      tfidf_cos требуют истории студента, которой у cold-студентов нет.

Что делаем в v13:
    1. FEATURE_COLS_V13 = FEATURE_COLS_V4 минус:
         - pop_rank_in_available
         - pop_rank_frac
       (оба зависят от состава available и от института; у cold-студентов
       институт=UNKNOWN, у test — другая смесь институтов.)
    2. Добавляем content-pop фичу:
         - content_pop_top5_in_available
       для каждого модуля = средняя pop_weighted_global top-5 модулей
       из ЕГО available, ближайших по TF-IDF к нему самому.
       Работает для test_only модулей с нулевой прямой популярностью.
    3. Добавляем n_in_train:
         - сколько раз модуль встречался в train (0 = test_only).
       Явный сигнал «модель не видела этот модуль».
    4. Всё остальное — как v7: truncation_level=15, WEIGHT_P2=3.0,
       multi-seed=5, max_history_periods=2, fallback отключён.

Запуск:
    python -m src.run_ltr_v13
    python -m src.run_ltr_v13 --n-seeds 5
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
from src.content import ContentModel
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

# Признаки, которые УБИРАЕМ относительно v7
REMOVE_FROM_V4 = [
    "pop_rank_in_available",
    "pop_rank_frac",
]

# Признаки, которые ДОБАВЛЯЕМ
ADD_TO_V4 = [
    "content_pop_top5_in_available",
    "n_in_train",
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


# ---------------------------------------------------------------------------
# Content-pop фичи и n_in_train
# ---------------------------------------------------------------------------
def add_content_pop_features(
    df: pd.DataFrame,
    content: ContentModel,
    pop_weighted: dict[str, float],
    n_in_train: dict[str, int],
    top_k_neighbors: int = 5,
) -> pd.DataFrame:
    """
    Добавляет в df три столбца:
        content_pop_top5_in_available — средняя pop_weighted top-k ближайших
            по TF-IDF модулей из available данного студента (исключая сам
            модуль). Для модулей с нулевой прямой pop — даёт ненулевой сигнал.
        n_in_train — сколько раз модуль встречался в train.

    Работает на уровне строк (student × module). Для каждого студента
    доступные модули считаются один раз, TF-IDF матрица уже L2-нормирована
    в ContentModel.

    Реализация:
        1. По groupby('student_id') получаем список уникальных модулей
           (available).
        2. Для каждого студента считаем матрицу косинусов между его
           available-модулями (на маленьком наборе это быстро).
        3. Для каждого модуля берём top-k соседей (исключая self),
           усредняем их pop_weighted.
    """
    df = df.copy()

    # Карта: модуль → его индекс в content.vectors (если есть)
    mod_to_idx = content.module_to_idx
    vectors = content.vectors

    # n_in_train — общий для всех строк с данным module_id
    df["n_in_train"] = df["module_id"].map(n_in_train).fillna(0).astype(np.int32)

    # content_pop_top5_in_available
    content_pop = np.zeros(len(df), dtype=np.float32)
    # Быстрее работать по группам
    for sid, group in df.groupby("student_id", sort=False):
        modules = group["module_id"].tolist()
        # индексы в TF-IDF матрице для модулей этой группы
        idxs = [mod_to_idx.get(m, -1) for m in modules]
        valid_mask = [i >= 0 for i in idxs]
        if not any(valid_mask):
            continue
        valid_idx_list = [i for i, ok in zip(idxs, valid_mask) if ok]
        # sub-матрица [n_avail, V]
        sub = vectors[valid_idx_list]
        # косинусы [n_avail, n_avail]
        cos = sub @ sub.T
        # маскируем self
        np.fill_diagonal(cos, -1.0)
        # top-k соседей
        k_eff = min(top_k_neighbors, cos.shape[1] - 1)
        if k_eff <= 0:
            continue
        # берём top-k
        top_idx = np.argpartition(-cos, kth=k_eff - 1, axis=1)[:, :k_eff]
        # для каждой строки — среднее pop_weighted её top-k соседей
        pops = np.array(
            [pop_weighted.get(m, 0.0) for m in modules],
            dtype=np.float32,
        )
        # sub_local_positions для валидных строк: каждая валидная строка
        # в исходной группе занимает позицию i; row_pos_in_sub — её номер
        # в sub.
        row_pos_in_sub = -np.ones(len(modules), dtype=np.int64)
        for sub_pos, orig_pos in enumerate(valid_idx_list):
            pass
        # Проще: построим обратное отображение orig_pos → sub_pos
        orig_to_sub = {orig_pos: sub_pos for sub_pos, orig_pos in enumerate(
            [i for i, ok in zip(idxs, valid_mask) if ok]
        )}
        # Хмм — выше путаница. Пересобираем аккуратно.
        # Соберём row_pos_in_sub ровно для valid-строк.
        valid_positions = [i for i, ok in enumerate(valid_mask) if ok]
        row_pos_in_sub = np.full(len(modules), -1, dtype=np.int64)
        for sub_pos, orig_pos in enumerate(valid_positions):
            row_pos_in_sub[orig_pos] = sub_pos

        # Считаем средний pop top-k соседей для каждой valid-строки
        for orig_pos, sub_pos in enumerate(valid_positions):
            neighbor_sub_positions = top_idx[sub_pos]
            neighbor_pops = pops[neighbor_sub_positions]
            content_pop[group.index[orig_pos]] = float(neighbor_pops.mean())

    df["content_pop_top5_in_available"] = content_pop
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="LTR v13")
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()
    n_seeds = max(1, int(args.n_seeds))
    seeds = [BASE_SEED + i for i in range(n_seeds)]

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print(f"LTR v13: чистка rank-фич + content-pop, n_seeds={n_seeds}")
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

    # n_in_train: сколько раз модуль встречался в train-историях
    n_in_train: dict[str, int] = {}
    for s in train:
        for h in s.get("history", []):
            for c in h.get("choices", []):
                n_in_train[c["module_id"]] = n_in_train.get(c["module_id"], 0) + 1
    print(f"      Уникальных модулей в train: {len(n_in_train)}")
    print(f"      Модулей с n_in_train=0 (test_only-подобные): "
          f"{len(set(stats.module_periods) - set(n_in_train))}")

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

    # ==== 5. Content-pop фичи ====
    print("\n[5/9] Content-pop фичи + n_in_train...")
    t_content = time.time()
    for name, df in [
        ("p1", train_df_p1), ("p0", train_df_p0),
        ("pmask", train_df_pmask), ("p2", p2_df),
        ("holdout", holdout_df), ("val", val_df), ("test", test_df),
    ]:
        updated = add_content_pop_features(
            df,
            content=stats.content_model,
            pop_weighted=stats.pop_weighted,
            n_in_train=n_in_train,
            top_k_neighbors=5,
        )
        if name == "p1":
            train_df_p1 = updated
        elif name == "p0":
            train_df_p0 = updated
        elif name == "pmask":
            train_df_pmask = updated
        elif name == "p2":
            p2_df = updated
        elif name == "holdout":
            holdout_df = updated
        elif name == "val":
            val_df = updated
        elif name == "test":
            test_df = updated
    print(f"      Время: {time.time() - t_content:.1f} сек")

    # Диагностика: content_pop для модулей с pop=0
    val_zero_pop = val_df[val_df["pop_weighted_global"] == 0]
    if len(val_zero_pop) > 0:
        print(f"      val: модулей с pop=0 = "
              f"{val_zero_pop['module_id'].nunique()} из "
              f"{val_df['module_id'].nunique()}, "
              f"средний content_pop у них = "
              f"{val_zero_pop['content_pop_top5_in_available'].mean():.3f}")
    test_zero_pop = test_df[test_df["pop_weighted_global"] == 0]
    if len(test_zero_pop) > 0:
        print(f"      test: модулей с pop=0 = "
              f"{test_zero_pop['module_id'].nunique()} из "
              f"{test_df['module_id'].nunique()}, "
              f"средний content_pop у них = "
              f"{test_zero_pop['content_pop_top5_in_available'].mean():.3f}")

    # ==== 6. Веса, префиксы, категориальные ====
    print("\n[6/9] Веса, префиксы, категориальные...")
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

    # Финальный список признаков
    FEATURE_COLS_V13: list[str] = [
        c for c in FEATURE_COLS_V4 if c not in REMOVE_FROM_V4
    ] + ADD_TO_V4
    print(f"      Признаков: {len(FEATURE_COLS_V13)} "
          f"(v7 было {len(FEATURE_COLS_V4)})")
    print(f"      Убрано: {REMOVE_FROM_V4}")
    print(f"      Добавлено: {ADD_TO_V4}")

    # ==== 7. Обучение ====
    print(f"\n[7/9] Обучение {n_seeds} моделей...")

    X_train, y_train, g_train, w_train, _ = prepare_xygw(
        train_df, feature_cols=FEATURE_COLS_V13,
    )
    X_ho, y_ho, g_ho, w_ho, ho_sorted = prepare_xygw(
        holdout_df, feature_cols=FEATURE_COLS_V13,
    )
    _, _, _, _, val_sorted = prepare_xygw(
        val_df, feature_cols=FEATURE_COLS_V13,
    )
    _, _, _, _, test_sorted = prepare_xygw(
        test_df, feature_cols=FEATURE_COLS_V13,
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
        scores_ho_list.append(predict_scores(model, ho_sorted, feature_cols=FEATURE_COLS_V13))
        scores_val_list.append(predict_scores(model, val_sorted, feature_cols=FEATURE_COLS_V13))
        scores_test_list.append(predict_scores(model, test_sorted, feature_cols=FEATURE_COLS_V13))
        last_model = model

    scores_ho = np.mean(scores_ho_list, axis=0)
    scores_val = np.mean(scores_val_list, axis=0)
    scores_test = np.mean(scores_test_list, axis=0)
    print(f"\n      best_iter: mean={np.mean(best_iters):.0f}, "
          f"min={min(best_iters)}, max={max(best_iters)}")

    save_model(last_model, MODELS_DIR / "lgbm_ranker_v13.txt")

    # ==== 8. Оценка ====
    print("\n[8/9] Оценка:")

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

    val_path = SUBMISSIONS_DIR / "val_submission_ltr_v13.csv"
    write_submission(rec_val, val_path)
    print(f"      Сохранено: {val_path}")

    rec_test = top_k_per_student(test_sorted, scores_test, k=5)
    test_path = SUBMISSIONS_DIR / "submission_v13.csv"
    write_submission(rec_test, test_path)
    print(f"      Сохранено: {test_path}")

    print("\n[Feature importance — top-15]")
    imp = feature_importance_report(last_model, feature_cols=FEATURE_COLS_V13, top_n=15)
    for _, row in imp.iterrows():
        print(f"      {row['feature']:<36}  gain = {row['gain']:>12.0f}")

    print(f"\n[OK] Пайплайн завершён за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()