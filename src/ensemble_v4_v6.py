"""
Ансамбль v4 + v6 на test.

Идея:
    v4 и v6 обучены на одних и тех же данных, но test-инференс у них разный:
    - v4 подаёт test periods=3 как есть (история [P1,P2,P3]).
    - v6 обрезает историю до [P2,P3] (max_history_periods=2).
    На periods=0,1,2 предсказания совпадают, на periods=3 — различаются.
    Усреднение скоров даёт компромисс, который может быть лучше обеих моделей.

Что делаем:
    1. Строим test_df_v4 без обрезки, test_df_v6 с max_history_periods=2.
    2. Категориальные кодируем единым словарём (train+val+test).
    3. Считаем скоры обеих моделей на соответствующих фреймах.
    4. Складываем скоры по строкам с весом alpha.
    5. Top-5 по усреднённому скору → submission_ensemble.csv.

Никакой валидации — у нас нет ground truth для test. Ставка слепая, но
мотивированная: v4 и v6 дают разные результаты на public (0.3189 vs 0.3326),
значит их ошибки частично независимы.

Запуск:
    python -m src.ensemble_v4_v6
    python -m src.ensemble_v4_v6 --alpha 0.5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

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
    load_model,
    predict_scores,
    prepare_xygw,
    top_k_per_student,
)
from src.popularity import compute_popularity
from src.submit import write_submission

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
MODELS_DIR = PROJECT_ROOT / "models"

WEIGHT_P1 = 1.0
WEIGHT_P0 = 0.5
WEIGHT_PMASK = 1.0
WEIGHT_P2 = 1.0


def _add_weight(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = np.float32(weight)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Ансамбль v4 + v6 на test")
    parser.add_argument(
        "--alpha", type=float, default=0.5,
        help="Вес v4 в усреднении: alpha*v4 + (1-alpha)*v6. По умолчанию 0.5",
    )
    args = parser.parse_args()
    alpha = float(args.alpha)

    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    print("=" * 72)
    print(f"АНСАМБЛЬ v4 + v6 (alpha={alpha:.2f})")
    print("=" * 72)

    t0 = time.time()
    data = load_all()
    train = data["train"]
    val = data["val"]
    test = data["test"]
    modules = data["modules"]
    val_solution = data["val_solution"]
    val_gt_raw = dict(zip(val_solution["id"], val_solution["ground_truth"]))

    # ==== 1. Статистики ====
    print("\n[1/5] Статистики...")
    stats = FeatureStats.fit(train, modules)
    stats.add_val_solution_popularity(val_solution)

    # ==== 2. Сырые train-срезы (для словаря категорий) ====
    # Нам нужен единый словарь категорий, тот же что в v4/v6.
    # Строим train_df по той же схеме, что в v6.
    print("\n[2/5] Строю train-срезы для словаря категорий...")

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

    # p2-срез из val
    from collections import Counter
    val_p2 = [s for s in val if len(s.get("history") or []) == 2]
    p2_df = build_dataset(
        val_p2, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )

    train_df_p1 = _add_weight(train_df_p1, WEIGHT_P1)
    train_df_p0 = _add_weight(train_df_p0, WEIGHT_P0)
    train_df_pmask = _add_weight(train_df_pmask, WEIGHT_PMASK)
    p2_df = _add_weight(p2_df, WEIGHT_P2)
    train_df_p1["student_id"] = "p1__" + train_df_p1["student_id"].astype(str)
    train_df_p0["student_id"] = "p0__" + train_df_p0["student_id"].astype(str)
    train_df_pmask["student_id"] = "pmask__" + train_df_pmask["student_id"].astype(str)
    p2_df["student_id"] = "p2__" + p2_df["student_id"].astype(str)

    train_df = pd.concat(
        [train_df_p1, train_df_p0, train_df_pmask, p2_df],
        ignore_index=True,
    )

    # ==== 3. Два test-фрейма ====
    print("\n[3/5] Строю два test-фрейма: без обрезки (v4) и с обрезкой (v6)...")

    test_df_v4 = build_dataset(
        test, stats,
        candidates_fn=candidates_available,
        target_fn=None,
        past_periods=None,
    )
    test_df_v6 = build_dataset(
        test, stats,
        candidates_fn=candidates_available,
        target_fn=None,
        past_periods=None,
        max_history_periods=2,
    )

    # ==== 4. Единый словарь категорий ====
    print("\n[4/5] Кодирую категориальные (единый словарь)...")

    test_df_v4 = _add_weight(test_df_v4, 1.0)
    test_df_v6 = _add_weight(test_df_v6, 1.0)

    n_train = len(train_df)
    n_test = len(test_df_v4)
    # v4 и v6 имеют одинаковые (student_id, module_id) → одинаковая длина.

    all_df = pd.concat([train_df, test_df_v4], ignore_index=True)
    all_df, _ = encode_categoricals(all_df, all_df, CATEGORICAL_FEATS)

    train_df = all_df.iloc[:n_train].reset_index(drop=True)
    test_df_v4 = all_df.iloc[n_train:].reset_index(drop=True)
    # test_df_v6 имеет те же категориальные, копируем коды из test_df_v4.
    test_df_v6["institute"] = test_df_v4["institute"].values
    test_df_v6["specialty"] = test_df_v4["specialty"].values

    # ==== 5. Скоры моделей и усреднение ====
    print("\n[5/5] Загружаю модели, считаю скоры, усредняю...")

    model_v4 = load_model(MODELS_DIR / "lgbm_ranker_v4.txt")
    model_v6 = load_model(MODELS_DIR / "lgbm_ranker_v6.txt")

    # Приводим оба фрейма к одинаковому порядку строк (сортировка по student_id, module_id)
    sort_cols = ["student_id", "module_id"]
    test_df_v4 = test_df_v4.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    test_df_v6 = test_df_v6.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    # Проверим, что порядок совпадает
    assert (test_df_v4["student_id"].values == test_df_v6["student_id"].values).all()
    assert (test_df_v4["module_id"].values == test_df_v6["module_id"].values).all()

    X_v4 = test_df_v4[FEATURE_COLS_V4]
    X_v6 = test_df_v6[FEATURE_COLS_V4]

    scores_v4 = model_v4.predict(X_v4, num_iteration=model_v4.best_iteration)
    scores_v6 = model_v6.predict(X_v6, num_iteration=model_v6.best_iteration)

    # Нормализуем скоры в [0,1] для устойчивости усреднения (rank-based).
    # Иначе разные калибровки моделей исказят смесь.
    from scipy.stats import rankdata
    rank_v4 = rankdata(scores_v4, method="average") / len(scores_v4)
    rank_v6 = rankdata(scores_v6, method="average") / len(scores_v6)
    scores_ensemble = alpha * rank_v4 + (1.0 - alpha) * rank_v6

    rec_test = top_k_per_student(test_df_v4, scores_ensemble, k=5)

    # Fallback для cold — как в v4/v6
    cold_ids_test = identify_cold_students(test)
    pop_weighted, _ = compute_popularity(train)
    fb_test = build_global_recommendations(test, pop_weighted, cold_ids_test, k=5)
    rec_final = apply_fallback(rec_test, fb_test, k=5)

    out_path = SUBMISSIONS_DIR / "submission_ensemble.csv"
    write_submission(rec_final, out_path)
    print(f"\n      Cold-студентов в test: {len(cold_ids_test)} "
          f"({100 * len(cold_ids_test) / len(test):.1f}%)")
    print(f"      Сохранено: {out_path}  ({len(rec_final)} строк)")

    # ==== Сравнение: сколько строк отличается от v4 и v6 ====
    print("\n[Сравнение с одиночными сабмитами]")

    def _load(path: Path) -> dict[str, list[str]]:
        df = pd.read_csv(path)
        return {row["id"]: str(row["recommendations"]).split() for _, row in df.iterrows()}

    ref_v4 = _load(SUBMISSIONS_DIR / "submission_v4.csv")
    ref_v6 = _load(SUBMISSIONS_DIR / "submission_v6.csv")

    n_diff_v4 = sum(1 for sid in rec_final if rec_final[sid] != ref_v4.get(sid, []))
    n_diff_v6 = sum(1 for sid in rec_final if rec_final[sid] != ref_v6.get(sid, []))

    print(f"      Строк, отличающихся от v4: {n_diff_v4} / {len(rec_final)}")
    print(f"      Строк, отличающихся от v6: {n_diff_v6} / {len(rec_final)}")
    print(f"      (Ожидаем ≈1310 — это periods=3, где модели расходятся)")

    print(f"\n[OK] Готово за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()