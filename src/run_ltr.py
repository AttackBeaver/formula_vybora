"""
End-to-end пайплайн LightGBM Ranker + cold-start маскирование + fallback.

Что нового по сравнению с предыдущей версией:
    * Три train-среза вместо двух:
        p1    (past=[1] → p2)         — warm студент
        p0    (past=[] → p1)          — есть профиль, нет истории
        pmask (past=[1] → p2, mask)   — полностью cold (profile={}, history=[])
    * Sample weights: p1=1.0, p0=0.5, pmask=1.0. Передаются в LightGBM
      через prepare_xygw.
    * Три стратегии на выходе для cold-студентов:
        LTR (raw)                 — что модель сама предсказывает для cold
        LTR + fallback global     — top-5 по глобальной популярности
        LTR + fallback institute  — top-5 по институтской популярности
      Победитель по cold-сегменту val идёт в финальный submission.

Запуск из корня проекта:
    python -m src.run_ltr
"""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.cold_start import (
    apply_fallback,
    build_global_recommendations,
    build_institute_index,
    build_institute_recommendations,
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
    encode_categoricals,
    feature_importance_report,
    predict_scores,
    prepare_xygw,
    save_model,
    top_k_per_student,
    train_lgbm_ranker,
)
from src.metric import evaluate_pndcg5
from src.popularity import (
    compute_popularity,
    compute_popularity_by_institute,
)
from src.submit import write_submission

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
MODELS_DIR = PROJECT_ROOT / "models"

# Доля cold-студентов в test — для перевзвешенной оценки.
TEST_COLD_SHARE = 0.352

# Веса срезов в обучающей выборке.
WEIGHT_P1 = 1.0       # warm
WEIGHT_P0 = 0.5       # профиль без истории (полу-cold)
WEIGHT_PMASK = 1.0    # полностью cold (маскированный)


# ---------------------------------------------------------------------------
# Вспомогательное
# ---------------------------------------------------------------------------
def _segment_means(
    rec: dict[str, list[str]],
    gt_raw: dict[str, str],
    cold_ids: set[str],
    k: int = 5,
) -> tuple[float, float, float]:
    """Возвращает (mean_all, cold_mean, warm_mean)."""
    mean_all, per_student = evaluate_pndcg5(rec, gt_raw, k=k)
    cold_scores = [s for sid, s in per_student.items() if sid in cold_ids]
    warm_scores = [s for sid, s in per_student.items() if sid not in cold_ids]
    c = float(np.mean(cold_scores)) if cold_scores else 0.0
    w = float(np.mean(warm_scores)) if warm_scores else 0.0
    return mean_all, c, w


def _add_weight(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    """Добавляет колонку weight со значением weight."""
    df = df.copy()
    df["weight"] = np.float32(weight)
    return df


# ---------------------------------------------------------------------------
# Основной сценарий
# ---------------------------------------------------------------------------
def main() -> None:
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 72)
    print("LIGHTGBM RANKER + COLD-START МАСКИРОВАНИЕ")
    print("=" * 72)

    t0 = time.time()
    data = load_all()
    train = data["train"]
    val = data["val"]
    test = data["test"]
    modules = data["modules"]
    val_gt_raw = dict(
        zip(data["val_solution"]["id"], data["val_solution"]["ground_truth"])
    )

    # ==== 1. Статистики ====
    print("\n[1/9] Обучаю статистики на train...")
    stats = FeatureStats.fit(train, modules)

    # ==== 2. Train-датасеты: три среза ====
    print("\n[2/9] Строю train-датасеты (3 среза)...")

    train_students_p1 = [
        s for s in train
        if any(h.get("period") == 2 for h in s.get("history", []))
    ]
    train_df_p1 = build_dataset(
        train_students_p1, stats,
        candidates_fn=make_candidates_train_period2(stats),
        target_fn=target_period_2,
        past_periods=[1],
        mask_profile=False,
    )
    print(f"      [p1: past=1→p2, warm]       студентов: "
          f"{train_df_p1['student_id'].nunique():,}, "
          f"строк: {len(train_df_p1):,}")

    train_students_p0 = [
        s for s in train
        if any(h.get("period") == 1 for h in s.get("history", []))
    ]
    train_df_p0 = build_dataset(
        train_students_p0, stats,
        candidates_fn=make_candidates_train_period1(stats),
        target_fn=target_period_1,
        past_periods=[],
        mask_profile=False,
    )
    print(f"      [p0: past=[]→p1, no-hist]   студентов: "
          f"{train_df_p0['student_id'].nunique():,}, "
          f"строк: {len(train_df_p0):,}")

    # Маскированный срез: те же студенты, что p1, но с стёртым профилем и историей.
    train_df_pmask = build_dataset(
        train_students_p1, stats,
        candidates_fn=make_candidates_train_period2(stats),
        target_fn=target_period_2,
        past_periods=[1],
        mask_profile=True,
    )
    print(f"      [pmask: past=1→p2, mask]    студентов: "
          f"{train_df_pmask['student_id'].nunique():,}, "
          f"строк: {len(train_df_pmask):,}")

    # Префиксы к student_id — чтобы LightGBM не смешал группы из разных срезов.
    train_df_p1 = _add_weight(train_df_p1, WEIGHT_P1)
    train_df_p0 = _add_weight(train_df_p0, WEIGHT_P0)
    train_df_pmask = _add_weight(train_df_pmask, WEIGHT_PMASK)

    train_df_p1["student_id"] = "p1__" + train_df_p1["student_id"].astype(str)
    train_df_p0["student_id"] = "p0__" + train_df_p0["student_id"].astype(str)
    train_df_pmask["student_id"] = "pmask__" + train_df_pmask["student_id"].astype(str)

    train_df = pd.concat(
        [train_df_p1, train_df_p0, train_df_pmask],
        ignore_index=True,
    )
    print(f"      [ИТОГО]                     студентов: "
          f"{train_df['student_id'].nunique():,}, "
          f"строк: {len(train_df):,}")

    # ==== 3. Val-датасет ====
    print("\n[3/9] Строю val-датасет (past=[1,2] → available)...")
    val_df = build_dataset(
        val, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )
    val_df = _add_weight(val_df, 1.0)
    print(f"      Студентов: {val_df['student_id'].nunique():,}, "
          f"строк: {len(val_df):,}")

    # ==== 3a. Test-датасет ====
    print("\n[3a/9] Строю test-датасет (past=[1,2,3] → available)...")
    test_df = build_dataset(
        test, stats,
        candidates_fn=candidates_available,
        target_fn=None,
        past_periods=None,
    )
    test_df = _add_weight(test_df, 1.0)
    print(f"      Студентов: {test_df['student_id'].nunique():,}, "
          f"строк: {len(test_df):,}")

    # ==== 4. Кодирование категориальных ====
    print("\n[4/9] Кодирую категориальные признаки...")
    train_df, val_df = encode_categoricals(train_df, val_df, CATEGORICAL_FEATS)
    _, test_df = encode_categoricals(val_df.copy(), test_df, CATEGORICAL_FEATS)
    print(f"      Кодированы: {CATEGORICAL_FEATS}")

    # ==== 5. Обучение ====
    print("\n[5/9] Обучаю LightGBM Ranker (с sample weights)...")
    X_train, y_train, g_train, w_train, _ = prepare_xygw(train_df)
    X_val, y_val, g_val, w_val, val_sorted = prepare_xygw(val_df)
    print(f"      Train: X={X_train.shape}, групп={len(g_train)}, "
          f"w_mean={w_train.mean():.3f}")
    print(f"      Val:   X={X_val.shape}, групп={len(g_val)}")

    t_train = time.time()
    model = train_lgbm_ranker(
        X_train, y_train, g_train,
        X_val, y_val, g_val,
        num_boost_round=5000,
        early_stopping_rounds=200,
        log_every=50,
        sample_weight_train=w_train,
        sample_weight_val=w_val,
    )
    print(f"      Обучение завершено за {time.time() - t_train:.1f} сек")
    print(f"      best_iteration = {model.best_iteration}")

    save_model(model, MODELS_DIR / "lgbm_ranker.txt")

    # ==== 6. Инференс val ====
    print("\n[6/9] Предсказание на val...")
    scores_val = predict_scores(model, val_sorted)
    rec_val_ltr = top_k_per_student(val_sorted, scores_val, k=5)

    cold_ids_val = identify_cold_students(val)
    print(f"      Cold-студентов в val: {len(cold_ids_val)} "
          f"({100 * len(cold_ids_val) / len(val):.1f}%)")

    # ==== 7. Fallback на val ====
    print("\n[7/9] Fallback для cold-сегмента val...")
    pop_weighted, _ = compute_popularity(train)
    pop_by_inst = compute_popularity_by_institute(train)

    # Индекс сигнатур available_modules строим из warm-студентов val+test.
    def _is_cold_rec(r: dict) -> bool:
        from src.cold_start import is_cold as _ic
        return _ic(r)
    val_warm = [s for s in val if not _is_cold_rec(s)]
    test_warm = [s for s in test if not _is_cold_rec(s)]
    index = build_institute_index(val_warm + test_warm)
    print(f"      Индекс сигнатур: {len(index)} уникальных наборов")

    rec_fb_global = build_global_recommendations(
        val, pop_weighted, cold_ids_val, k=5,
    )
    rec_fb_inst, _ = build_institute_recommendations(
        val, pop_by_inst, pop_weighted, index, cold_ids_val, k=5,
    )
    rec_val_global = apply_fallback(rec_val_ltr, rec_fb_global, k=5)
    rec_val_inst = apply_fallback(rec_val_ltr, rec_fb_inst, k=5)

    # ==== 8. Метрики на val ====
    print("\n[8/9] Оценка pNDCG@5 на val (3 стратегии):")
    print()
    mean_ltr, cold_ltr, warm_ltr = _segment_means(
        rec_val_ltr, val_gt_raw, cold_ids_val,
    )
    print(f"  [LTR (raw)]")
    print(f"      mean = {mean_ltr:.4f}  cold = {cold_ltr:.4f}  "
          f"warm = {warm_ltr:.4f}")
    print()
    mean_g, cold_g, warm_g = _segment_means(
        rec_val_global, val_gt_raw, cold_ids_val,
    )
    print(f"  [LTR + fallback global]")
    print(f"      mean = {mean_g:.4f}  cold = {cold_g:.4f}  "
          f"warm = {warm_g:.4f}")
    print()
    mean_i, cold_i, warm_i = _segment_means(
        rec_val_inst, val_gt_raw, cold_ids_val,
    )
    print(f"  [LTR + fallback institute]")
    print(f"      mean = {mean_i:.4f}  cold = {cold_i:.4f}  "
          f"warm = {warm_i:.4f}")

    # ---- Выбор победителя по cold-сегменту ----
    candidates = [
        ("LTR (raw)", cold_ltr, rec_val_ltr),
        ("LTR + fallback global", cold_g, rec_val_global),
        ("LTR + fallback institute", cold_i, rec_val_inst),
    ]
    candidates.sort(key=lambda t: -t[1])
    winner_name, winner_cold, winner_rec = candidates[0]

    print("\n  " + "-" * 60)
    print(f"  Победитель по cold-сегменту: {winner_name} "
          f"(cold = {winner_cold:.4f})")

    est_ltr = TEST_COLD_SHARE * cold_ltr + (1 - TEST_COLD_SHARE) * warm_ltr
    est_win = TEST_COLD_SHARE * winner_cold + (1 - TEST_COLD_SHARE) * warm_ltr
    print(f"  Перевзвешенная оценка под test (cold_share={TEST_COLD_SHARE:.3f}):")
    print(f"      LTR (raw):  {est_ltr:.4f}")
    print(f"      Победитель: {est_win:.4f}   (Δ = {est_win - est_ltr:+.4f})")

    # Сохраняем val submission победителя
    val_final_path = SUBMISSIONS_DIR / "val_submission_ltr.csv"
    write_submission(winner_rec, val_final_path)
    print(f"\n  Сохранено: {val_final_path.name}")

    # ==== 9. Инференс test + финальный сабмит ====
    print("\n[9/9] Инференс test + финальный submission...")
    _, _, _, _, test_sorted = prepare_xygw(test_df)
    scores_test = predict_scores(model, test_sorted)
    rec_test_ltr = top_k_per_student(test_sorted, scores_test, k=5)

    cold_ids_test = identify_cold_students(test)
    print(f"      Cold-студентов в test: {len(cold_ids_test)} "
          f"({100 * len(cold_ids_test) / len(test):.1f}%)")

    if winner_name == "LTR (raw)":
        rec_test_final = rec_test_ltr
    elif winner_name == "LTR + fallback global":
        fb = build_global_recommendations(test, pop_weighted, cold_ids_test, k=5)
        rec_test_final = apply_fallback(rec_test_ltr, fb, k=5)
    else:  # institute
        fb, _ = build_institute_recommendations(
            test, pop_by_inst, pop_weighted, index, cold_ids_test, k=5,
        )
        rec_test_final = apply_fallback(rec_test_ltr, fb, k=5)

    test_path = SUBMISSIONS_DIR / "submission.csv"
    write_submission(rec_test_final, test_path)
    print(f"      Сохранено: {test_path}  ({len(rec_test_final)} строк)")

    # ==== Feature importance ====
    print("\n[Feature importance — top-10 по gain]")
    imp = feature_importance_report(model, top_n=10)
    for _, row in imp.iterrows():
        print(f"      {row['feature']:<28}  gain = {row['gain']:>12.0f}")

    print(f"\n[OK] Пайплайн завершён за {time.time() - t0:.1f} сек.")
    print(f"     Финальный сабмит: {test_path}")


if __name__ == "__main__":
    main()