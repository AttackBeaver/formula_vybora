"""
ensemble_v7_v39.py — ансамбль v7 + v39 (z-score avg).

ПОЧЕМУ:
    v7  = V4,  p2-срез (w=3.0),  ES на val_p2 holdout,  public = 0.3240
    v39 = V11, p0mask, БЕЗ p2,    ES на train,           public = 0.3142

    Разные feature-set (V4 vs V11), разные срезы (с/без p2), разные ES.
    Ошибки должны частично декореллировать.

МЕТОД:
    Для каждого студента:
        1. Берём top-кандидатов из объединения (v7 top-10 ∪ v39 top-10).
        2. Нормализуем скоры (mean=0, std=1) ВНУТРИ студента.
        3. Усредняем: 0.5·z(v7) + 0.5·z(v39).
        4. Сортируем по avg-z, берём top-5.

    Почему z-score, а не raw: у моделей разные калибровки скоров
    (LightGBM lambdarank не калиброван). Z-score убирает разницу.

ВАЖНО: нам нужны СКОРЫ, а не готовые top-5. Перегенерируем их:
    - v7:   запускаем run_ltr_v7 --save-scores (если есть) ИЛИ читаем
            заново из модели models/lgbm_ranker_v7.txt
    - v39:  аналогично из models/lgbm_ranker_v39_v11_p0mask3p0_p00p3.txt

    Если save-скоров нет — быстро переобучаем (или используем готовые
    модели, если они сохранены).

ПРОЩЕ: сделаем ансамбль через ПОВТОРНЫЙ инференс сохранённых моделей.

ЗАПУСК:
    python -m src.ensemble_v7_v39
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.cold_start import (
    apply_fallback,
    build_global_recommendations,
    identify_cold_students,
)
from src.data_loader import load_all
from src.features import FeatureStats, build_dataset, candidates_available
from src.ltr_model import (
    CATEGORICAL_FEATS,
    FEATURE_COLS_V4,
    FEATURE_COLS_V11,
    encode_categoricals,
    load_model,
    predict_scores,
    prepare_xygw,
    top_k_per_student,
)
from src.metric import evaluate_pndcg5
from src.popularity import compute_popularity
from src.submit import write_submission

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
MODELS_DIR = PROJECT_ROOT / "models"

WEIGHT_V7 = 0.5
WEIGHT_V39 = 0.5
TOP_K_FOR_AVG = 10   # объединяем top-10 каждой модели, потом усредняем


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


def _ensemble_per_student(
    df: pd.DataFrame,
    z_v7: np.ndarray,
    z_v39: np.ndarray,
    weight_v7: float,
    weight_v39: float,
    k: int = 5,
) -> dict[str, list[str]]:
    work = pd.DataFrame({
        "student_id": df["student_id"].values,
        "module_id": df["module_id"].values,
        "z_v7": z_v7,
        "z_v39": z_v39,
    })
    work["score"] = weight_v7 * work["z_v7"] + weight_v39 * work["z_v39"]
    work = work.sort_values(
        ["student_id", "score", "module_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    result: dict[str, list[str]] = {}
    for sid, grp in work.groupby("student_id", sort=False):
        result[sid] = grp["module_id"].head(k).tolist()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Ensemble v7 + v39")
    ap.add_argument("--weight-v7", type=float, default=WEIGHT_V7)
    ap.add_argument("--weight-v39", type=float, default=WEIGHT_V39)
    args = ap.parse_args()

    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print(f"Ансамбль v7 + v39 (z-score avg, "
          f"w_v7={args.weight_v7}, w_v39={args.weight_v39})")
    print("=" * 78)

    t0 = time.time()
    data = load_all()
    train = data["train"]
    val = data["val"]
    test = data["test"]
    modules = data["modules"]
    val_solution = data["val_solution"]
    val_gt_raw = dict(zip(val_solution["id"], val_solution["ground_truth"]))

    # ==== 1. Проверка моделей ====
    print("\n[1/6] Проверка сохранённых моделей...")
    v7_path = MODELS_DIR / "lgbm_ranker_v7.txt"
    v39_path = MODELS_DIR / "lgbm_ranker_v39_v11_p0mask3p0_p00p3.txt"

    if not v7_path.exists():
        print(f"  ! Нет {v7_path}")
        print(f"  Сначала: python -m src.run_ltr_v7")
        return
    if not v39_path.exists():
        # пробуем альтернативное имя
        candidates = list(MODELS_DIR.glob("lgbm_ranker_v39*.txt"))
        if not candidates:
            print(f"  ! Нет lgbm_ranker_v39*.txt в {MODELS_DIR}")
            print(f"  Сначала: python -m src.run_ltr_v39")
            return
        v39_path = candidates[0]

    print(f"  v7:  {v7_path.name}")
    print(f"  v39: {v39_path.name}")

    model_v7 = load_model(v7_path)
    model_v39 = load_model(v39_path)

    # ==== 2. FeatureStats ====
    print("\n[2/6] FeatureStats (tfidf)...")
    stats = FeatureStats.fit(train, modules, content_backend="tfidf")

    # ==== 3. Val + test ====
    print("\n[3/6] Val + test датасеты...")
    val_df = build_dataset(
        val, stats,
        candidates_fn=candidates_available,
        target_fn=None,
        past_periods=None,
    )
    test_df = build_dataset(
        test, stats,
        candidates_fn=candidates_available,
        target_fn=None,
        past_periods=None,
        max_history_periods=2,
    )
    val_df["weight"] = 1.0
    test_df["weight"] = 1.0
    print(f"  val={len(val_df):,}  test={len(test_df):,}")

    # ==== 4. Кодирование + матрицы ====
    print("\n[4/6] Матрицы X...")
    n_val = len(val_df)
    all_df = pd.concat([val_df, test_df], ignore_index=True)
    all_df, _ = encode_categoricals(all_df, all_df, CATEGORICAL_FEATS)
    val_df = all_df.iloc[:n_val].reset_index(drop=True)
    test_df = all_df.iloc[n_val:].reset_index(drop=True)

    _, _, _, _, val_sorted = prepare_xygw(
        val_df, feature_cols=FEATURE_COLS_V4, weight_col="weight",
    )
    _, _, _, _, test_sorted = prepare_xygw(
        test_df, feature_cols=FEATURE_COLS_V4, weight_col="weight",
    )
    # test_sorted совместим с V11 тоже — те же колонки плюс 3 новые,
    # которые есть в датасете.

    # ==== 5. Инференс обеих моделей ====
    print("\n[5/6] Инференс...")
    t = time.time()
    val_scores_v7 = predict_scores(model_v7, val_sorted, feature_cols=FEATURE_COLS_V4)
    val_scores_v39 = predict_scores(model_v39, val_sorted, feature_cols=FEATURE_COLS_V11)
    test_scores_v7 = predict_scores(model_v7, test_sorted, feature_cols=FEATURE_COLS_V4)
    test_scores_v39 = predict_scores(model_v39, test_sorted, feature_cols=FEATURE_COLS_V11)
    print(f"  инференс: {time.time() - t:.1f} сек")

    # ==== 6. Ансамбль ====
    print("\n[6/6] Ансамбль + оценка...")

    cold_ids_val = identify_cold_students(val)
    val_z_v7 = _zscore_per_student(val_sorted, val_scores_v7)
    val_z_v39 = _zscore_per_student(val_sorted, val_scores_v39)
    test_z_v7 = _zscore_per_student(test_sorted, test_scores_v7)
    test_z_v39 = _zscore_per_student(test_sorted, test_scores_v39)

    rec_val = _ensemble_per_student(
        val_sorted, val_z_v7, val_z_v39,
        args.weight_v7, args.weight_v39, k=5,
    )

    # Сегментная оценка (raw LTR, без fallback)
    mean_val, per_student = evaluate_pndcg5(rec_val, val_gt_raw, k=5)
    cold_scores = [per_student[sid] for sid in cold_ids_val if sid in per_student]
    warm_scores = [per_student[sid] for sid in per_student
                   if sid not in cold_ids_val]
    m_cold_raw = float(np.mean(cold_scores)) if cold_scores else 0.0
    m_warm_raw = float(np.mean(warm_scores)) if warm_scores else 0.0

    print(f"  val LTR raw: mean={mean_val:.4f}  cold={m_cold_raw:.4f}  "
          f"warm={m_warm_raw:.4f}")

    # Fallback на cold
    pop_weighted, _ = compute_popularity(train)
    fb_val = build_global_recommendations(val, pop_weighted, cold_ids_val, k=5)
    rec_val_fb = apply_fallback(rec_val, fb_val, k=5)
    mean_val_fb, per_student_fb = evaluate_pndcg5(rec_val_fb, val_gt_raw, k=5)

    m_cold_fb = float(np.mean([per_student_fb[sid] for sid in cold_ids_val
                               if sid in per_student_fb]))
    m_warm_fb = float(np.mean([per_student_fb[sid] for sid in per_student_fb
                               if sid not in cold_ids_val]))

    print(f"  val + fallback: mean={mean_val_fb:.4f}  cold={m_cold_fb:.4f}  "
          f"warm={m_warm_fb:.4f}")

    # Сохраняем val
    val_path = SUBMISSIONS_DIR / (
        f"val_ensemble_v7_v39_w{args.weight_v7}_{args.weight_v39}.csv"
    )
    write_submission(rec_val_fb, val_path)

    # Test
    rec_test = _ensemble_per_student(
        test_sorted, test_z_v7, test_z_v39,
        args.weight_v7, args.weight_v39, k=5,
    )
    cold_ids_test = identify_cold_students(test)
    fb_test = build_global_recommendations(test, pop_weighted, cold_ids_test, k=5)
    rec_test_final = apply_fallback(rec_test, fb_test, k=5)

    test_path = SUBMISSIONS_DIR / (
        f"submission_ensemble_v7_v39_w{args.weight_v7}_{args.weight_v39}.csv"
    )
    write_submission(rec_test_final, test_path)
    print(f"\n  Cold в test: {len(cold_ids_test)}")
    print(f"  Сохранено: {test_path}  ({len(rec_test_final)} строк)")

    # ==== Дополнительно: сравнение с v7 in-sample ====
    print("\n" + "=" * 78)
    print("СРАВНЕНИЕ")
    print("=" * 78)
    print(f"  v7 in-sample (val, с p2-утечкой)   = 0.4803")
    print(f"  v39 honest (val, без p2)           = 0.3464 (in-sample c fallback)")
    print(f"  Ансамбль (val, in-sample mix)      = {mean_val_fb:.4f}")
    print()
    print(f"  Если ансамбль на val ≈ v7 (0.4803) — warm не сломан, шанс на public есть.")
    print(f"  Если ансамбль на val < 0.47 — модель деградировала, не сабмитим.")
    print()
    print(f"[OK] Ансамбль завершён за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()