"""
Ensemble LTR v7 + ALS.
Стратегия:
1. Получаем скоры от LTR v7 и ALS для каждого студента.
2. Нормализуем скоры внутри каждого студента (Min-Max scaling), чтобы привести их к общему масштабу [0, 1].
3. Ансамблируем: score_ens = alpha * score_ltr + (1 - alpha) * score_als.
4. Подбираем оптимальный alpha на val-выборке.
5. Применяем лучший alpha к test-выборке и сохраняем submission.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.als_model import predict_als_scores, train_als_model
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
    make_target_from_val_solution,
)
from src.ltr_model import (
    FEATURE_COLS_V4,
    load_model,
    prepare_xygw,
    predict_scores,
)
from src.metric import evaluate_pndcg5
from src.popularity import compute_popularity
from src.submit import write_submission

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"


def normalize_scores(scores_dict: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """Min-Max нормализация скоров для каждого студента."""
    normalized = {}
    for sid, scores in scores_dict.items():
        if not scores:
            normalized[sid] = scores
            continue
        vals = list(scores.values())
        min_v, max_v = min(vals), max(vals)
        if max_v - min_v > 1e-9:
            normalized[sid] = {m: (s - min_v) / (max_v - min_v) for m, s in scores.items()}
        else:
            normalized[sid] = {m: 1.0 for m in scores.keys()}
    return normalized


def ensemble_scores(
    ltr_scores: dict[str, dict[str, float]],
    als_scores: dict[str, dict[str, float]],
    alpha: float,
) -> dict[str, dict[str, float]]:
    """Ансамблирование скоров: alpha * LTR + (1 - alpha) * ALS."""
    ens = {}
    for sid in ltr_scores.keys():
        ltr = ltr_scores.get(sid, {})
        als = als_scores.get(sid, {})
        all_modules = set(ltr.keys()) | set(als.keys())
        ens[sid] = {}
        for m in all_modules:
            s_ltr = ltr.get(m, 0.0)
            s_als = als.get(m, 0.0)
            ens[sid][m] = alpha * s_ltr + (1.0 - alpha) * s_als
    return ens


def evaluate_ensemble(
    ens_scores: dict[str, dict[str, float]],
    val_gt_raw: dict[str, str],
    k: int = 5,
) -> float:
    """Оценка ансамбля на val."""
    predictions = {}
    for sid, scores in ens_scores.items():
        sorted_modules = sorted(scores.items(), key=lambda x: -x[1])
        predictions[sid] = [m for m, _ in sorted_modules[:k]]
    
    mean_score, _ = evaluate_pndcg5(predictions, val_gt_raw, k=k)
    return mean_score


def encode_categories_for_inference(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Приводит категориальные признаки к единому набору категорий,
    собранному из всех студентов (train + val + test).
    Это необходимо для корректного инференса LightGBM, так как модель
    при обучении видела определенный набор категорий.
    """
    all_institutes = set()
    all_specialties = set()
    for s in train + val + test:
        prof = s.get("profile") or {}
        all_institutes.add(str(prof.get("institute") or "UNKNOWN"))
        all_specialties.add(str(prof.get("specialty") or "UNKNOWN"))
    
    inst_dtype = pd.CategoricalDtype(categories=sorted(all_institutes))
    spec_dtype = pd.CategoricalDtype(categories=sorted(all_specialties))
    
    val_df = val_df.copy()
    test_df = test_df.copy()
    
    val_df["institute"] = val_df["institute"].astype(str).astype(inst_dtype)
    val_df["specialty"] = val_df["specialty"].astype(str).astype(spec_dtype)
    
    test_df["institute"] = test_df["institute"].astype(str).astype(inst_dtype)
    test_df["specialty"] = test_df["specialty"].astype(str).astype(spec_dtype)
    
    return val_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Ensemble LTR v7 + ALS")
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="Вес LTR (если None, будет grid search на val)"
    )
    args = parser.parse_args()

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    print("=" * 78)
    print("ENSEMBLE LTR v7 + ALS")
    print("=" * 78)
    t0 = time.time()

    # 1. Данные
    print("\n[1/6] Загрузка данных...")
    data = load_all()
    train = data["train"]
    val = data["val"]
    test = data["test"]
    modules = data["modules"]
    val_solution = data["val_solution"]
    val_gt_raw = dict(zip(val_solution["id"], val_solution["ground_truth"]))

    # 2. ALS модель
    print("\n[2/6] Обучение ALS...")
    als_model, user_to_idx, idx_to_module, module_to_idx = train_als_model(train, modules)
    
    print("      Инференс ALS на val...")
    als_scores_val = predict_als_scores(val, als_model, user_to_idx, idx_to_module, module_to_idx)
    print("      Инференс ALS на test...")
    als_scores_test = predict_als_scores(test, als_model, user_to_idx, idx_to_module, module_to_idx)

    # 3. LTR v7 модель
    print("\n[3/6] Загрузка LTR v7...")
    ltr_model_path = MODELS_DIR / "lgbm_ranker_v7.txt"
    if not ltr_model_path.exists():
        print(f"      [!] Модель не найдена: {ltr_model_path}")
        print("      Запустите сначала: python -m src.run_ltr_v7")
        return
    
    ltr_model = load_model(ltr_model_path)
    
    print("      Построение статистик и val датасета для LTR...")
    stats = FeatureStats.fit(train, modules)
    stats.add_val_solution_popularity(val_solution)
    
    val_df = build_dataset(
        val, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )
    _, _, _, _, val_sorted = prepare_xygw(val_df, feature_cols=FEATURE_COLS_V4)
    
    print("      Построение test датасета для LTR...")
    test_df = build_dataset(
        test, stats,
        candidates_fn=candidates_available,
        target_fn=None,
        past_periods=None,
        max_history_periods=2,
    )
    _, _, _, _, test_sorted = prepare_xygw(test_df, feature_cols=FEATURE_COLS_V4)

    # === КРИТИЧНО: Кодирование категориальных признаков ===
    print("      Кодирование категориальных признаков...")
    val_sorted, test_sorted = encode_categories_for_inference(
        val_sorted, test_sorted, train, val, test
    )

    print("      Инференс LTR на val...")
    ltr_scores_val_arr = predict_scores(ltr_model, val_sorted, feature_cols=FEATURE_COLS_V4)
    val_sorted = val_sorted.copy()
    val_sorted["ltr_score"] = ltr_scores_val_arr
    ltr_scores_val = {
        sid: dict(zip(group["module_id"], group["ltr_score"]))
        for sid, group in val_sorted.groupby("student_id", sort=False)
    }

    print("      Инференс LTR на test...")
    ltr_scores_test_arr = predict_scores(ltr_model, test_sorted, feature_cols=FEATURE_COLS_V4)
    test_sorted = test_sorted.copy()
    test_sorted["ltr_score"] = ltr_scores_test_arr
    ltr_scores_test = {
        sid: dict(zip(group["module_id"], group["ltr_score"]))
        for sid, group in test_sorted.groupby("student_id", sort=False)
    }

    # 4. Нормализация скоров
    print("\n[4/6] Нормализация скоров (Min-Max per student)...")
    ltr_scores_val_norm = normalize_scores(ltr_scores_val)
    als_scores_val_norm = normalize_scores(als_scores_val)
    
    ltr_scores_test_norm = normalize_scores(ltr_scores_test)
    als_scores_test_norm = normalize_scores(als_scores_test)

    # 5. Подбор alpha на val
    if args.alpha is None:
        print("\n[5/6] Grid search alpha на val...")
        alphas = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        best_alpha = 0.8
        best_score = -1.0
        for alpha in alphas:
            ens_scores = ensemble_scores(ltr_scores_val_norm, als_scores_val_norm, alpha)
            score = evaluate_ensemble(ens_scores, val_gt_raw, k=5)
            print(f"      alpha={alpha:.2f} -> val pNDCG@5 = {score:.4f}")
            if score > best_score:
                best_score = score
                best_alpha = alpha
        print(f"\n      Лучший alpha = {best_alpha} (val pNDCG@5 = {best_score:.4f})")
        alpha = best_alpha
    else:
        alpha = args.alpha
        print(f"\n[5/6] Использование заданного alpha = {alpha}")

    # 6. Финальный инференс и fallback
    print("\n[6/6] Финальный инференс на test + fallback...")
    ens_scores_test = ensemble_scores(ltr_scores_test_norm, als_scores_test_norm, alpha)
    
    rec_test_ens = {}
    for sid, scores in ens_scores_test.items():
        sorted_modules = sorted(scores.items(), key=lambda x: -x[1])
        rec_test_ens[sid] = [m for m, _ in sorted_modules[:5]]

    cold_ids_test = identify_cold_students(test)
    pop_weighted, _ = compute_popularity(train)
    fb_test = build_global_recommendations(test, pop_weighted, cold_ids_test, k=5)
    rec_test_final = apply_fallback(rec_test_ens, fb_test, k=5)

    test_path = SUBMISSIONS_DIR / "submission_ensemble.csv"
    write_submission(rec_test_final, test_path)
    
    print(f"\n      Cold-студентов в test: {len(cold_ids_test)}")
    print(f"      Сохранено: {test_path} ({len(rec_test_final)} строк)")
    print(f"\n[OK] Пайплайн завершён за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()