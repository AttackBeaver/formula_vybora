"""
Модель коллаборативной фильтрации ALS (Alternating Least Squares).
Используется для учета нелинейных режимов взаимодействия пользователя с модулем, которые трудно охватить с помощью разработки функций LTR (латентные фильтры).
Как независимый источник сигнала, в будущем он будет объединен с LTR v7.

Использование:
    python -m src.als_model
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares

from src.data_loader import load_all
from src.metric import evaluate_pndcg5, parse_ground_truth

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Веса приоритетов (совпадают с метрикой и LTR)
PRIORITY_WEIGHTS: dict[int, float] = {
    1: 1.00, 2: 0.80, 3: 0.60, 4: 0.45, 5: 0.35,
}

# Гиперпараметры ALS (можно тюнить позже)
ALS_FACTORS = 64
ALS_REGULARIZATION = 0.1
ALS_ITERATIONS = 20


# =========================================================================
# 1. Построение матрицы взаимодействий
# =========================================================================
def build_interaction_matrix(
    students: list[dict[str, Any]],
    user_to_idx: dict[str, int],
    module_to_idx: dict[str, int],
) -> sparse.csr_matrix:
    """
    Строит разреженную матрицу (студенты × модули) на основе истории.
    Значения — сумма весов приоритетов (если студент выбирал модуль 
    в разных периодах, веса суммируются, хотя обычно это 1 выбор).
    """
    n_users = len(user_to_idx)
    n_items = len(module_to_idx)
    
    # Используем lil_matrix для быстрого добавления элементов
    matrix = sparse.lil_matrix((n_users, n_items), dtype=np.float32)
    
    for student in students:
        sid = student.get("id") or student.get("student_id")
        if sid not in user_to_idx:
            continue
            
        u_idx = user_to_idx[sid]
        
        for h in student.get("history", []):
            for choice in h.get("choices", []):
                mid = choice["module_id"]
                prio = choice.get("priority", 5)
                weight = PRIORITY_WEIGHTS.get(prio, 0.35)
                
                if mid in module_to_idx:
                    m_idx = module_to_idx[mid]
                    matrix[u_idx, m_idx] += weight
                    
    return matrix.tocsr()


# =========================================================================
# 2. Обучение модели
# =========================================================================
def train_als_model(
    train_students: list[dict[str, Any]],
    all_modules: list[dict[str, Any]],
    factors: int = ALS_FACTORS,
    regularization: float = ALS_REGULARIZATION,
    iterations: int = ALS_ITERATIONS,
) -> Tuple[AlternatingLeastSquares, dict[str, int], dict[int, str], dict[str, int]]:
    """
    Обучает ALS на train-выборке.
    Возвращает: (model, user_to_idx, idx_to_module, module_to_idx)
    """
    print(f"[ALS] Сборка матрицы взаимодействий ({len(train_students)} студентов)...")
    
    # Словари для маппинга ID -> индексы матрицы
    user_to_idx: dict[str, int] = {}
    for i, s in enumerate(train_students):
        sid = s.get("id") or s.get("student_id")
        if sid:
            user_to_idx[sid] = i
            
    module_to_idx: dict[str, int] = {m["module_id"]: i for i, m in enumerate(all_modules)}
    idx_to_module: dict[int, str] = {i: m["module_id"] for i, m in enumerate(all_modules)}
    
    matrix = build_interaction_matrix(train_students, user_to_idx, module_to_idx)
    print(f"[ALS] Матрица готова: shape={matrix.shape}, nnz={matrix.nnz}")
    
    print(f"[ALS] Обучение (factors={factors}, reg={regularization}, iter={iterations})...")
    t0 = time.time()
    
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        random_state=42,
    )
    model.fit(matrix)
    
    print(f"[ALS] Обучение завершено за {time.time() - t0:.1f} сек.")
    return model, user_to_idx, idx_to_module, module_to_idx


# =========================================================================
# 3. Инференс (подсчёт скоров для кандидатов)
# =========================================================================
def predict_als_scores(
    students: list[dict[str, Any]],
    model: AlternatingLeastSquares,
    user_to_idx: dict[str, int],
    idx_to_module: dict[int, str],
    module_to_idx: dict[str, int],
) -> Dict[str, Dict[str, float]]:
    """
    Считает скоры для всех available_modules каждого студента.
    Использует прямое перемножение латентных векторов (быстрее и прозрачнее, 
    чем model.recommend).
    
    Returns:
        dict[student_id, dict[module_id, score]]
    """
    # Извлекаем латентные факторы (numpy arrays)
    user_factors = model.user_factors
    item_factors = model.item_factors
    
    # Для cold-студентов (нет в train) будем использовать нулевой вектор
    # (их скоры будут 0, и они уйдут в fallback по популярности)
    n_factors = user_factors.shape[1]
    zero_user_vector = np.zeros(n_factors, dtype=np.float32)
    
    all_scores: Dict[str, Dict[str, float]] = {}
    
    for student in students:
        sid = student.get("id") or student.get("student_id")
        if not sid:
            continue
            
        available = student.get("available_modules", [])
        if not available:
            all_scores[sid] = {}
            continue
            
        # Получаем вектор пользователя
        u_idx = user_to_idx.get(sid)
        if u_idx is not None:
            u_vec = user_factors[u_idx]
        else:
            u_vec = zero_user_vector
            
        # Получаем индексы доступных модулей
        candidate_indices = []
        valid_modules = []
        for mid in available:
            if mid in module_to_idx:
                candidate_indices.append(module_to_idx[mid])
                valid_modules.append(mid)
                
        if not candidate_indices:
            all_scores[sid] = {}
            continue
            
        candidate_indices_arr = np.array(candidate_indices)
        
        # Матричное умножение: (1, factors) x (N, factors).T -> (1, N)
        i_vecs = item_factors[candidate_indices_arr]
        scores_arr = np.dot(u_vec, i_vecs.T).flatten()
        
        # Сохраняем в словарь
        all_scores[sid] = {
            valid_modules[i]: float(scores_arr[i]) 
            for i in range(len(valid_modules))
        }
        
    return all_scores


# =========================================================================
# 4. Оценка на Val (для локального контроля)
# =========================================================================
def evaluate_als_on_val(
    val_students: list[dict[str, Any]],
    val_gt_raw: dict[str, str],
    als_scores: Dict[str, Dict[str, float]],
    k: int = 5,
) -> float:
    """
    Формирует top-K рекомендаций на основе ALS-скоров и считает pNDCG@5.
    """
    predictions: dict[str, list[str]] = {}
    
    for student in val_students:
        sid = student.get("id") or student.get("student_id")
        if sid not in als_scores:
            continue
            
        # Сортируем доступные модули по убыванию скора
        scored_modules = sorted(
            als_scores[sid].items(), 
            key=lambda x: (-x[1], x[0]) # по скору, затем по ID для стабильности
        )
        
        # Берем top-k
        top_k = [mid for mid, _ in scored_modules[:k]]
        predictions[sid] = top_k
        
    mean_score, _ = evaluate_pndcg5(predictions, val_gt_raw, k=k)
    return mean_score


# =========================================================================
# CLI: Запуск и проверка
# =========================================================================
def main() -> None:
    print("=" * 72)
    print("ALS MODEL: Обучение и оценка на Val")
    print("=" * 72)
    
    data = load_all()
    train = data["train"]
    val = data["val"]
    modules = data["modules"]
    val_solution = data["val_solution"]
    
    val_gt_raw = dict(zip(val_solution["id"], val_solution["ground_truth"]))
    
    # 1. Обучение
    model, user_to_idx, idx_to_module, module_to_idx = train_als_model(
        train, modules,
        factors=ALS_FACTORS,
        regularization=ALS_REGULARIZATION,
        iterations=ALS_ITERATIONS,
    )
    
    # 2. Инференс на val
    print("\n[ALS] Инференс скоров для val-студентов...")
    val_scores = predict_als_scores(val, model, user_to_idx, idx_to_module, module_to_idx)
    
    # 3. Оценка
    print("\n[ALS] Оценка pNDCG@5 на val (только ALS, без fallback и ансамбля):")
    als_metric = evaluate_als_on_val(val, val_gt_raw, val_scores, k=5)
    print(f"      Val pNDCG@5 (ALS only) = {als_metric:.4f}")
    
    print("\n[INFO] Сравнение с базовыми линиями:")
    print(f"      Popularity Baseline  ≈ 0.28")
    print(f"      LTR v7 (Holdout)     ≈ 0.5134")
    print(f"      ALS (Val)            = {als_metric:.4f}")
    
    if als_metric > 0.25:
        print("\n[OK] ALS обучился и даёт сигнал лучше рандома/популярности.")
        print("     Следующий шаг: Ensemble (LTR + ALS) в run_ensemble.py.")
    else:
        print("\n[!] ALS даёт слабый сигнал. Возможно, нужно увеличить factors или iterations.")

if __name__ == "__main__":
    main()