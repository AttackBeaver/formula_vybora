"""
Content-based cold-start fallback.
Идея: для каждого cold-студента найти top-K наиболее похожих warm-студентов
по available_modules (Jaccard similarity), затем использовать их рекомендации.
Это должно дать более качественный fallback, чем global popularity.
Запуск:
python -m src.content_cold_start
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.cold_start import identify_cold_students, is_cold
from src.data_loader import load_all
from src.metric import evaluate_pndcg5
from src.popularity import compute_popularity
from src.submit import write_submission

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"


def jaccard_similarity(set1: set[str], set2: set[str]) -> float:
    """Jaccard similarity между двумя множествами."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def build_warm_student_index(
    students: list[dict[str, Any]],
    cold_ids: set[str],
) -> dict[str, set[str]]:
    """
    Строит индекс warm-студентов: student_id → set(available_modules).
    """
    index: dict[str, set[str]] = {}
    for s in students:
        sid = s.get("id") or s.get("student_id")
        if sid is None or sid in cold_ids:
            continue
        avail = set(s.get("available_modules") or [])
        if avail:
            index[sid] = avail
    return index


def find_similar_warm_students(
    cold_avail: set[str],
    warm_index: dict[str, set[str]],
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """
    Находит top-K наиболее похожих warm-студентов для cold-студента.
    Возвращает список (student_id, similarity).
    """
    similarities: list[tuple[str, float]] = []
    for sid, warm_avail in warm_index.items():
        sim = jaccard_similarity(cold_avail, warm_avail)
        if sim > 0:
            similarities.append((sid, sim))
    
    # Сортируем  по убыванию similarity
    similarities.sort(key=lambda x: -x[1])
    return similarities[:top_k]


def build_content_recommendations(
    students: list[dict[str, Any]],
    warm_index: dict[str, set[str]],
    val_gt_raw: dict[str, str],
    cold_ids: set[str],
    top_k_similar: int = 20,
    k: int = 5,
) -> dict[str, list[str]]:
    """
    Строит рекомендации для cold-студентов на основе похожих warm-студентов.
    Для каждого cold-студента:
    1. Находит top-K похожих warm-студентов по available_modules
    2. Собирает их history (все модули, которые они выбирали)
    3. Ранжирует по частоте выбора (с учётом приоритетов)
    4. Фильтрует по available_modules cold-студента
    """
    from src.metric import parse_ground_truth
    
    # Строим индекс рекомендаций warm-студентов из val_solution
    warm_recs: dict[str, dict[str, int]] = {}
    for sid, gt_str in val_gt_raw.items():
        if sid in warm_index:
            gt = parse_ground_truth(gt_str)
            warm_recs[sid] = gt
    
    # Также добавляем историю из train
    # (но для val/test warm-студентов у нас есть val_solution)
    
    recommendations: dict[str, list[str]] = {}
    
    for s in students:
        sid = s.get("id") or s.get("student_id")
        if sid is None or sid not in cold_ids:
            continue
        
        cold_avail = set(s.get("available_modules") or [])
        if not cold_avail:
            recommendations[sid] = []
            continue
        
        # Находим похожих warm-студентов
        similar = find_similar_warm_students(cold_avail, warm_index, top_k_similar)
        
        if not similar:
            # Нет похожих — оставляем пустым (потом заполним global popularity)
            recommendations[sid] = []
            continue
        
        # Собираем модули от похожих студентов с весами
        module_scores: dict[str, float] = defaultdict(float)
        for warm_sid, sim in similar:
            if warm_sid in warm_recs:
                for mid, prio in warm_recs[warm_sid].items():
                    if mid in cold_avail:
                        # Вес = similarity * weight(priority)
                        weight = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.45, 5: 0.35}.get(prio, 0.35)
                        module_scores[mid] += sim * weight
        
        # Ранжируем по score
        scored_modules = sorted(module_scores.items(), key=lambda x: -x[1])
        recommendations[sid] = [mid for mid, _ in scored_modules[:k]]
    
    return recommendations


def compute_segment_metrics(
    rec: dict[str, list[str]],
    gt_raw: dict[str, str],
    cold_ids: set[str],
    k: int = 5,
) -> tuple[float, float, float]:
    """
    Вычисляет метрики по сегментам: mean, cold, warm.
    """
    mean_all, per_student = evaluate_pndcg5(rec, gt_raw, k=k)
    cold_scores = [s for sid, s in per_student.items() if sid in cold_ids]
    warm_scores = [s for sid, s in per_student.items() if sid not in cold_ids]
    c = float(np.mean(cold_scores)) if cold_scores else 0.0
    w = float(np.mean(warm_scores)) if warm_scores else 0.0
    return mean_all, c, w


def main() -> None:
    parser = argparse.ArgumentParser(description="Content-based cold-start fallback")
    parser.add_argument("--top-k-similar", type=int, default=20)
    args = parser.parse_args()
    
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    
    print("=" * 78)
    print("CONTENT-BASED COLD-START FALLBACK")
    print("=" * 78)
    
    # Загружаем данные
    data = load_all()
    train = data["train"]
    val = data["val"]
    test = data["test"]
    val_solution = data["val_solution"]
    val_gt_raw = dict(zip(val_solution["id"], val_solution["ground_truth"]))
    
    # Идентифицируем cold-студентов
    cold_ids_val = identify_cold_students(val)
    cold_ids_test = identify_cold_students(test)
    
    print(f"\n[1/5] Cold-студентов: val={len(cold_ids_val)}, test={len(cold_ids_test)}")
    
    # Строим индекс warm-студентов из val + test
    # (используем оба набора для максимального покрытия)
    warm_index_val = build_warm_student_index(val, cold_ids_val)
    warm_index_test = build_warm_student_index(test, cold_ids_test)
    warm_index = {**warm_index_val, **warm_index_test}
    
    print(f"[2/5] Warm-студентов в индексе: {len(warm_index)}")
    
    # Строим content-based рекомендации для val
    print(f"\n[3/5] Строю content-based рекомендации для val (top_k_similar={args.top_k_similar})...")
    rec_content_val = build_content_recommendations(
        val, warm_index, val_gt_raw, cold_ids_val,
        top_k_similar=args.top_k_similar, k=5,
    )
    
    # Сравниваем с global popularity
    pop_weighted, _ = compute_popularity(train)
    
    # Загружаем LTR-рекомендации для val
    from src.evaluate import load_predictions
    ltr_val_path = SUBMISSIONS_DIR / "val_submission_ltr_v9_tuned.csv"
    if not ltr_val_path.exists():
        print(f"\n[!] Не найден LTR submission: {ltr_val_path}")
        print("    Сначала запустите: python -m src.run_ltr_v9_tuned")
        return
    
    rec_ltr_val = load_predictions(ltr_val_path)
    
    # Применяем fallback
    from src.cold_start import apply_fallback, build_global_recommendations
    
    # Global popularity fallback
    rec_global_val = build_global_recommendations(val, pop_weighted, cold_ids_val, k=5)
    rec_ltr_global = apply_fallback(rec_ltr_val, rec_global_val, k=5)
    
    # Content-based fallback
    rec_ltr_content = apply_fallback(rec_ltr_val, rec_content_val, k=5)
    
    # Оцениваем метрики
    print(f"\n[4/5] Метрики на val:")
    
    mean_ltr, cold_ltr, warm_ltr = compute_segment_metrics(rec_ltr_val, val_gt_raw, cold_ids_val, k=5)
    mean_global, cold_global, warm_global = compute_segment_metrics(rec_ltr_global, val_gt_raw, cold_ids_val, k=5)
    mean_content, cold_content, warm_content = compute_segment_metrics(rec_ltr_content, val_gt_raw, cold_ids_val, k=5)
    
    print(f"  LTR (без fallback):      mean={mean_ltr:.4f}  cold={cold_ltr:.4f}  warm={warm_ltr:.4f}")
    print(f"  LTR + Global popularity: mean={mean_global:.4f}  cold={cold_global:.4f}  warm={warm_global:.4f}")
    print(f"  LTR + Content-based:     mean={mean_content:.4f}  cold={cold_content:.4f}  warm={warm_content:.4f}")
    
    # Выбираем лучший fallback
    if cold_content > cold_global + 0.005:
        print(f"\n[OK] Content-based лучше global popularity на cold: {cold_content:.4f} vs {cold_global:.4f}")
        best_fallback = "content"
        best_cold = cold_content
    else:
        print(f"\n[!] Global popularity лучше content-based на cold: {cold_global:.4f} vs {cold_content:.4f}")
        best_fallback = "global"
        best_cold = cold_global
    
    # Применяем к test
    print(f"\n[5/5] Применяю к test...")
    
    ltr_test_path = SUBMISSIONS_DIR / "submission_v9_tuned.csv"
    if not ltr_test_path.exists():
        print(f"\n[!] Не найден LTR submission для test: {ltr_test_path}")
        return
    
    rec_ltr_test = load_predictions(ltr_test_path)
    
    if best_fallback == "content":
        rec_content_test = build_content_recommendations(
            test, warm_index, val_gt_raw, cold_ids_test,
            top_k_similar=args.top_k_similar, k=5,
        )
        rec_test = apply_fallback(rec_ltr_test, rec_content_test, k=5)
    else:
        rec_global_test = build_global_recommendations(test, pop_weighted, cold_ids_test, k=5)
        rec_test = apply_fallback(rec_ltr_test, rec_global_test, k=5)
    
    output_path = SUBMISSIONS_DIR / f"submission_v9_tuned_{best_fallback}.csv"
    write_submission(rec_test, output_path)
    
    print(f"\n[OK] Сохранено: {output_path}")
    print(f"     Лучший fallback: {best_fallback} (cold={best_cold:.4f})")


if __name__ == "__main__":
    main()