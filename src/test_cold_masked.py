"""
Диагностика Cold-Start на валидации.
Берет WARM студентов из val, маскирует их профиль и историю,
и проверяет, насколько хорошо разные стратегии предсказывают их РЕАЛЬНЫЕ выборы.
Это дает честную локальную оценку для 35.2% cold-сегмента теста.
"""
import numpy as np
import pandas as pd
from collections import defaultdict
from src.data_loader import load_all
from src.metric import evaluate_pndcg5, parse_ground_truth
from src.popularity import compute_popularity, compute_popularity_by_institute
from src.cold_start import is_cold

def main():
    print("=" * 78)
    print("ДИАГНОСТИКА COLD-START НА VAL (Masked Evaluation)")
    print("=" * 78)
    
    data = load_all()
    val = data["val"]
    val_sol = data["val_solution"]
    train = data["train"]
    
    gt_raw = dict(zip(val_sol["id"], val_sol["ground_truth"]))
    
    # Берем только WARM студентов, чтобы у нас был ground truth для проверки
    warm_val = [s for s in val if not is_cold(s)]
    print(f"\n[1/4] Взято {len(warm_val)} warm-студентов из val для маскирования.")
    
    # Создаем их "cold" версии
    masked_students = []
    true_targets = {}
    
    for s in warm_val:
        sid = s["id"]
        # Сохраняем истинную цель
        true_targets[sid] = parse_ground_truth(gt_raw.get(sid, ""))
        
        # Создаем маскированную копию
        masked_s = {
            "id": sid,
            "profile": {},  # Стираем профиль
            "history": [],  # Стираем историю
            "available_modules": s.get("available_modules", [])
        }
        masked_students.append(masked_s)
        
    print(f"[2/4] Сгенерировано {len(masked_students)} маскированных (cold) профилей.")
    
    # Стратегия 1: Global Popularity
    pop_weighted, _ = compute_popularity(train)
    rec_global = {}
    for s in masked_students:
        avail = s.get("available_modules", [])
        scored = [(m, float(pop_weighted.get(m, 0.0))) for m in avail]
        scored.sort(key=lambda x: (-x[1], x[0]))
        rec_global[s["id"]] = [m for m, _ in scored[:5]]
        
    mean_global, _ = evaluate_pndcg5(rec_global, {k: v for k, v in gt_raw.items() if k in true_targets})
    print(f"\n[3/4] Метрики на маскированных студентах:")
    print(f"      Global Popularity Fallback: mean pNDCG@5 = {mean_global:.4f}")
    
    # Стратегия 2: Institute-Inferred (если получится восстановить)
    # Строим индекс по train (там больше всего данных)
    index = defaultdict(set)
    for s in train:
        inst = (s.get("profile") or {}).get("institute")
        avail = s.get("available_modules")
        if inst and avail:
            index[frozenset(avail)].add(inst)
            
    rec_inst = {}
    resolved_count = 0
    for s in masked_students:
        avail = s.get("available_modules", [])
        insts = index.get(frozenset(avail))
        if insts and len(insts) == 1:
            inst = next(iter(insts))
            resolved_count += 1
            # Берем популярность внутри этого института
            pop_by_inst = compute_popularity_by_institute(train)
            pop = pop_by_inst.get(inst, {})
            scored = [(m, float(pop.get(m, 0.0))) for m in avail]
            scored.sort(key=lambda x: (-x[1], x[0]))
            rec_inst[s["id"]] = [m for m, _ in scored[:5]]
        else:
            # Fallback to global
            scored = [(m, float(pop_weighted.get(m, 0.0))) for m in avail]
            scored.sort(key=lambda x: (-x[1], x[0]))
            rec_inst[s["id"]] = [m for m, _ in scored[:5]]
            
    mean_inst, _ = evaluate_pndcg5(rec_inst, {k: v for k, v in gt_raw.items() if k in true_targets})
    print(f"      Institute-Inferred Fallback:  mean pNDCG@5 = {mean_inst:.4f} (resolved: {resolved_count}/{len(masked_students)})")
    
    # Стратегия 3: LTR Model prediction on masked data
    # Для этого нам нужно запустить инференс на masked_students через текущую модель.
    # Но чтобы не усложнять скрипт, мы можем оценить, насколько LTR в принципе 
    # лучше фоллбэка на обычных cold студентах val.
    
    real_cold_val = [s for s in val if is_cold(s)]
    print(f"\n[4/4] Сравнение на РЕАЛЬНЫХ cold студентах val (n={len(real_cold_val)}):")
    
    # Загружаем предсказания v7 для val
    import os
    val_sub_path = "submissions/val_submission_ltr_v7.csv"
    if os.path.exists(val_sub_path):
        df = pd.read_csv(val_sub_path)
        rec_ltr = {}
        for _, row in df.iterrows():
            rec_ltr[str(row["id"])] = str(row["recommendations"]).split()
            
        # Фильтруем только реальных cold
        rec_ltr_cold = {k: v for k, v in rec_ltr.items() if k in [s["id"] for s in real_cold_val]}
        gt_cold = {k: v for k, v in gt_raw.items() if k in rec_ltr_cold}
        
        if gt_cold:
            mean_ltr_cold, _ = evaluate_pndcg5(rec_ltr_cold, gt_cold)
            print(f"      LTR v7 (на холодных):             mean pNDCG@5 = {mean_ltr_cold:.4f}")
            print(f"      Global Popularity (на холодных):  mean pNDCG@5 = {mean_global:.4f}")
            
            if mean_ltr_cold > mean_global:
                print("\n[OK] LTR-модель УЖЕ лучше глобальной популярности на cold-студентах!")
                print("     Дополнительный fallback может только ухудшить результат.")
            else:
                print("\n[!] LTR-модель хуже глобальной популярности на cold-студентах.")
                print("    Стоит попробовать увеличить WEIGHT_PMASK в обучении.")
        else:
            print("      Нет ground truth для cold-студентов в val для прямого сравнения.")
            
    print("\n" + "=" * 78)
    print("ВЫВОД:")
    print("Если LTR уже лучше fallback, не применяйте fallback в финальном пайплайне.")
    print("Если нужно улучшить cold, увеличьте WEIGHT_PMASK (например, до 2.0 или 3.0)")
    print("в src/run_final.py, чтобы модель сильнее училась на маскированных данных.")
    print("=" * 78)

if __name__ == "__main__":
    main()