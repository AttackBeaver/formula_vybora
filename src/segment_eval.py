"""
Оценка submission по сегментам на val-наборе.
Позволяет увидеть реальную картину по periods=0, 1, 2, 3 и cold,
вместо обманчивого усреднённого test_estimate.
Запуск:
python -m src.segment_eval --submission submissions/val_submission_ltr_v7.csv
"""
import argparse
import pandas as pd
from pathlib import Path
from src.data_loader import load_all
from src.metric import evaluate_pndcg5
from src.cold_start import is_cold

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser(description="Оценка по сегментам")
    parser.add_argument("--submission", type=str, default=str(PROJECT_ROOT / "submissions" / "val_submission_ltr_v7.csv"))
    args = parser.parse_args()
    
    sub_path = Path(args.submission)
    if not sub_path.exists():
        print(f"[!] Файл не найден: {sub_path}")
        return
        
    print(f"Оцениваем: {sub_path.name}")
    
    data = load_all()
    val = data["val"]
    val_sol = data["val_solution"]
    gt_raw = dict(zip(val_sol["id"], val_sol["ground_truth"]))
    
    df = pd.read_csv(sub_path)
    rec = {}
    for _, row in df.iterrows():
        rec[str(row["id"])] = str(row["recommendations"]).split()
        
    mean_all, per_student = evaluate_pndcg5(rec, gt_raw, k=5)
    
    segments = {
        "cold": [],
        "periods=0": [],
        "periods=1": [],
        "periods=2": [],
        "periods=3": [],
        "periods>=4": []
    }
    
    for s in val:
        sid = str(s["id"])
        if sid not in per_student:
            continue
            
        score = per_student[sid]
        hist = s.get("history", [])
        # Считаем периоды, в которых были выборы
        n_periods = len([h for h in hist if len(h.get("choices", [])) > 0])
        
        if is_cold(s):
            segments["cold"].append(score)
            segments["periods=0"].append(score)
        else:
            if n_periods == 1:
                segments["periods=1"].append(score)
            elif n_periods == 2:
                segments["periods=2"].append(score)
            elif n_periods == 3:
                segments["periods=3"].append(score)
            else:
                segments["periods>=4"].append(score)
                
    print("\n" + "="*60)
    print(f"{'Сегмент':<12} | {'N':<5} | {'Доля':<6} | {'Mean pNDCG@5'}")
    print("-" * 60)
    
    total = sum(len(scores) for scores in segments.values() if scores)
    for name, scores in segments.items():
        if not scores:
            continue
        mean = sum(scores) / len(scores)
        share = len(scores) / total * 100 if total > 0 else 0
        print(f"{name:<12} | {len(scores):<5} | {share:>5.1f}% | {mean:.4f}")
    print("=" * 60)
    
    # Расчет "честного" estimate под распределение test
    # Из README: periods=0: 35.2%, periods=1: 2.0%, periods=2: 32.4%, periods=3: 30.4%
    # Если в val нет periods=3, мы это сразу увидим (будет 0 или N/A)
    
    test_weights = {"periods=0": 0.352, "periods=1": 0.020, "periods=2": 0.324, "periods=3": 0.304}
    estimate = 0.0
    for seg, weight in test_weights.items():
        if segments[seg]:
            estimate += (sum(segments[seg]) / len(segments[seg])) * weight
            
    print(f"\n[!] Взвешенная оценка под распределение TEST: {estimate:.4f}")
    print(f"    (Сравните с вашим лучшим Public Score)")

if __name__ == "__main__":
    main()