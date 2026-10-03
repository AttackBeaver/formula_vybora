"""
End-to-end baseline на популярности.

Запуск из корня проекта:
    python -m src.run_baseline
"""
from __future__ import annotations

from pathlib import Path

from src.data_loader import load_all
from src.inference import infer_popularity
from src.popularity import (
    compute_popularity,
    compute_popularity_by_institute,
)
from src.submit import write_submission

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"


def main() -> None:
    SUBMISSIONS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("BASELINE: Popularity")
    print("=" * 60)

    data = load_all()

    # ---- 1. Считаем популярность ----------------------------------------
    print("\n[1/4] Считаю популярность по train...")
    pop_global, counts_global = compute_popularity(data["train"])
    pop_by_inst = compute_popularity_by_institute(data["train"])
    print(f"      Модулей с score>0 (global): {len(pop_global)}")
    print(f"      Институтов в статистике:    {len(pop_by_inst)}")
    top3 = sorted(pop_global.items(), key=lambda x: -x[1])[:3]
    print(f"      Топ-3 по популярности:")
    for mid, sc in top3:
        print(f"        {mid}: {sc:.1f}  (raw count = {counts_global.get(mid, 0)})")

    # ---- 2. Рекомендации для val ----------------------------------------
    print("\n[2/4] Рекомендации для val...")
    rec_val = infer_popularity(data["val"], pop_global, pop_by_inst, k=5)
    val_path = SUBMISSIONS_DIR / "val_submission_popularity.csv"
    write_submission(rec_val, val_path)
    print(f"      Записано: {val_path}  ({len(rec_val)} строк)")

    # ---- 3. Рекомендации для test ---------------------------------------
    print("\n[3/4] Рекомендации для test...")
    rec_test = infer_popularity(data["test"], pop_global, pop_by_inst, k=5)
    test_path = SUBMISSIONS_DIR / "submission_popularity.csv"
    write_submission(rec_test, test_path)
    print(f"      Записано: {test_path}  ({len(rec_test)} строк)")

    # ---- 4. Быстрая проверка первых строк -------------------------------
    print("\n[4/4] Первые 3 строки val_submission:")
    with val_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 3:
                break
            print(f"      {line.rstrip()}")

    print("\n[OK] Baseline готов.")


if __name__ == "__main__":
    main()