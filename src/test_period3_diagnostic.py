"""
Диагностика группы test periods=3 (30.4% test, public ≈ 0.17).

Сравниваем periods=2 (32.4%, public ≈ 0.50) и periods=3 (30.4%, public ≈ 0.17)
в test. Одинаковая задача (target=P4), но метрика различается в 3 раза.
Ищем, что отличает эти группы.

Смотрим:
    1. Пересечение модулей между соседними периодами (повторы).
    2. Число уникальных модулей в истории.
    3. Общий пул модулей — те же модули или другие.
    4. Содержимое available_modules (одинаковое / разное).
    5. Различия в профиле (institute, specialty).
    6. Плотность / распределение популярности модулей.

Запуск:
    python -m src.test_period3_diagnostic
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from src.data_loader import load_all


def _n_periods(r: dict[str, Any]) -> int:
    return len(r.get("history") or [])


def _periods_list(r: dict[str, Any]) -> list[int]:
    return [h.get("period") for h in r.get("history") or []]


def _mods_in_period(r: dict[str, Any], period: int) -> list[str]:
    for h in r.get("history") or []:
        if h.get("period") == period:
            return [c["module_id"] for c in h.get("choices", [])]
    return []


def _profile(r: dict[str, Any]) -> tuple[str, str]:
    p = r.get("profile") or {}
    return (p.get("institute") or "UNKNOWN", p.get("specialty") or "UNKNOWN")


def _avg_repeat_rate(records: list[dict[str, Any]]) -> float:
    """Средняя доля модулей, повторяющихся между соседними периодами."""
    rates = []
    for r in records:
        history = r.get("history") or []
        if len(history) < 2:
            continue
        for i in range(1, len(history)):
            prev_mods = {c["module_id"] for c in history[i - 1].get("choices", [])}
            cur_mods = {c["module_id"] for c in history[i].get("choices", [])}
            if not prev_mods:
                continue
            rates.append(len(prev_mods & cur_mods) / len(prev_mods))
    return float(np.mean(rates)) if rates else 0.0


def main() -> None:
    data = load_all()
    test = data["test"]
    train = data["train"]
    val = data["val"]

    print("=" * 78)
    print("ДИАГНОСТИКА TEST periods=3 vs periods=2")
    print("=" * 78)

    test_p2 = [r for r in test if _n_periods(r) == 2]
    test_p3 = [r for r in test if _n_periods(r) == 3]

    print(f"\n[Размеры]")
    print(f"  test periods=2: {len(test_p2)}")
    print(f"  test periods=3: {len(test_p3)}")

    # --- 1. Какие периоды встречаются в истории каждой группы ---
    print(f"\n[1] Периоды в истории")
    for name, records in [("periods=2", test_p2), ("periods=3", test_p3)]:
        pc = Counter()
        for r in records:
            for p in _periods_list(r):
                pc[p] += 1
        print(f"  {name}: {dict(sorted(pc.items()))}")

    # --- 2. Повторы модулей между соседними периодами ---
    print(f"\n[2] Повторы модулей между соседними периодами")
    for name, records in [("periods=2", test_p2), ("periods=3", test_p3)]:
        rate = _avg_repeat_rate(records)
        print(f"  {name}: средняя доля повторов = {rate:.3f}")

    # --- 3. Число уникальных модулей в истории ---
    print(f"\n[3] Число уникальных модулей в истории")
    for name, records in [("periods=2", test_p2), ("periods=3", test_p3)]:
        counts = [
            len({c["module_id"] for h in r.get("history") or [] for c in h.get("choices", [])})
            for r in records
        ]
        print(f"  {name}: mean={np.mean(counts):.2f}  std={np.std(counts):.2f}  "
              f"min={min(counts)}  max={max(counts)}")

    # --- 4. Доступные модули — те же или разные ---
    print(f"\n[4] Сигнатуры available_modules")
    for name, records in [("periods=2", test_p2), ("periods=3", test_p3)]:
        sigs = Counter(frozenset(r.get("available_modules") or []) for r in records)
        print(f"  {name}: {len(sigs)} уникальных сигнатур, "
              f"топ размер = {len(sigs.most_common(1)[0][0])}, "
              f"студентов в топ = {sigs.most_common(1)[0][1]}")

    # --- 5. Профиль ---
    print(f"\n[5] Топ-институты (по частоте)")
    for name, records in [("periods=2", test_p2), ("periods=3", test_p3)]:
        insts = Counter(_profile(r)[0] for r in records)
        print(f"  {name}:")
        for inst, cnt in insts.most_common(4):
            print(f"      {cnt:>5}  {inst}")

    # --- 6. Пересечение с train и val ---
    print(f"\n[6] Пересечение с train / val по id")
    train_ids = {s.get("student_id") for s in train}
    val_ids = {s.get("id") for s in val}
    for name, records in [("periods=2", test_p2), ("periods=3", test_p3)]:
        in_train = sum(1 for r in records if r.get("id") in train_ids)
        in_val = sum(1 for r in records if r.get("id") in val_ids)
        print(f"  {name}: in_train={in_train}  in_val={in_val}")

    # --- 7. Популярность модулей: какие модули выбирали в P3 ---
    # Для periods=3 в test P3 — самый последний период, и их выбор может отличаться
    # от того, что делают periods=2.
    print(f"\n[7] Распределение модулей в последнем периоде истории")
    for name, records in [("periods=2", test_p2), ("periods=3", test_p3)]:
        mods = Counter()
        for r in records:
            history = r.get("history") or []
            if not history:
                continue
            last = history[-1]
            for c in last.get("choices", []):
                mods[c["module_id"]] += 1
        print(f"  {name}: {len(mods)} модулей; топ-5:")
        for mid, cnt in mods.most_common(5):
            print(f"      {mid}: {cnt}")

    # --- 8. Совпадает ли последний период history с available_modules ---
    print(f"\n[8] Пересечение последнего периода с available_modules")
    for name, records in [("periods=2", test_p2), ("periods=3", test_p3)]:
        inter = []
        for r in records:
            history = r.get("history") or []
            if not history:
                continue
            last_mods = {c["module_id"] for c in history[-1].get("choices", [])}
            avail = set(r.get("available_modules") or [])
            inter.append(len(last_mods & avail))
        print(f"  {name}: mean={np.mean(inter):.2f}  "
              f"min={min(inter)}  max={max(inter)}")


if __name__ == "__main__":
    main()