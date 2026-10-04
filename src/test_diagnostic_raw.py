"""
Детальная диагностика test-студентов по группам.

Цель:
    1. Показать сырые (id, history, val_solution, available_modules)
       для 2-3 студентов каждой группы (periods=0,1,2,3).
    2. Для групп periods=2 и periods=3 — точно определить, какие
       периоды в test.history, и соответствуют ли они периодам val.history.
    3. Проверить гипотезу: для periods=2 target = val_solution (period 3).
       Для periods=3: target = period 4, а period 3 уже в истории.

Запуск:
    python -m src.test_diagnostic_raw
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.data_loader import load_all


def get_id(r: dict[str, Any]) -> str:
    return r.get("id") or r.get("student_id")


def periods_in_history(r: dict[str, Any]) -> list[int]:
    return [h.get("period") for h in r.get("history") or []]


def fmt_history(r: dict[str, Any]) -> str:
    out = []
    for h in r.get("history") or []:
        mods = ", ".join(c["module_id"] for c in h.get("choices", []))
        out.append(f"P{h['period']}=[{mods}]")
    return " | ".join(out) if out else "<empty>"


def fmt_solution(gt_str: str) -> str:
    if not gt_str:
        return "<none>"
    pairs = [tok for tok in str(gt_str).split() if ":" in tok]
    return ", ".join(pairs)


def main() -> None:
    data = load_all()
    test = data["test"]
    val = data["val"]
    val_sol = data["val_solution"]

    val_by_id = {get_id(v): v for v in val}
    sol_by_id = dict(zip(val_sol["id"].astype(str), val_sol["ground_truth"]))

    print("=" * 100)
    print("ДЕТАЛЬНАЯ ДИАГНОСТИКА: сырые данные по группам")
    print("=" * 100)

    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for r in test:
        groups[len(periods_in_history(r))].append(r)

    for g in sorted(groups):
        print(f"\n{'=' * 100}")
        print(f"ГРУППА periods={g}   (n={len(groups[g])})")
        print(f"{'=' * 100}")

        # Периоды, встречающиеся в history этой группы
        period_counter = Counter()
        for r in groups[g]:
            for p in periods_in_history(r):
                period_counter[p] += 1
        print(f"  Какие периоды встречаются в history: "
              f"{dict(sorted(period_counter.items()))}")

        # Показать 2 примера
        for r in groups[g][:2]:
            sid = get_id(r)
            print(f"\n  --- {sid} ---")
            print(f"    test.history:         {fmt_history(r)}")
            print(f"    available_modules:    {len(r.get('available_modules') or [])} шт.")
            v_rec = val_by_id.get(sid)
            if v_rec is not None:
                print(f"    val.history:          {fmt_history(v_rec)}")
            else:
                print(f"    val.history:          <нет в val>")
            gt = sol_by_id.get(sid)
            if gt is not None:
                print(f"    val_solution:         {fmt_solution(gt)}")
            else:
                print(f"    val_solution:         <нет>")

    # --- Проверка гипотезы: для periods=2 → target=3, известен ли он? ---
    print(f"\n{'=' * 100}")
    print("ПРОВЕРКА ГИПОТЕЗЫ: для periods=2 target = val_solution?")
    print(f"{'=' * 100}")

    # Смотрим: если test.history — это [1, 2], а val.history — тоже [1, 2],
    # и при этом history_signature не совпадает — возможно, порядок модулей другой.
    # Проверим количество совпадающих модулей между test.history и val.history.
    n_checked = 0
    n_same_periods = 0
    n_same_modules_set = 0

    for r in groups.get(2, []):
        sid = get_id(r)
        v_rec = val_by_id.get(sid)
        if v_rec is None:
            continue
        n_checked += 1
        tp = sorted(periods_in_history(r))
        vp = sorted(periods_in_history(v_rec))
        if tp == vp:
            n_same_periods += 1

        # Множество модулей в test.history vs val.history
        t_mods = {c["module_id"] for h in r.get("history") or [] for c in h.get("choices", [])}
        v_mods = {c["module_id"] for h in v_rec.get("history") or [] for c in h.get("choices", [])}
        if t_mods == v_mods:
            n_same_modules_set += 1

    print(f"  Проверено periods=2 студентов с парой в val: {n_checked}")
    print(f"    Совпадают периоды:       {n_same_periods}")
    print(f"    Совпадают множества модулей: {n_same_modules_set}")

    # --- Для periods=3: какие периоды в test.history? ---
    print(f"\n{'=' * 100}")
    print("ГРУППА periods=3: распределение периодов в test.history")
    print(f"{'=' * 100}")
    tuple_counter = Counter()
    for r in groups.get(3, []):
        tup = tuple(sorted(periods_in_history(r)))
        tuple_counter[tup] += 1
    for tup, cnt in tuple_counter.most_common():
        print(f"  periods={tup}: {cnt} студентов")

    # --- Для periods=2: какие периоды в test.history? ---
    print(f"\n{'=' * 100}")
    print("ГРУППА periods=2: распределение периодов в test.history")
    print(f"{'=' * 100}")
    tuple_counter = Counter()
    for r in groups.get(2, []):
        tup = tuple(sorted(periods_in_history(r)))
        tuple_counter[tup] += 1
    for tup, cnt in tuple_counter.most_common():
        print(f"  periods={tup}: {cnt} студентов")

    # --- val: какие периоды в val.history? ---
    print(f"\n{'=' * 100}")
    print("val: распределение периодов в val.history")
    print(f"{'=' * 100}")
    tuple_counter = Counter()
    for v in val:
        tup = tuple(sorted(periods_in_history(v)))
        tuple_counter[tup] += 1
    for tup, cnt in tuple_counter.most_common():
        print(f"  periods={tup}: {cnt} студентов")


if __name__ == "__main__":
    main()