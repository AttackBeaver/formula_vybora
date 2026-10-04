"""
Диагностика структуры test: разные ли целевые периоды у разных групп.

Гипотеза:
    test содержит 4 группы студентов (по числу периодов истории):
        - 0 периодов → target = период 1 (cold)
        - 1 период   → target = период 2
        - 2 периода  → target = период 3
        - 3 периода  → target = период 4
    Мы обучаем LTR только на переходах в 1-й и 2-й периоды, поэтому для
    62.7% test-студентов модель решает не ту задачу.

Что выясняем:
    1. Распределение test по числу периодов и сигнатурам available_modules.
    2. Пересечение test с train и val по каждой группе.
    3. Для test-студентов с 2 периодами: восстанавливается ли их target
       из val_solution (т.е. их 3-й период известен).
    4. Для test-студентов с 3 периодами: одинаковое ли у них
       available_modules с другими группами и есть ли что-то в train,
       похожее на их целевую задачу.

Запуск:
    python -m src.test_diagnostic
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.data_loader import load_all


# ---------------------------------------------------------------------------
# Хелперы
# ---------------------------------------------------------------------------
def n_periods(record: dict[str, Any]) -> int:
    return len(record.get("history") or [])


def get_id(record: dict[str, Any]) -> str | None:
    return record.get("id") or record.get("student_id")


def n_choices(record: dict[str, Any]) -> int:
    return sum(len(h.get("choices", [])) for h in record.get("history") or [])


def avail_signature(record: dict[str, Any]) -> frozenset[str]:
    return frozenset(record.get("available_modules") or [])


def history_signature(record: dict[str, Any]) -> tuple[tuple[int, tuple[str, ...]], ...]:
    """
    Возвращает хешируемое представление истории:
    ((period, (module_id, ...)), ...) — без приоритетов.
    """
    hist = []
    for h in record.get("history") or []:
        mods = tuple(sorted(c["module_id"] for c in h.get("choices", [])))
        hist.append((h.get("period"), mods))
    return tuple(hist)


def target_period_guess(n_p: int) -> int:
    """Гипотеза: target = n_periods + 1."""
    return n_p + 1


# ---------------------------------------------------------------------------
# Основная логика
# ---------------------------------------------------------------------------
def main() -> None:
    data = load_all()
    train = data["train"]
    val = data["val"]
    test = data["test"]
    val_sol = data["val_solution"]

    print("=" * 78)
    print("ДИАГНОСТИКА TEST: разные ли целевые периоды у групп")
    print("=" * 78)

    # --- 1. Распределение test по числу периодов ---
    print("\n[1] test: распределение по числу периодов истории")
    cnt = Counter(n_periods(r) for r in test)
    for k in sorted(cnt):
        share = 100 * cnt[k] / len(test)
        print(f"    periods={k}: {cnt[k]:>5}  ({share:5.1f}%)  "
              f"→ предполагаемый target = {target_period_guess(k)}")

    # --- 2. Сигнатура available_modules по группам ---
    print("\n[2] test: сигнатуры available_modules по группам")
    by_group_sig: dict[int, Counter] = defaultdict(Counter)
    for r in test:
        by_group_sig[n_periods(r)][avail_signature(r)] += 1

    for k in sorted(by_group_sig):
        sigs = by_group_sig[k]
        sizes = [len(s) for s in sigs]
        print(f"    periods={k}: {len(sigs)} уник. сигнатур, "
              f"размеры={sorted(set(sizes))}, "
              f"топ сигнатура = {sigs.most_common(1)[0][1]} студентов")

    # --- 3. Пересечение test с train / val по группам ---
    print("\n[3] Пересечение test с train / val")
    train_ids = {get_id(s) for s in train}
    val_ids = {get_id(s) for s in val}
    val_sol_ids = set(val_sol["id"].astype(str))

    per_group_stats: dict[int, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "in_train": 0, "in_val": 0, "in_val_sol": 0}
    )
    for r in test:
        g = n_periods(r)
        sid = get_id(r)
        per_group_stats[g]["n"] += 1
        if sid in train_ids:
            per_group_stats[g]["in_train"] += 1
        if sid in val_ids:
            per_group_stats[g]["in_val"] += 1
        if sid in val_sol_ids:
            per_group_stats[g]["in_val_sol"] += 1

    for g in sorted(per_group_stats):
        s = per_group_stats[g]
        print(f"    periods={g}: n={s['n']:>5}  "
              f"in_train={s['in_train']:>5}  "
              f"in_val={s['in_val']:>5}  "
              f"in_val_sol={s['in_val_sol']:>5}")

    # --- 4. Группа с 2 периодами: target=3 известен из val_solution ---
    print("\n[4] test periods=2: target=3 восстановим из val_solution?")
    n_with_target = 0
    n_match_history = 0
    for r in test:
        if n_periods(r) != 2:
            continue
        sid = get_id(r)
        if sid in val_sol_ids:
            n_with_target += 1
            # Проверим, что история test совпадает с историей val
            val_rec = next((v for v in val if get_id(v) == sid), None)
            if val_rec is not None:
                if history_signature(val_rec) == history_signature(r):
                    n_match_history += 1
    print(f"    С target в val_solution:      {n_with_target}")
    print(f"    История совпадает с val:      {n_match_history}")

    # --- 5. Группа с 3 периодами: target=4 ---
    print("\n[5] test periods=3: target=4")
    print("    В train и val максимальное число периодов = 2.")
    print("    Значит, напрямую обучить модель на (past=[1,2,3] → target=4) невозможно.")
    # Но: возможно, эти студенты в train фигурируют с 1-2 периодами.
    n_in_train = 0
    train_lookup = {get_id(s): s for s in train}
    for r in test:
        if n_periods(r) != 3:
            continue
        sid = get_id(r)
        t_rec = train_lookup.get(sid)
        if t_rec is not None:
            n_in_train += 1
    print(f"    Есть в train:                  {n_in_train}")
    print(f"    Их train-история — это первые 1-2 периода.")

    # --- 6. Есть ли в val студенты с 3 периодами (для обучения на target=4)? ---
    print("\n[6] val: распределение по периодам")
    val_periods = Counter(n_periods(r) for r in val)
    for k in sorted(val_periods):
        print(f"    periods={k}: {val_periods[k]:>5}")
    print("    (val_solution даёт target = следующий период после истории.)")

    # --- 7. Train: сколько студентов имеют 2 периода ---
    print("\n[7] train: распределение по периодам")
    train_periods = Counter(n_periods(r) for r in train)
    for k in sorted(train_periods):
        print(f"    periods={k}: {train_periods[k]:>5}")

    # --- 8. Что содержит val_solution: точно ли это период 3? ---
    print("\n[8] val_solution: какой период это даёт")
    # Пересечём val и val_solution: у val max 2 периода, у val_sol — ответ в 3-м.
    # Проверим, что модули из val_sol НЕ пересекаются с историей val (защита от утечки)
    overlap_counts = []
    for r in val:
        sid = get_id(r)
        gt_row = val_sol[val_sol["id"] == sid]
        if gt_row.empty:
            continue
        gt_str = gt_row["ground_truth"].iloc[0]
        gt_mods = {
            tok.split(":")[0] for tok in str(gt_str).split() if ":" in tok
        }
        hist_mods = {
            c["module_id"]
            for h in r.get("history") or []
            for c in h.get("choices", [])
        }
        overlap_counts.append(len(gt_mods & hist_mods))
    import numpy as np
    if overlap_counts:
        print(f"    Пересечение val_solution с историей val:")
        print(f"      mean = {np.mean(overlap_counts):.3f}")
        print(f"      max  = {max(overlap_counts)}")
        print(f"      нулей: {sum(1 for x in overlap_counts if x == 0)} / {len(overlap_counts)}")
    print("    (Если overlap > 0 — student мог выбрать один и тот же модуль повторно.)")

    print("\n" + "=" * 78)
    print("ВЫВОД")
    print("=" * 78)
    print("""
    Если в test 62.7% студентов имеют 2+ периодов истории, и val_solution
    покрывает часть из них, то ключевой шаг — добавить в обучение срез
    (past=[1,2] → target=3) на данных val. Это даст модели опыт работы
    с 3-м периодом, которого сейчас нет.

    Для группы test с 3 периодами истории (target=4) прямого сигнала
    нет, но можно попробовать:
      - использовать более общие признаки (cooc, popularity по институту)
      - дообучиться на похожих студентах (тех, у кого 2 периода → target=3)
      - обучить отдельную модель под целевую задачу "период 4" через
        псевдо-target: следующий выбор в test мы не знаем, но можно
        воспользоваться "следующим модулем, который студент выбирал
        в каком-то другом периоде".
    """)


if __name__ == "__main__":
    main()