"""
Диагностика: сравнение распределений val и test.

Цель — понять, почему модели, обученные и оценённые на val,
проваливаются на public. Наиболее вероятные причины:
    - Разные размеры available_modules.
    - Разный состав модулей в available_modules.
    - Разное распределение популярности модулей.
    - Разные профили студентов (institute/specialty).

Всё это напрямую влияет на признаки pop_rank_in_available,
pop_rank_global_in_available и производные.

Модуль НЕ трогает модели и НЕ строит submission — только печатает
распределения. Результат должен дать чёткий ответ: отличается ли
test от val фундаментально, или отличие только в долях сегментов.

Запуск:
    python -m src.diag_val_vs_test
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from src.cold_start import is_cold
from src.data_loader import load_all
from src.popularity import compute_popularity


def _stats(name: str, values: list[int]) -> None:
    arr = np.array(values, dtype=np.float64)
    if len(arr) == 0:
        print(f"  {name}: пусто")
        return
    print(
        f"  {name}: n={len(arr):>5}, "
        f"mean={arr.mean():6.2f}, median={np.median(arr):5.1f}, "
        f"min={int(arr.min()):>3}, max={int(arr.max()):>4}, "
        f"p10={np.percentile(arr, 10):5.1f}, p90={np.percentile(arr, 90):5.1f}"
    )


def _union_modules(students: list[dict]) -> set[str]:
    out: set[str] = set()
    for s in students:
        out.update(s.get("available_modules") or [])
    return out


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _institute_distribution(students: list[dict], label: str) -> None:
    cnt = Counter()
    for s in students:
        prof = s.get("profile") or {}
        inst = prof.get("institute") or "UNKNOWN"
        cnt[inst] += 1
    if not cnt:
        print(f"  {label}: пусто")
        return
    total = sum(cnt.values())
    print(f"  {label}: {len(cnt)} уникальных институтов, total={total}")
    for inst, c in cnt.most_common(10):
        print(f"    {c:>5} ({100 * c / total:>4.1f}%)  {inst}")
    if len(cnt) > 10:
        print(f"    ... и ещё {len(cnt) - 10} институтов")


def main() -> None:
    print("=" * 78)
    print("Диагностика: val vs test")
    print("=" * 78)

    data = load_all()
    val = data["val"]
    test = data["test"]
    train = data["train"]

    val_cold = [s for s in val if is_cold(s)]
    val_warm = [s for s in val if not is_cold(s)]
    test_cold = [s for s in test if is_cold(s)]
    test_warm = [s for s in test if not is_cold(s)]

    print(f"\nРазмеры выборок:")
    print(f"  val:  {len(val):>5} (cold={len(val_cold)}, warm={len(val_warm)})")
    print(f"  test: {len(test):>5} (cold={len(test_cold)}, warm={len(test_warm)})")

    # === 1. Размеры available_modules ===
    print("\n[1/6] Размеры available_modules:")
    _stats("val  all ", [len(s.get("available_modules") or []) for s in val])
    _stats("val  cold", [len(s.get("available_modules") or []) for s in val_cold])
    _stats("val  warm", [len(s.get("available_modules") or []) for s in val_warm])
    _stats("test all ", [len(s.get("available_modules") or []) for s in test])
    _stats("test cold", [len(s.get("available_modules") or []) for s in test_cold])
    _stats("test warm", [len(s.get("available_modules") or []) for s in test_warm])

    # === 2. Уникальные модули в available ===
    print("\n[2/6] Уникальные модули в available_modules:")
    u_val = _union_modules(val)
    u_test = _union_modules(test)
    u_val_cold = _union_modules(val_cold)
    u_test_cold = _union_modules(test_cold)
    u_val_warm = _union_modules(val_warm)
    u_test_warm = _union_modules(test_warm)

    print(f"  val  all  : |U|={len(u_val)}")
    print(f"  test all  : |U|={len(u_test)}")
    print(f"  val  cold : |U|={len(u_val_cold)}")
    print(f"  test cold : |U|={len(u_test_cold)}")
    print(f"  val  warm : |U|={len(u_val_warm)}")
    print(f"  test warm : |U|={len(u_test_warm)}")

    # === 3. Jaccard ===
    print("\n[3/6] Jaccard между объединениями:")
    print(f"  val_all  ∩ test_all : {_jaccard(u_val, u_test):.4f}")
    print(f"  val_cold ∩ test_cold: {_jaccard(u_val_cold, u_test_cold):.4f}")
    print(f"  val_warm ∩ test_warm: {_jaccard(u_val_warm, u_test_warm):.4f}")

    # === 4. Эксклюзивные модули ===
    print("\n[4/6] Эксклюзивные модули (all):")
    val_only = u_val - u_test
    test_only = u_test - u_val
    both = u_val & u_test
    print(f"  только в val : {len(val_only)}")
    print(f"  только в test: {len(test_only)}")
    print(f"  в обоих      : {len(both)}")
    if val_only:
        sample = sorted(val_only)[:10]
        print(f"  примеры val_only : {sample}")
    if test_only:
        sample = sorted(test_only)[:10]
        print(f"  примеры test_only: {sample}")

    # === 5. Распределение популярности ===
    print("\n[5/6] Распределение pop_weighted_global из train:")
    pop_weighted, _ = compute_popularity(train)

    def _pop_stats(name: str, module_set: set) -> None:
        if not module_set:
            print(f"  {name}: пусто")
            return
        vals = [pop_weighted.get(m, 0.0) for m in module_set]
        arr = np.array(vals, dtype=np.float64)
        print(
            f"  {name:<12}: n={len(arr):>4}, "
            f"mean={arr.mean():7.3f}, median={np.median(arr):7.3f}, "
            f"zero_count={(arr == 0).sum():>4} ({100 * (arr == 0).sum() / len(arr):>5.1f}%)"
        )

    _pop_stats("все модули", set(pop_weighted))
    _pop_stats("val_all", u_val)
    _pop_stats("test_all", u_test)
    _pop_stats("val_only", val_only)
    _pop_stats("test_only", test_only)
    _pop_stats("both", both)

    # === 6. Институты: val vs test ===
    print("\n[6/6] Распределение институтов:")
    _institute_distribution(val, "val")
    print()
    _institute_distribution(test, "test")

    # === 7. Распределение приоритетов в val_solution ===
    val_sol = data["val_solution"]
    print("\n[7/7] Распределение приоритетов в val_solution:")
    priorities = []
    for raw in val_sol["ground_truth"].astype(str):
        raw = raw.strip()
        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1]
        for tok in raw.split():
            if ":" in tok:
                _, p = tok.split(":", 1)
                try:
                    priorities.append(int(p))
                except ValueError:
                    pass
    cnt = Counter(priorities)
    total = sum(cnt.values())
    print(f"  n={total}")
    for p in sorted(cnt):
        print(f"    priority={p}: {cnt[p]:>6} ({100 * cnt[p] / total:>5.1f}%)")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()