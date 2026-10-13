"""
Аудит пайплайна: проверка базовых модулей и pipeline-логики.

ВОПРОСЫ, на которые отвечает:
    1. Распределение числа кандидатов в train-срезах (p0/p1/pmask/p2) vs
       val/test (available_modules).
    2. Все ли target-модули содержатся в candidates (sanity check).
    3. Уникальные модули в train-кандидатах vs в available val/test.
    4. Jaccard между train-кандидатами и val/test available.
    5. n_periods в test ДО и ПОСЛЕ обрезки max_history_periods=2.
    6. pop_by_period: сколько модулей имеют ненулевые значения в [1], [2], [3].
    7. Реальный состав test: cold / P1 / P2 / P3.
    8. Содержатся ли test-target-модули в train-истории (контроль "угадываемости").
    9. Распределение приоритетов и модулей в val_solution.
   10. Смещение target-модулей: доля val_solution-модулей, встречающихся
       в train-популярности vs только в val_solution.

Запуск:
    python -m src.diag_pipeline_audit
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_all
from src.features import (
    FeatureStats,
    build_dataset,
    candidates_available,
    make_candidates_train_period1,
    make_candidates_train_period2,
    target_period_1,
    target_period_2,
)
from src.metric import parse_ground_truth

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _student_id(s: dict) -> str | None:
    return s.get("id") or s.get("student_id")


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def _print_hist(name: str, arr: list[int], bins: int = 10) -> None:
    a = np.array(arr, dtype=np.int64)
    if len(a) == 0:
        print(f"  {name}: пусто")
        return
    uniq = sorted(set(a.tolist()))
    print(f"  {name}: n={len(a)}, min={a.min()}, max={a.max()}, "
          f"mean={a.mean():.2f}, median={np.median(a):.1f}, "
          f"уникальных значений: {len(uniq)}")
    if len(uniq) <= bins:
        cnt = Counter(a.tolist())
        print(f"    распределение: {dict(sorted(cnt.items()))}")


def main() -> None:
    print("=" * 78)
    print("АУДИТ ПАЙПЛАЙНА")
    print("=" * 78)

    data = load_all()
    train = data["train"]
    val = data["val"]
    test = data["test"]
    modules = data["modules"]
    val_solution = data["val_solution"]

    # ==== 1. Модули и каталог ====
    print("\n[1/10] Каталог модулей")
    all_module_ids = {m["module_id"] for m in modules}
    print(f"  Всего модулей в каталоге: {len(all_module_ids)}")
    periods_per_module = {
        m["module_id"]: set(m.get("available_in_periods") or [])
        for m in modules
    }
    n_p1_only = sum(1 for s in periods_per_module.values() if s == {1})
    n_p2_only = sum(1 for s in periods_per_module.values() if s == {2})
    n_p12 = sum(1 for s in periods_per_module.values() if s == {1, 2})
    n_other = len(periods_per_module) - n_p1_only - n_p2_only - n_p12
    print(f"  Модули только в P1: {n_p1_only}")
    print(f"  Модули только в P2: {n_p2_only}")
    print(f"  Модули в P1 и P2:   {n_p12}")
    print(f"  Прочие:             {n_other}")

    # ==== 2. Train: срезы и их кандидаты ====
    print("\n[2/10] Train-срезы и их кандидаты")
    stats = FeatureStats.fit(train, modules)
    stats.add_val_solution_popularity(val_solution)

    train_students_p1 = [
        s for s in train
        if any(h.get("period") == 2 for h in s.get("history", []))
    ]
    train_students_p0 = [
        s for s in train if any(h.get("period") == 1 for h in s.get("history", []))
    ]
    cand_p2 = make_candidates_train_period2(stats)(train_students_p1[0]) if train_students_p1 else []
    cand_p1 = make_candidates_train_period1(stats)(train_students_p0[0]) if train_students_p0 else []
    print(f"  p1-срез: pool модулей P2 = {len(cand_p2)}")
    print(f"  p0-срез: pool модулей P1 = {len(cand_p1)}")

    # Что реально попадает в target на p1-срезе?
    # (для train_students_p1: target_period_2)
    target_mods_p2 = set()
    for s in train_students_p1[:500]:
        target_mods_p2.update(target_period_2(s).keys())
    target_mods_p1 = set()
    for s in train_students_p0[:500]:
        target_mods_p1.update(target_period_1(s).keys())
    print(f"  p1-срез (target=P2): target-модулей у 500 студентов = {len(target_mods_p2)}")
    print(f"  p0-срез (target=P1): target-модулей у 500 студентов = {len(target_mods_p1)}")
    print(f"  Все target P2 в pool P2: "
          f"{target_mods_p2.issubset(set(cand_p2))}")
    print(f"  Все target P1 в pool P1: "
          f"{target_mods_p1.issubset(set(cand_p1))}")

    # ==== 3. Val / test: available_modules ====
    print("\n[3/10] Val/test available_modules")
    n_avail_val = [len(s.get("available_modules") or []) for s in val]
    n_avail_test = [len(s.get("available_modules") or []) for s in test]
    _print_hist("val  available_modules", n_avail_val)
    _print_hist("test available_modules", n_avail_test)

    u_val = set()
    for s in val:
        u_val.update(s.get("available_modules") or [])
    u_test = set()
    for s in test:
        u_test.update(s.get("available_modules") or [])
    print(f"  Уникальных модулей в available val:  {len(u_val)}")
    print(f"  Уникальных модулей в available test: {len(u_test)}")

    # ==== 4. Jaccard: train pool vs val/test available ====
    print("\n[4/10] Jaccard: train-пулы vs val/test available")
    print(f"  Jaccard(pool P2, val_available):  {_jaccard(set(cand_p2), u_val):.4f}")
    print(f"  Jaccard(pool P2, test_available): {_jaccard(set(cand_p2), u_test):.4f}")
    print(f"  Jaccard(pool P1, val_available):  {_jaccard(set(cand_p1), u_val):.4f}")
    print(f"  Jaccard(pool P1, test_available): {_jaccard(set(cand_p1), u_test):.4f}")
    print(f"  Jaccard(val_available, test_available): "
          f"{_jaccard(u_val, u_test):.4f}")

    # ==== 5. n_periods в test до/после обрезки ====
    print("\n[5/10] n_periods в test: до и после max_history_periods=2")

    def _n_periods_unique(s: dict) -> int:
        return len({h.get("period") for h in (s.get("history") or [])
                    if h.get("period") is not None})

    before_test = Counter(_n_periods_unique(s) for s in test)
    print(f"  ДО обрезки:    {dict(sorted(before_test.items()))}")
    n_periods_counts = Counter()
    for s in test:
        h = s.get("history") or []
        periods = sorted({x.get("period") for x in h if x.get("period") is not None})
        n_periods_counts[len(periods)] += 1
    print(f"  n_periods ≤2:  {sum(v for k, v in n_periods_counts.items() if k <= 2)}")
    print(f"  n_periods =3:  {n_periods_counts.get(3, 0)}")
    print(f"  n_periods >3:  {sum(v for k, v in n_periods_counts.items() if k > 3)}")

    # ==== 6. pop_by_period ====
    print("\n[6/10] pop_by_period: покрытие модулей")
    for p in sorted(stats.pop_by_period.keys()):
        d = stats.pop_by_period[p]
        n_nonzero = sum(1 for v in d.values() if v > 0)
        print(f"  period={p}: ненулевых модулей = {n_nonzero}")

    # ==== 7. Состав test по типам ====
    print("\n[7/10] Состав test: cold / P1 / P2 / P3")

    def _n_choices(s: dict) -> int:
        return sum(len(h.get("choices", [])) for h in (s.get("history") or []))

    def _is_cold(s: dict) -> bool:
        prof = s.get("profile") or {}
        return (_n_choices(s) == 0
                and not prof.get("institute")
                and not prof.get("specialty"))

    n_cold = sum(1 for s in test if _is_cold(s))
    cnt = Counter()
    for s in test:
        if _is_cold(s):
            continue
        n = _n_periods_unique(s)
        cnt[n] += 1
    print(f"  cold (P0):  {n_cold} ({100 * n_cold / len(test):.1f}%)")
    for p in sorted(cnt):
        print(f"  warm P{p}:   {cnt[p]} ({100 * cnt[p] / len(test):.1f}%)")

    # Аналогично для val
    n_cold_v = sum(1 for s in val if _is_cold(s))
    cnt_v = Counter()
    for s in val:
        if _is_cold(s):
            continue
        cnt_v[_n_periods_unique(s)] += 1
    print(f"  --- VAL для сравнения ---")
    print(f"  cold (P0):  {n_cold_v} ({100 * n_cold_v / len(val):.1f}%)")
    for p in sorted(cnt_v):
        print(f"  warm P{p}:   {cnt_v[p]} ({100 * cnt_v[p] / len(val):.1f}%)")

    # ==== 8. Test-target ∈ train-история ====
    print("\n[8/10] Содержатся ли target-модули в train-истории")
    train_history_modules: set[str] = set()
    for s in train:
        for h in s.get("history", []):
            for c in h.get("choices", []):
                train_history_modules.add(c["module_id"])
    print(f"  Уникальных модулей в train-истории: {len(train_history_modules)}")

    val_gt = dict(zip(val_solution["id"], val_solution["ground_truth"]))
    n_val_gt_in_train_hist = 0
    n_val_gt_total = 0
    for raw in val_solution["ground_truth"].astype(str):
        gt = parse_ground_truth(raw)
        for m in gt:
            n_val_gt_total += 1
            if m in train_history_modules:
                n_val_gt_in_train_hist += 1
    if n_val_gt_total:
        print(f"  val target-модулей в train-истории: "
              f"{n_val_gt_in_train_hist}/{n_val_gt_total} "
              f"({100 * n_val_gt_in_train_hist / n_val_gt_total:.1f}%)")

    # Есть ли у test студентов те же модули, что в train-истории?
    test_avail_mods = set()
    for s in test:
        test_avail_mods.update(s.get("available_modules") or [])
    print(f"  test available ∩ train-история: "
          f"{len(test_avail_mods & train_history_modules)}/{len(test_avail_mods)} "
          f"({100 * len(test_avail_mods & train_history_modules) / max(1, len(test_avail_mods)):.1f}%)")

    # ==== 9. val_solution: распределение приоритетов и модулей ====
    print("\n[9/10] val_solution: приоритеты и модули")
    priorities = []
    gt_modules = Counter()
    for raw in val_solution["ground_truth"].astype(str):
        gt = parse_ground_truth(raw)
        for m, p in gt.items():
            priorities.append(p)
            gt_modules[m] += 1
    print(f"  Всего (student, module) пар: {len(priorities)}")
    print(f"  Распределение приоритетов: "
          f"{dict(sorted(Counter(priorities).items()))}")
    print(f"  Уникальных модулей в val GT: {len(gt_modules)}")
    print(f"  Топ-5 по частоте в val GT:")
    for m, c in gt_modules.most_common(5):
        print(f"    {m}: {c}")

    # Пересечение val GT-модулей с train-историей
    in_train = sum(1 for m in gt_modules if m in train_history_modules)
    print(f"  val GT-модулей, встречающихся в train-истории: "
          f"{in_train}/{len(gt_modules)}")

    # ==== 10. Проверка build_dataset: консистентность для train/val/test ====
    print("\n[10/10] build_dataset: консистентность префиксов и n_periods")
    # Строим маленькие подвыборки и смотрим n_periods после обрезки
    sample_train_p1 = train_students_p1[:5]
    sample_val = val[:5]
    sample_test_p3 = [s for s in test if _n_periods_unique(s) >= 3][:5]

    df_tr = build_dataset(
        sample_train_p1, stats,
        candidates_fn=make_candidates_train_period2(stats),
        target_fn=target_period_2, past_periods=[1],
    )
    print(f"  train p1-срез (5 студентов): строк={len(df_tr)}, "
          f"n_periods уникальные={sorted(df_tr['n_periods'].unique())}")

    df_val = build_dataset(
        sample_val, stats,
        candidates_fn=candidates_available,
        target_fn=None, past_periods=[1, 2],
    )
    print(f"  val (5 студентов): строк={len(df_val)}, "
          f"n_periods уникальные={sorted(df_val['n_periods'].unique())}")

    df_test = build_dataset(
        sample_test_p3, stats,
        candidates_fn=candidates_available,
        target_fn=None, past_periods=None, max_history_periods=2,
    )
    if len(df_test) > 0:
        print(f"  test P3 (5 студентов): строк={len(df_test)}, "
              f"n_periods уникальные={sorted(df_test['n_periods'].unique())}")
        # Проверим, попадает ли в df_test модуль из target P3
        for s in sample_test_p3[:2]:
            sid = s["id"]
            sub = df_test[df_test["student_id"] == sid]
            print(f"    студент {sid}: строк={len(sub)}, "
                  f"n_periods={sub['n_periods'].iloc[0] if len(sub) > 0 else 'NA'}")

    print("\n" + "=" * 78)
    print("ВЫВОДЫ:")
    print("=" * 78)
    print("""
    Пункт 4 (Jaccard):
        Если Jaccard(pool_train, val/test available) < 0.9 — train и val/test
        имеют разные наборы кандидатов. Признаки pop_rank_in_available,
        pop_rank_frac, pop_ratio_in_institute могут давать сдвиг на инференсе.

    Пункт 7 (состав test):
        Если warm P1/P2 доминируют, а cold <30% — наша модель должна хорошо
        работать на warm. Если наоборот — нужна отдельная логика cold.

    Пункт 8 (target ∈ train history):
        Если высокая доля val GT в train-истории, модель может "угадывать"
        через уже виденные модули. Это не баг, это сильный сигнал.

    Пункт 10 (n_periods после обрезки):
        Если для test P3 после max_history_periods=2 значение n_periods = 2
        (то же, что для val P2) — модель НЕ различает эти два случая.
        Это структурное ограничение фиксируется add_correct_period_features,
        но только для pop_*; для остальных фич оно остаётся.
""")


if __name__ == "__main__":
    main()