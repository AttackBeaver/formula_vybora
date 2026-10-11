"""
Диагностика: доля повторов в target vs история студента.

ВОПРОС, который решает v18:
    972 already_chosen в top-5 у 284 holdout студентов (3.4 на студента).
    Модель учится рекомендовать студенту его же прошлые выборы,
    и это РАБОТАЕТ на public (0.3348 у v7).

    Значит, студенты часто выбирают те же модули повторно?
    Или val-target содержит history по какой-то причине?

Что считаем:
    A. Что доля GT-модулей, которые ЕСТЬ в истории студента?
       Отдельно для cold / warm / periods=1/2/3.
    B. Что доля history-модулей, которые попали в GT?
    C. pNDCG@5 для простых стратегий на val:
       - global_pop (top-5 по глобальной популярности)
       - history_back (top-5 из истории, отсортированные по priority)
       - history_back + global_pop fallback
       - model v7 (если есть submission)
    D. Разбивка по типам студентов (cold/warm/periods).

Запуск:
    python -m src.v19_diag_repeats
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.cold_start import is_cold
from src.data_loader import load_all
from src.metric import evaluate_pndcg5, parse_ground_truth
from src.popularity import compute_popularity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
WEIGHTS = {1: 1.00, 2: 0.80, 3: 0.60, 4: 0.45, 5: 0.35}


def _history_priority_map(student: dict) -> dict[str, int]:
    """
    Для каждого модуля истории — лучший (наименьший) priority.
    Если модуль выбирался в P1 с prio 3 и P2 с prio 1 → вернём 1.
    """
    best: dict[str, int] = {}
    for h in student.get("history") or []:
        for c in h.get("choices") or []:
            mid = c["module_id"]
            p = int(c.get("priority", 5))
            if mid not in best or p < best[mid]:
                best[mid] = p
    return best


def _history_modules_ordered(student: dict) -> list[str]:
    """Уникальные модули истории, отсортированные по (priority asc, module_id)."""
    prio_map = _history_priority_map(student)
    items = sorted(prio_map.items(), key=lambda x: (x[1], x[0]))
    return [m for m, _ in items]


def _fill_to_k(primary: list[str], fallback: list[str], k: int = 5) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for m in primary:
        if m in seen:
            continue
        result.append(m)
        seen.add(m)
        if len(result) == k:
            return result
    for m in fallback:
        if m in seen:
            continue
        result.append(m)
        seen.add(m)
        if len(result) == k:
            return result
    return result


def _topk_global_pop(
    available: list[str],
    pop_weighted: dict[str, float],
    k: int = 5,
) -> list[str]:
    scored = [(m, pop_weighted.get(m, 0.0)) for m in available]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return [m for m, _ in scored[:k]]


def _segment_stats(
    name: str,
    students: list[dict],
    gt_raw: dict[str, str],
) -> None:
    """Считает A и B для сегмента студентов."""
    if not students:
        print(f"  [{name}] пусто")
        return

    n_students = len(students)
    total_gt_modules = 0
    total_gt_in_history = 0
    total_history_modules = 0
    total_history_in_gt = 0
    n_with_any_overlap = 0

    for s in students:
        sid = s["id"]
        gt = parse_ground_truth(gt_raw.get(sid, ""))
        hist = _history_priority_map(s)
        if gt:
            total_gt_modules += len(gt)
            in_hist = sum(1 for m in gt if m in hist)
            total_gt_in_history += in_hist
            if in_hist > 0:
                n_with_any_overlap += 1
        total_history_modules += len(hist)
        total_history_in_gt += sum(1 for m in hist if m in gt)

    avg_gt = total_gt_modules / n_students if n_students else 0
    avg_hist = total_history_modules / n_students if n_students else 0
    frac_gt_in_hist = (
        total_gt_in_history / total_gt_modules if total_gt_modules else 0
    )
    frac_hist_in_gt = (
        total_history_in_gt / total_history_modules
        if total_history_modules
        else 0
    )
    pct_with_overlap = 100 * n_with_any_overlap / n_students

    print(f"\n  [{name}] n={n_students}")
    print(f"      avg |GT|            = {avg_gt:.2f}")
    print(f"      avg |history|       = {avg_hist:.2f}")
    print(f"      доля GT в history   = {frac_gt_in_hist:.3f} "
          f"({total_gt_in_history}/{total_gt_modules})")
    print(f"      доля history в GT   = {frac_hist_in_gt:.3f} "
          f"({total_history_in_gt}/{total_history_modules})")
    print(f"      студентов с хотя бы 1 пересечением: "
          f"{n_with_any_overlap}/{n_students} ({pct_with_overlap:.1f}%)")


def _strategy_scores(
    name: str,
    students: list[dict],
    gt_raw: dict[str, str],
    pop_weighted: dict[str, float],
) -> dict[str, float]:
    """
    Оценивает 3 стратегии по pNDCG@5 на данном подмножестве студентов.
    Возвращает {'global_pop': x, 'history_back': y, 'hist_then_pop': z}.
    """
    rec_gp: dict[str, list[str]] = {}
    rec_hb: dict[str, list[str]] = {}
    rec_hp: dict[str, list[str]] = {}

    for s in students:
        sid = s["id"]
        avail = list(s.get("available_modules") or [])
        prio_map = _history_priority_map(s)

        # global pop: top-5 из available по pop_weighted
        gp = _topk_global_pop(avail, pop_weighted, k=5)

        # history back: сначала модули из истории, отсортированные по prio
        # (только те, что в available)
        hist_in_avail = sorted(
            [(m, prio_map[m]) for m in prio_map if m in set(avail)],
            key=lambda x: (x[1], x[0]),
        )
        hb_primary = [m for m, _ in hist_in_avail]
        hb = _fill_to_k(hb_primary, gp, k=5)

        # history back + global pop, но с приоритетом истории
        hp = hb

        rec_gp[sid] = gp
        rec_hb[sid] = hb
        rec_hp[sid] = hp

    # GT подмножество — только для этих студентов
    sids = {s["id"] for s in students}
    gt_sub = {sid: gt for sid, gt in gt_raw.items() if sid in sids}

    m_gp, _ = evaluate_pndcg5(rec_gp, gt_sub, k=5)
    m_hb, _ = evaluate_pndcg5(rec_hb, gt_sub, k=5)
    m_hp, _ = evaluate_pndcg5(rec_hp, gt_sub, k=5)

    print(f"\n  [{name}] (n={len(students)})")
    print(f"      global_pop:    {m_gp:.4f}")
    print(f"      history_back:  {m_hb:.4f}")
    print(f"      hist+pop:      {m_hp:.4f}")

    return {"global_pop": m_gp, "history_back": m_hb, "hist_then_pop": m_hp}


def _load_v7_predictions() -> dict[str, list[str]] | None:
    """Пробуем загрузить v7 submission для сравнения."""
    for name in ["val_submission_ltr_v7.csv", "val_submission_ltr.csv"]:
        path = SUBMISSIONS_DIR / name
        if path.exists():
            df = pd.read_csv(path)
            rec: dict[str, list[str]] = {}
            for _, row in df.iterrows():
                rec[str(row["id"])] = str(row["recommendations"]).split()
            print(f"[v19_diag] Загружен {path.name} для сравнения (n={len(rec)})")
            return rec
    return None


def main() -> None:
    print("=" * 78)
    print("ДИАГНОСТИКА: доля повторов (GT ∩ history) на val")
    print("=" * 78)

    data = load_all()
    val = data["val"]
    train = data["train"]
    val_solution = data["val_solution"]
    gt_raw = dict(zip(val_solution["id"], val_solution["ground_truth"]))

    # ==== 1. Сегментация val ====
    val_cold = [s for s in val if is_cold(s)]
    val_warm = [s for s in val if not is_cold(s)]

    def _n_periods(s: dict) -> int:
        return len([h for h in (s.get("history") or [])
                    if len(h.get("choices", [])) > 0])

    val_p1 = [s for s in val_warm if _n_periods(s) == 1]
    val_p2 = [s for s in val_warm if _n_periods(s) == 2]

    print(f"\n[Размеры]: val={len(val)} "
          f"(cold={len(val_cold)}, warm={len(val_warm)}, "
          f"warm P1={len(val_p1)}, warm P2={len(val_p2)})")

    # ==== 2. Доля повторов по сегментам ====
    print("\n" + "=" * 78)
    print("[A/B] Доля повторов: GT ∩ history")
    print("=" * 78)

    _segment_stats("ALL val", val, gt_raw)
    _segment_stats("cold", val_cold, gt_raw)
    _segment_stats("warm", val_warm, gt_raw)
    _segment_stats("warm, periods=1", val_p1, gt_raw)
    _segment_stats("warm, periods=2", val_p2, gt_raw)

    # ==== 3. Стратегии на val ====
    print("\n" + "=" * 78)
    print("[C] pNDCG@5 для простых стратегий")
    print("=" * 78)

    pop_weighted, _ = compute_popularity(train)

    _strategy_scores("ALL val", val, gt_raw, pop_weighted)
    _strategy_scores("cold", val_cold, gt_raw, pop_weighted)
    _strategy_scores("warm", val_warm, gt_raw, pop_weighted)
    _strategy_scores("warm, periods=1", val_p1, gt_raw, pop_weighted)
    _strategy_scores("warm, periods=2", val_p2, gt_raw, pop_weighted)

    # ==== 4. Сравнение с v7 ====
    print("\n" + "=" * 78)
    print("[D] Сравнение с v7-submission (если есть)")
    print("=" * 78)

    v7_pred = _load_v7_predictions()
    if v7_pred is not None:
        m_v7_all, per_all = evaluate_pndcg5(v7_pred, gt_raw, k=5)
        print(f"\n  v7 val: mean={m_v7_all:.4f}")

        cold_ids = {s["id"] for s in val_cold}
        warm_ids = {s["id"] for s in val_warm}
        p1_ids = {s["id"] for s in val_p1}
        p2_ids = {s["id"] for s in val_p2}

        def _mean_for(ids: set[str]) -> float:
            vals = [s for sid, s in per_all.items() if sid in ids]
            return float(np.mean(vals)) if vals else 0.0

        print(f"  v7 cold: {_mean_for(cold_ids):.4f}")
        print(f"  v7 warm: {_mean_for(warm_ids):.4f}")
        print(f"  v7 warm P1: {_mean_for(p1_ids):.4f}")
        print(f"  v7 warm P2: {_mean_for(p2_ids):.4f}")

        # Сколько модулей v7 в top-5 совпадают с history для warm
        n_hits = 0
        n_students_with = 0
        for s in val_warm:
            sid = s["id"]
            hist = _history_priority_map(s)
            rec = v7_pred.get(sid, [])
            this = sum(1 for m in rec[:5] if m in hist)
            if this > 0:
                n_hits += this
                n_students_with += 1
        if val_warm:
            print(f"\n  v7 warm: already_chosen в top-5 "
                  f"{n_hits} хитов у {n_students_with}/{len(val_warm)} "
                  f"студентов ({100 * n_students_with / len(val_warm):.1f}%)")
            if n_students_with > 0:
                print(f"          среднее хитов на затронутого студента: "
                      f"{n_hits / n_students_with:.2f}")

    # ==== 5. Итог ====
    print("\n" + "=" * 78)
    print("ИНТЕРПРЕТАЦИЯ:")
    print("=" * 78)
    print("""
    Если доля GT в history высокая (>= 0.5):
        -> студенты повторяют модули. Стратегия history_back почти
           наверняка даст высокий pNDCG@5.
        -> Следующий шаг: v20, где модель ОБЪЕДИНЯЕТ скор модели с
           bonuсом за попадание в историю (module_history_priority).

    Если доля GT в history низкая (< 0.3):
        -> повтор — редкое явление. Стратегия history_back провалится.
        -> Надо искать другой сигнал.
""")


if __name__ == "__main__":
    main()