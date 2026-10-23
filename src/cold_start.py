"""
Cold-start fallback для рекомендаций.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.data_loader import load_all
from src.evaluate import load_predictions
from src.metric import evaluate_pndcg5
from src.popularity import (
    compute_popularity,
    compute_popularity_by_institute,
)
from src.submit import write_submission

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUBMISSION = PROJECT_ROOT / "submissions" / "val_submission_ltr.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "submissions" / "val_submission_ltr_cold.csv"


def is_cold(record: dict[str, Any]) -> bool:
    """Холодный студент = нет ни профиля, ни истории."""
    profile = record.get("profile") or {}
    history = record.get("history") or []
    has_profile = bool(profile.get("institute") or profile.get("specialty"))
    has_history = any(len(h.get("choices", [])) > 0 for h in history)
    return not has_profile and not has_history


def identify_cold_students(students: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for s in students:
        sid = s.get("id") or s.get("student_id")
        if sid is None:
            continue
        if is_cold(s):
            out.add(sid)
    return out


def build_institute_index(
    students: list[dict[str, Any]],
) -> dict[frozenset[str], set[str]]:
    """Обратный индекс: frozenset(available_modules) → {institute, ...}."""
    index: dict[frozenset[str], set[str]] = defaultdict(set)
    for s in students:
        profile = s.get("profile") or {}
        inst = profile.get("institute")
        avail = s.get("available_modules")
        if not inst or not avail:
            continue
        index[frozenset(avail)].add(inst)
    return dict(index)


def infer_institute(
    available_modules: list[str],
    index: dict[frozenset[str], set[str]],
) -> str | None:
    """None, если сигнатура не найдена или неоднозначна (>1 институт)."""
    if not available_modules:
        return None
    insts = index.get(frozenset(available_modules))
    if insts is None or len(insts) != 1:
        return None
    return next(iter(insts))


def fallback_global_top_k(
    available_modules: list[str],
    pop_weighted: dict[str, float],
    k: int = 5,
) -> list[str]:
    if not available_modules:
        return []
    scored = [(m, float(pop_weighted.get(m, 0.0))) for m in available_modules]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return [mid for mid, _ in scored[:k]]


def fallback_institute_top_k(
    available_modules: list[str],
    inferred_inst: str | None,
    pop_by_institute: dict[str, dict[str, float]],
    pop_weighted_global: dict[str, float],
    k: int = 5,
) -> list[str]:
    if not available_modules:
        return []
    pop = pop_by_institute.get(inferred_inst, {}) if inferred_inst else {}
    if not pop:
        pop = pop_weighted_global
    scored = [(m, float(pop.get(m, 0.0))) for m in available_modules]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return [mid for mid, _ in scored[:k]]


def build_global_recommendations(
    students: list[dict[str, Any]],
    pop_weighted: dict[str, float],
    cold_ids: set[str],
    k: int = 5,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for s in students:
        sid = s.get("id") or s.get("student_id")
        if sid is None or sid not in cold_ids:
            continue
        avail = list(s.get("available_modules") or [])
        result[sid] = fallback_global_top_k(avail, pop_weighted, k=k)
    return result


def build_institute_recommendations(
    students: list[dict[str, Any]],
    pop_by_institute: dict[str, dict[str, float]],
    pop_weighted_global: dict[str, float],
    index: dict[frozenset[str], set[str]],
    cold_ids: set[str],
    k: int = 5,
) -> tuple[dict[str, list[str]], dict[str, str | None]]:
    result: dict[str, list[str]] = {}
    inferred: dict[str, str | None] = {}
    for s in students:
        sid = s.get("id") or s.get("student_id")
        if sid is None or sid not in cold_ids:
            continue
        avail = list(s.get("available_modules") or [])
        inst = infer_institute(avail, index)
        inferred[sid] = inst
        result[sid] = fallback_institute_top_k(
            avail, inst, pop_by_institute, pop_weighted_global, k=k,
        )
    return result, inferred


def apply_fallback(
    rec_ltr: dict[str, list[str]],
    rec_fallback: dict[str, list[str]],
    k: int = 5,
) -> dict[str, list[str]]:
    """Перезаписывает рекомендации для cold-студентов, сохраняя длину k."""
    merged: dict[str, list[str]] = {}
    for sid, ltr_rec in rec_ltr.items():
        fb = rec_fallback.get(sid)
        if fb is None:
            merged[sid] = list(ltr_rec)
            continue
        rec = list(fb)
        if len(rec) < k:
            seen = set(rec)
            for m in ltr_rec:
                if len(rec) >= k:
                    break
                if m not in seen:
                    rec.append(m)
                    seen.add(m)
        merged[sid] = rec[:k]
    return merged


def _report_segments(
    name: str,
    rec: dict[str, list[str]],
    gt_raw: dict[str, str],
    cold_ids: set[str],
    k: int = 5,
) -> tuple[float, float, float]:
    """Печатает mean / cold / warm; возвращает их."""
    mean_all, per_student = evaluate_pndcg5(rec, gt_raw, k=k)
    cold_scores = [s for sid, s in per_student.items() if sid in cold_ids]
    warm_scores = [s for sid, s in per_student.items() if sid not in cold_ids]
    c = float(np.mean(cold_scores)) if cold_scores else 0.0
    w = float(np.mean(warm_scores)) if warm_scores else 0.0

    print(f"  [{name}]")
    print(f"      mean    = {mean_all:.4f}   (n={len(per_student)})")
    print(f"      cold    = {c:.4f}   (n={len(cold_scores)})")
    print(f"      warm    = {w:.4f}   (n={len(warm_scores)})")
    return mean_all, c, w


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cold-start fallback: сравнение двух стратегий",
    )
    parser.add_argument(
        "--submission", type=str, default=str(DEFAULT_SUBMISSION),
        help="Входной submission (по умолчанию val_submission_ltr.csv)",
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help="Куда сохранить лучшую стратегию",
    )
    parser.add_argument(
        "--test-cold-share", type=float, default=0.352,
        help="Доля cold-студентов в test — для перевзвешенной оценки",
    )
    args = parser.parse_args()

    sub_path = Path(args.submission)
    out_path = Path(args.output)
    test_cold_share = float(args.test_cold_share)

    print("=" * 78)
    print("COLD-START FALLBACK: сравнение стратегий")
    print("=" * 78)
    print(f"\nInput:  {sub_path}")
    print(f"Output: {out_path}")

    if not sub_path.exists():
        print(f"\n[!] Файл не найден: {sub_path}")
        return

    # 1. Данные
    data = load_all()
    val = data["val"]
    test = data["test"]
    train = data["train"]
    val_sol = data["val_solution"]
    gt_raw = dict(zip(val_sol["id"], val_sol["ground_truth"]))

    print(f"\n[1/6] Данные: val={len(val)}, test={len(test)}, train={len(train)}")

    # 2. Cold-студенты в val
    cold_ids_val = identify_cold_students(val)
    print(f"\n[2/6] Cold-студентов в val: {len(cold_ids_val)} "
          f"({100 * len(cold_ids_val) / len(val):.1f}%)")

    # 3. Индекс сигнатур available_modules (val + test warm)
    val_warm = [s for s in val if not is_cold(s)]
    test_warm = [s for s in test if not is_cold(s)]
    index = build_institute_index(val_warm + test_warm)
    print(f"\n[3/6] Индекс сигнатур available_modules: {len(index)} уникальных наборов")

    inferred_map: dict[str, str | None] = {}
    for s in val:
        sid = s.get("id") or s.get("student_id")
        if sid is None or sid not in cold_ids_val:
            continue
        avail = list(s.get("available_modules") or [])
        inferred_map[sid] = infer_institute(avail, index)

    n_resolved = sum(1 for v in inferred_map.values() if v is not None)
    print(f"      Cold-val с восстановленным институтом: "
          f"{n_resolved} / {len(cold_ids_val)}")
    if n_resolved > 0:
        dist = Counter(v for v in inferred_map.values() if v is not None)
        print("      Распределение восстановленных институтов:")
        for inst, cnt in dist.most_common():
            print(f"        {cnt:>4}  {inst}")

    # 4. Статистики популярности
    print("\n[4/6] Популярность: глобальная + по институтам")
    pop_weighted, _ = compute_popularity(train)
    pop_by_inst = compute_popularity_by_institute(train)
    print(f"      Глобальная: {len(pop_weighted)} модулей")
    print(f"      Институтов: {len(pop_by_inst)}")

    # 5. Три варианта
    rec_ltr = load_predictions(sub_path)

    rec_fb_global = build_global_recommendations(
        val, pop_weighted, cold_ids_val, k=5,
    )
    rec_global = apply_fallback(rec_ltr, rec_fb_global, k=5)

    rec_fb_inst, _ = build_institute_recommendations(
        val, pop_by_inst, pop_weighted, index, cold_ids_val, k=5,
    )
    rec_inst = apply_fallback(rec_ltr, rec_fb_inst, k=5)

    print("\n[5/6] Метрики на val:\n")
    _, cold_ltr, warm_ref = _report_segments(
        "LTR (без fallback)", rec_ltr, gt_raw, cold_ids_val, k=5,
    )
    print()
    _, cold_g, _ = _report_segments(
        "A: Global popularity", rec_global, gt_raw, cold_ids_val, k=5,
    )
    print()
    _, cold_i, _ = _report_segments(
        "B: Institute-inferred", rec_inst, gt_raw, cold_ids_val, k=5,
    )

    # 6. Выбор лучшей и сохранение
    print("\n" + "-" * 78)
    print("Сравнение cold-сегмента:")
    print(f"    LTR:                   cold = {cold_ltr:.4f}")
    print(f"    A: Global popularity:  cold = {cold_g:.4f}   "
          f"(Δ vs LTR = {cold_g - cold_ltr:+.4f})")
    print(f"    B: Institute-inferred: cold = {cold_i:.4f}   "
          f"(Δ vs LTR = {cold_i - cold_ltr:+.4f})")

    if cold_i > cold_g + 0.005:
        winner_name = "B (institute-inferred)"
        winner_rec = rec_inst
        winner_cold = cold_i
    else:
        winner_name = "A (global popularity)"
        winner_rec = rec_global
        winner_cold = cold_g

    print(f"\n    Победитель: {winner_name}")

    # Перевзвешенная оценка под test (warm одинаков во всех стратегиях)
    est_ltr = test_cold_share * cold_ltr + (1 - test_cold_share) * warm_ref
    est_win = test_cold_share * winner_cold + (1 - test_cold_share) * warm_ref

    print("\n" + "-" * 78)
    print(f"Перевзвешенная оценка под test (cold_share={test_cold_share:.3f}):")
    print(f"    LTR (без fallback):  {est_ltr:.4f}")
    print(f"    Победитель:          {est_win:.4f}")
    print(f"    Δ:                   {est_win - est_ltr:+.4f}")

    print("\n[6/6] Сохраняю победителя...")
    write_submission(winner_rec, out_path)
    print(f"      Сохранено: {out_path}")


if __name__ == "__main__":
    main()