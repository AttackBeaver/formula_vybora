"""
Сегментный анализ качества LTR-модели на val.

Отвечает на вопросы:
    1. Где именно теряются баллы: на warm или на cold студентах?
    2. Как распределён pNDCG@5 по числу периодов истории?
    3. Какой ожидаемый private score на test с учётом его распределения
       (35.2% cold vs 2.9% в val)?

Идея перевзвешенной оценки:
    В test доля cold-студентов = 35.2%.
    В val доля cold-студентов = 2.9%.
    Значит, val-скор систематически завышен.
    Пересчитываем: test_estimate = cold_share * mean_cold + (1 - cold_share) * mean_warm.

Запуск из корня проекта:
    python -m src.segment_analysis
    python -m src.segment_analysis --submission submissions/val_submission_ltr.csv
    python -m src.segment_analysis --test-cold-share 0.352
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from src.data_loader import load_all
from src.evaluate import load_predictions
from src.metric import parse_ground_truth, pndcg_at_k

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUBMISSION = PROJECT_ROOT / "submissions" / "val_submission_ltr.csv"
DEFAULT_TEST_COLD_SHARE = 0.352


# ---------------------------------------------------------------------------
# Хелперы для сегментации
# ---------------------------------------------------------------------------
def is_cold(record: dict[str, Any]) -> bool:
    """Холодный = нет ни профиля, ни истории."""
    profile = record.get("profile") or {}
    history = record.get("history") or []
    has_profile = bool(profile.get("institute") or profile.get("specialty"))
    has_history = any(len(h.get("choices", [])) > 0 for h in history)
    return not has_profile and not has_history


def n_periods(record: dict[str, Any]) -> int:
    return len(record.get("history") or [])


def n_history_choices(record: dict[str, Any]) -> int:
    return sum(len(h.get("choices", [])) for h in record.get("history") or [])


def get_institute(record: dict[str, Any]) -> str:
    inst = (record.get("profile") or {}).get("institute")
    return inst or "UNKNOWN"


# ---------------------------------------------------------------------------
# Подсчёт pNDCG@5 по студентам
# ---------------------------------------------------------------------------
def compute_per_student_scores(
    predictions: dict[str, list[str]],
    gt_raw: dict[str, str],
    k: int = 5,
) -> dict[str, float]:
    """Возвращает {student_id: pNDCG@5} только для студентов с непустым GT."""
    per_student: dict[str, float] = {}
    for sid, gt_str in gt_raw.items():
        gt = parse_ground_truth(gt_str)
        if not gt:
            continue
        ranked = predictions.get(sid, [])
        per_student[sid] = pndcg_at_k(ranked, gt, k=k)
    return per_student


# ---------------------------------------------------------------------------
# Печать сегмента
# ---------------------------------------------------------------------------
def _print_segment(name: str, scores: list[float], width: int = 52) -> None:
    if not scores:
        return
    arr = np.array(scores, dtype=np.float64)
    print(
        f"  {name:<{width}} "
        f"n={len(arr):>5}  "
        f"mean={arr.mean():.4f}  "
        f"median={np.median(arr):.4f}  "
        f"std={arr.std():.4f}"
    )


def _bucket_by_periods(scores_by_sid: dict[str, float], val_by_id: dict[str, dict]) -> dict[str, list[float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for sid, score in scores_by_sid.items():
        rec = val_by_id.get(sid)
        if rec is None:
            continue
        buckets[f"periods={n_periods(rec)}"].append(score)
    return dict(buckets)


def _bucket_by_institute(scores_by_sid: dict[str, float], val_by_id: dict[str, dict]) -> dict[str, list[float]]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for sid, score in scores_by_sid.items():
        rec = val_by_id.get(sid)
        if rec is None:
            continue
        buckets[f"inst={get_institute(rec)}"].append(score)
    return dict(buckets)


def _bucket_by_choices(scores_by_sid: dict[str, float], val_by_id: dict[str, dict]) -> dict[str, list[float]]:
    """Бакеты по числу выборов в истории (0, 1-5, 6-10, 11+)."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for sid, score in scores_by_sid.items():
        rec = val_by_id.get(sid)
        if rec is None:
            continue
        n = n_history_choices(rec)
        if n == 0:
            key = "choices=0"
        elif n <= 5:
            key = "choices=1-5"
        elif n <= 10:
            key = "choices=6-10"
        else:
            key = "choices=11+"
        buckets[key].append(score)
    return dict(buckets)


# ---------------------------------------------------------------------------
# Основной сценарий
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Сегментный анализ pNDCG@5")
    parser.add_argument(
        "--submission", type=str, default=str(DEFAULT_SUBMISSION),
        help="Путь к submission (по умолчанию val_submission_ltr.csv)",
    )
    parser.add_argument(
        "--test-cold-share", type=float, default=DEFAULT_TEST_COLD_SHARE,
        help="Доля cold-студентов в test (по умолчанию 0.352)",
    )
    parser.add_argument(
        "--top-worst", type=int, default=5,
        help="Сколько худших студентов показать",
    )
    args = parser.parse_args()

    sub_path = Path(args.submission)
    test_cold_share = float(args.test_cold_share)

    print("=" * 78)
    print("СЕГМЕНТНЫЙ АНАЛИЗ pNDCG@5")
    print("=" * 78)
    print(f"\nSubmission:        {sub_path}")
    print(f"test_cold_share:   {test_cold_share:.3f}")

    if not sub_path.exists():
        print(f"\n[!] Файл не найден: {sub_path}")
        print("    Сначала запусти: python -m src.run_ltr")
        return

    # ---- 1. Данные ----
    data = load_all()
    val = data["val"]
    val_sol = data["val_solution"]

    preds = load_predictions(sub_path)
    gt_raw = dict(zip(val_sol["id"], val_sol["ground_truth"]))
    val_by_id = {s["id"]: s for s in val}

    print(f"\n[1/4] Загружено: val={len(val)}, preds={len(preds)}, gt={len(gt_raw)}")

    # ---- 2. Поштучные скоры ----
    print("\n[2/4] Считаю pNDCG@5 по студентам...")
    per_student = compute_per_student_scores(preds, gt_raw, k=5)
    values = np.array(list(per_student.values()), dtype=np.float64)

    print(
        f"      mean={values.mean():.4f}  "
        f"median={np.median(values):.4f}  "
        f"std={values.std():.4f}  "
        f"p25={np.percentile(values, 25):.4f}  "
        f"p75={np.percentile(values, 75):.4f}"
    )

    # ---- 3. Сегменты ----
    print("\n[3/4] Сегменты:")

    # cold vs warm
    cold_scores: list[float] = []
    warm_scores: list[float] = []
    for sid, score in per_student.items():
        rec = val_by_id.get(sid)
        if rec is None:
            continue
        (cold_scores if is_cold(rec) else warm_scores).append(score)

    print("\n  [cold vs warm]")
    _print_segment("cold", cold_scores)
    _print_segment("warm", warm_scores)

    print("\n  [по числу периодов истории]")
    for name, scores in sorted(_bucket_by_periods(per_student, val_by_id).items()):
        _print_segment(name, scores)

    print("\n  [по числу выборов в истории]")
    order = ["choices=0", "choices=1-5", "choices=6-10", "choices=11+"]
    buckets = _bucket_by_choices(per_student, val_by_id)
    for name in order:
        if name in buckets:
            _print_segment(name, buckets[name])

    print("\n  [по институтам]")
    inst_buckets = _bucket_by_institute(per_student, val_by_id)
    for name, scores in sorted(inst_buckets.items(), key=lambda kv: -len(kv[1])):
        _print_segment(name, scores, width=58)

    # ---- 4. Перевзвешенная оценка под test ----
    print("\n[4/4] Перевзвешенная оценка под test-распределение:")

    cold_mean = float(np.mean(cold_scores)) if cold_scores else 0.0
    warm_mean = float(np.mean(warm_scores)) if warm_scores else 0.0
    val_mean = float(values.mean())
    reweighted = test_cold_share * cold_mean + (1.0 - test_cold_share) * warm_mean
    gap = val_mean - reweighted

    print(f"      cold_mean (val)       = {cold_mean:.4f}   (n={len(cold_scores)})")
    print(f"      warm_mean (val)       = {warm_mean:.4f}   (n={len(warm_scores)})")
    print(f"      val_mean              = {val_mean:.4f}")
    print(
        f"      test_estimate         = "
        f"{test_cold_share:.3f}·{cold_mean:.4f} + "
        f"{1.0 - test_cold_share:.3f}·{warm_mean:.4f} "
        f"= {reweighted:.4f}"
    )
    print(f"      разрыв (val − test)   = {gap:+.4f}")

    # ---- Топ-худших студентов ----
    if args.top_worst > 0:
        print(f"\n[Топ-{args.top_worst} худших студентов на val]")
        worst = sorted(per_student.items(), key=lambda kv: kv[1])[: args.top_worst]
        for sid, score in worst:
            rec = val_by_id.get(sid)
            if rec is None:
                continue
            n_p = n_periods(rec)
            n_c = n_history_choices(rec)
            cold_flag = "COLD" if is_cold(rec) else "warm"
            print(
                f"      {sid}  score={score:.4f}  "
                f"periods={n_p}  choices={n_c}  {cold_flag}"
            )

    print("\n" + "=" * 78)
    print("ВЫВОД")
    print("=" * 78)
    print(
        f"\n  Текущий val-скор: {val_mean:.4f}"
        f"\n  Ожидание на test:  ~{reweighted:.4f}"
        f"\n  Разрыв:            {gap:+.4f} (val завышает)"
    )

    if cold_scores and cold_mean < warm_mean - 0.10:
        print(
            "\n  [!] Холодные студенты сильно отстают. "
            "Приоритет: cold-start fallback."
        )
    if gap > 0.03:
        print(
            "\n  [!] Разрыв val/test > 0.03. "
            "Целиться в val нужно с запасом."
        )

    print()


if __name__ == "__main__":
    main()