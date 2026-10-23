"""
Популярность дисциплин — baseline-модель для рекомендаций.

Популярность модуля = сумма весов приоритетов из train. Модуль, который
чаще ставили на 1-е место, получает больший скор, чем модуль с тем же
числом выборов, но с низкими приоритетами. Институтская популярность
используется как персонализированный cold-start fallback.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

# pNDCG@5 weights из ТЗ
PRIORITY_WEIGHTS: dict[int, float] = {
    1: 1.00, 2: 0.80, 3: 0.60, 4: 0.45, 5: 0.35,
}

DEFAULT_INSTITUTE = "UNKNOWN"


def compute_popularity(
    train: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, int]]:
    """Возвращает (weighted_scores, raw_counts): {module_id: значение}."""
    weighted: dict[str, float] = defaultdict(float)
    counts: Counter[str] = Counter()

    for student in train:
        for history in student.get("history", []):
            for choice in history.get("choices", []):
                mid = choice["module_id"]
                priority = choice.get("priority", 5)
                weighted[mid] += PRIORITY_WEIGHTS.get(priority, 0.0)
                counts[mid] += 1

    return dict(weighted), dict(counts)


def compute_popularity_by_institute(
    train: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    """→ {institute_name: {module_id: weighted_score}}."""
    per_inst: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for student in train:
        inst = (student.get("profile") or {}).get("institute") or DEFAULT_INSTITUTE
        for history in student.get("history", []):
            for choice in history.get("choices", []):
                mid = choice["module_id"]
                priority = choice.get("priority", 5)
                per_inst[inst][mid] += PRIORITY_WEIGHTS.get(priority, 0.0)

    return {inst: dict(m) for inst, m in per_inst.items()}