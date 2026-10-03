"""
Генерация рекомендаций (top-K) для набора студентов.

Обобщённый интерфейс: работаем с любой моделью, которая умеет возвращать
score для (student, module). Модель передаётся как функция score_fn(module_id) -> float.
Так мы сможем на следующих шагах подключить сюда LightGBM без переписывания
всей обвязки.
"""
from __future__ import annotations

from typing import Any, Callable


def recommend_top_k(
    candidates: list[str],
    score_fn: Callable[[str], float],
    k: int = 5,
) -> list[str]:
    """
    Возвращает top-k кандидатов по убыванию score.

    Tie-break: при равных score сортируем по module_id (алфавит).
    Это нужно для воспроизводимости: одинаковый вход -> одинаковый выход.

    Args:
        candidates: список module_id — претенденты на рекомендацию.
        score_fn: функция, считающая релевантность модуля.
        k: сколько рекомендаций вернуть.

    Returns:
        Список из k module_id, отсортированный от самого релевантного к менее.
    """
    if not candidates:
        raise ValueError("Пустой список кандидатов — невозможно сформировать топ-k")

    scored: list[tuple[str, float]] = [(m, float(score_fn(m))) for m in candidates]
    # Сортировка: сначала по score (убыв.), потом по module_id (возр.) для стабильности
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return [mid for mid, _ in scored[:k]]


def infer_popularity(
    students: list[dict[str, Any]],
    pop_global: dict[str, float],
    pop_by_inst: dict[str, dict[str, float]] | None = None,
    k: int = 5,
) -> dict[str, list[str]]:
    """
    Baseline: рекомендация по популярности.

    Логика:
        - Если у студента есть институт и есть статистика по институту —
          берём топ-k модулей из его института.
        - Иначе — глобальная популярность.
        - Всегда ранжируем ТОЛЬКО из available_modules, если он задан.

    Args:
        students: список записей val/test.
        pop_global: {module_id: score} — глобальная популярность.
        pop_by_inst: {institute: {module_id: score}} — опционально.
        k: размер рекомендации.

    Returns:
        dict[student_id, list[module_id]] длины k.
    """
    results: dict[str, list[str]] = {}

    for student in students:
        sid = student.get("id") or student.get("student_id")
        if sid is None:
            raise ValueError(f"У записи нет ни 'id', ни 'student_id': {student}")

        profile = student.get("profile") or {}
        inst = profile.get("institute") or None

        # Выбор источника популярности
        if pop_by_inst and inst and inst in pop_by_inst:
            pop = pop_by_inst[inst]
            source = f"institute:{inst}"
        else:
            pop = pop_global
            source = "global"

        # Кандидаты: только из available_modules, если он задан
        avail = student.get("available_modules")
        if avail:
            candidates = list(avail)
        else:
            # Fallback (для train без available): берём все известные модули
            candidates = list(pop.keys())

        if not candidates:
            raise ValueError(f"У студента {sid} нет кандидатов ({source})")

        def score(mid: str, _pop: dict[str, float] = pop) -> float:
            return _pop.get(mid, 0.0)

        results[sid] = recommend_top_k(candidates, score, k=k)

    return results