"""
Запись submission.csv в формате соревнования.

Формат (из ТЗ):
    id,recommendations
    s_000123,m_014 m_087 m_031 m_155 m_022

Правила:
    - одна строка на студента;
    - ровно k различных module_id;
    - порядок — от наиболее релевантного к менее;
    - module_id разделены пробелами.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_submission(
    recommendations: dict[str, list[str]],
    path: str | Path,
    k: int = 5,
) -> pd.DataFrame:
    """
    Сохраняет рекомендации в CSV.

    Args:
        recommendations: dict[student_id, list[module_id]].
        path: куда сохранять.
        k: требуемая длина каждой рекомендации.

    Returns:
        DataFrame, который был записан (для проверок).
    """
    rows: list[dict[str, str]] = []

    for sid, mods in recommendations.items():
        mods = list(mods)

        if len(mods) != k:
            raise ValueError(
                f"У студента {sid} длина рекомендации {len(mods)}, ожидалось {k}"
            )
        if len(set(mods)) != k:
            raise ValueError(
                f"У студента {sid} есть дубликаты в рекомендации: {mods}"
            )

        rows.append({"id": sid, "recommendations": " ".join(mods)})

    df = pd.DataFrame(rows)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    return df


def validate_submission(
    df: pd.DataFrame,
    expected_ids: set[str],
    valid_modules: set[str],
    k: int = 5,
) -> None:
    """
    Проверяет формат submission перед отправкой.

    Raises:
        AssertionError: при любом нарушении.
    """
    assert list(df.columns) == ["id", "recommendations"], \
        f"Неверные колонки: {list(df.columns)}"

    assert len(df) == len(expected_ids), \
        f"Строк {len(df)}, ожидалось {len(expected_ids)}"

    assert set(df["id"]) == expected_ids, \
        "Множество id не совпадает с ожидаемым"

    for _, row in df.iterrows():
        mods = row["recommendations"].split()
        assert len(mods) == k, f"{row['id']}: {len(mods)} модулей, нужно {k}"
        assert len(set(mods)) == k, f"{row['id']}: дубликаты в рекомендации"
        for m in mods:
            assert m in valid_modules, f"{row['id']}: модуль {m} отсутствует в каталоге"

    print(f"[submit] Валидация пройдена: {len(df)} строк, все по {k} модулей.")