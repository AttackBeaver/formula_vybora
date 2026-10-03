"""
Модуль загрузки данных для хакатона "Формула выбора".

Отвечает за:
- чтение JSONL и CSV файлов;
- кэширование в pickle (чтобы не парсить JSON каждый раз);
- базовую валидацию структуры данных.

Использование:
    from src.data_loader import load_all, print_summary
    data = load_all()
    print_summary(data)
"""
from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------
# __file__        = .../formula_vybora/src/data_loader.py
# .parent         = .../formula_vybora/src
# .parent.parent  = .../formula_vybora
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "_cache"


# ---------------------------------------------------------------------------
# Базовое чтение JSONL
# ---------------------------------------------------------------------------
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Читает JSONL-файл: один JSON-объект на строку.

    Args:
        path: путь к .jsonl файлу.

    Returns:
        Список словарей (каждая строка — dict).

    Raises:
        FileNotFoundError: если файла нет.
        ValueError: если в строке битый JSON (с указанием номера строки).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {path}\n"
            f"Проверь, что данные лежат в {DATA_DIR}"
        )

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:                # пропускаем пустые строки
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Битый JSON в {path.name}, строка {line_num}: {e}"
                ) from e
    return records


# ---------------------------------------------------------------------------
# Публичные функции загрузки
# ---------------------------------------------------------------------------
def load_train() -> list[dict[str, Any]]:
    """История выбора дисциплин за первые два периода."""
    return _read_jsonl(DATA_DIR / "train.jsonl")


def load_val() -> list[dict[str, Any]]:
    """Валидация: студенты с доступными модулями, но без ответов."""
    return _read_jsonl(DATA_DIR / "val.jsonl")


def load_test() -> list[dict[str, Any]]:
    """Тест: для него нужно сформировать submission.csv."""
    return _read_jsonl(DATA_DIR / "test.jsonl")


def load_modules() -> list[dict[str, Any]]:
    """Каталог всех 322 дисциплин."""
    return _read_jsonl(DATA_DIR / "modules.jsonl")


def load_val_solution() -> pd.DataFrame:
    """Правильные ответы для val: id, ground_truth ('m_XXX:priority ...')."""
    return pd.read_csv(DATA_DIR / "val_solution.csv")


def load_sample_submission() -> pd.DataFrame:
    """Пример итогового файла — чтобы не ошибиться в формате."""
    return pd.read_csv(DATA_DIR / "sample_submission.csv")


# ---------------------------------------------------------------------------
# Кэширование (pickle)
# ---------------------------------------------------------------------------
def _cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.pkl"


def save_cache(obj: Any, name: str) -> None:
    """Сохраняет объект в pickle-файл."""
    with _cache_path(name).open("wb") as f:
        pickle.dump(obj, f)


def load_cache(name: str) -> Any:
    """Загружает объект из pickle-файла. None, если кэша нет."""
    path = _cache_path(name)
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Загрузка всего сразу (с кэшем)
# ---------------------------------------------------------------------------
def load_all(use_cache: bool = True, force_reload: bool = False) -> dict[str, Any]:
    """
    Загружает все данные проекта.

    Args:
        use_cache: если True и кэш есть — берёт из pickle.
        force_reload: если True — игнорирует кэш и перечитывает JSONL.

    Returns:
        Словарь с ключами:
            train, val, test, modules, val_solution, sample_submission
    """
    cache_name = "all_data"
    if use_cache and not force_reload:
        cached = load_cache(cache_name)
        if cached is not None:
            print("[data_loader] Данные загружены из кэша.")
            return cached

    print("[data_loader] Читаю JSONL и CSV...")
    data = {
        "train": load_train(),
        "val": load_val(),
        "test": load_test(),
        "modules": load_modules(),
        "val_solution": load_val_solution(),
        "sample_submission": load_sample_submission(),
    }
    if use_cache:
        save_cache(data, cache_name)
        print(f"[data_loader] Кэш сохранён в {_cache_path(cache_name)}")
    return data


# ---------------------------------------------------------------------------
# Диагностика: печатаем базовую статистику
# ---------------------------------------------------------------------------
def _example_student(record: dict[str, Any]) -> str:
    """Красиво печатает одного студента (обрезка длинных полей)."""
    rec = dict(record)
    if "history" in rec:
        rec["history"] = [
            {
                "period": h.get("period"),
                "n_choices": len(h.get("choices", [])),
                "choices_preview": h.get("choices", [])[:3],
            }
            for h in rec["history"]
        ]
    if "available_modules" in rec:
        rec["available_modules"] = (
            f"[{len(rec['available_modules'])} модулей] "
            f"{rec['available_modules'][:5]}..."
        )
    return json.dumps(rec, ensure_ascii=False, indent=2)


def print_summary(data: dict[str, Any]) -> None:
    """Печатает ключевую статистику по всем файлам."""
    train = data["train"]
    val = data["val"]
    test = data["test"]
    modules = data["modules"]
    val_sol = data["val_solution"]
    sample_sub = data["sample_submission"]

    print("\n" + "=" * 70)
    print("СВОДКА ПО ДАННЫМ")
    print("=" * 70)

    # --- размеры ---
    print("\n[Размеры]")
    print(f"  train:            {len(train):>7} студентов")
    print(f"  val:              {len(val):>7} студентов")
    print(f"  test:             {len(test):>7} студентов")
    print(f"  modules:          {len(modules):>7} дисциплин")
    print(f"  val_solution:     {len(val_sol):>7} строк")
    print(f"  sample_submission:{len(sample_sub):>7} строк")

    # --- ключи первого студента в каждом файле ---
    print("\n[Ключи первого студента]")
    print(f"  train[0].keys():   {list(train[0].keys())}")
    print(f"  val[0].keys():     {list(val[0].keys())}")
    print(f"  test[0].keys():    {list(test[0].keys())}")
    print(f"  modules[0].keys(): {list(modules[0].keys())}")

    # --- пример одного студента из train ---
    print("\n[Пример train[0]]")
    print(_example_student(train[0]))

    # --- пример одного студента из val ---
    print("\n[Пример val[0]]")
    print(_example_student(val[0]))

    # --- пример модуля ---
    print("\n[Пример modules[0]]")
    print(json.dumps(modules[0], ensure_ascii=False, indent=2))

    # --- статистика по history ---
    print("\n[История выбора в train]")
    n_periods = [len(s.get("history", [])) for s in train]
    n_choices = [
        sum(len(h.get("choices", [])) for h in s.get("history", []))
        for s in train
    ]
    priorities: list[int] = []
    for s in train:
        for h in s.get("history", []):
            for c in h.get("choices", []):
                priorities.append(c.get("priority"))

    print(
        f"  Периодов на студента: min={min(n_periods)}, "
        f"max={max(n_periods)}, avg={sum(n_periods)/len(n_periods):.2f}"
    )
    print(
        f"  Выборов на студента:  min={min(n_choices)}, "
        f"max={max(n_choices)}, avg={sum(n_choices)/len(n_choices):.2f}"
    )
    print(f"  Распределение приоритетов: {dict(sorted(Counter(priorities).items()))}")

    # --- холодные пользователи в test ---
    print("\n[Холодный старт]")

    def is_cold(rec: dict[str, Any]) -> bool:
        prof = rec.get("profile") or {}
        hist = rec.get("history") or []
        has_profile = bool(prof.get("institute") or prof.get("specialty"))
        has_history = any(len(h.get("choices", [])) > 0 for h in hist)
        return not has_profile and not has_history

    n_cold_test = sum(is_cold(s) for s in test)
    n_cold_val = sum(is_cold(s) for s in val)
    print(
        f"  test: {n_cold_test} / {len(test)} "
        f"({100 * n_cold_test / len(test):.1f}%)"
    )
    print(
        f"  val:  {n_cold_val} / {len(val)} "
        f"({100 * n_cold_val / len(val):.1f}%)"
    )

    # --- available_modules ---
    if val and "available_modules" in val[0]:
        n_avail = [len(s.get("available_modules", [])) for s in val]
        print("\n[available_modules в val]")
        print(
            f"  min={min(n_avail)}, max={max(n_avail)}, "
            f"avg={sum(n_avail)/len(n_avail):.1f}"
        )

    # --- val_solution пример ---
    print("\n[Пример val_solution]")
    print(val_sol.head(3).to_string(index=False))

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# CLI: python -m src.data_loader
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = load_all(use_cache=True, force_reload=False)
    print_summary(data)