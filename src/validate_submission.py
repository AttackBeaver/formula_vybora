"""
Разовая валидация submission.csv перед загрузкой на платформу.

Проверяет:
    * Колонки id, recommendations.
    * Все student_id из test присутствуют.
    * У каждого ровно 5 различных module_id.
    * Все module_id есть в каталоге modules.jsonl.

Запуск из корня проекта:
    python -m src.validate_submission
    python -m src.validate_submission --submission submissions/submission.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data_loader import load_all

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUBMISSION = PROJECT_ROOT / "submissions" / "submission.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Валидация submission.csv")
    parser.add_argument(
        "--submission", type=str, default=str(DEFAULT_SUBMISSION),
        help="Путь к submission (по умолчанию submissions/submission.csv)",
    )
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    sub_path = Path(args.submission)
    k = args.k

    print("=" * 72)
    print("ВАЛИДАЦИЯ SUBMISSION")
    print("=" * 72)
    print(f"\nФайл: {sub_path}")

    if not sub_path.exists():
        print(f"\n[!] Файл не найден: {sub_path}")
        return

    # --- Данные ---
    data = load_all()
    test = data["test"]
    modules = data["modules"]

    test_ids = {s["id"] for s in test}
    valid_modules = {m["module_id"] for m in modules}

    print(f"Test: {len(test_ids)} студентов, каталог: {len(valid_modules)} модулей")

    # --- Чтение ---
    df = pd.read_csv(sub_path)
    print(f"\nПрочитано: {len(df)} строк, колонки = {list(df.columns)}")

    # --- Проверки ---
    errors: list[str] = []

    # 1. Колонки
    if list(df.columns) != ["id", "recommendations"]:
        errors.append(f"Неверные колонки: {list(df.columns)}")

    # 2. Количество строк
    if len(df) != len(test_ids):
        errors.append(
            f"Строк {len(df)}, ожидалось {len(test_ids)}"
        )

    # 3. Множество id
    got_ids = set(df["id"].astype(str))
    missing = test_ids - got_ids
    extra = got_ids - test_ids
    if missing:
        errors.append(f"Отсутствуют id ({len(missing)}): {sorted(missing)[:5]} ...")
    if extra:
        errors.append(f"Лишние id ({len(extra)}): {sorted(extra)[:5]} ...")

    # 4. Каждая строка: k модулей, уникальные, из каталога
    n_wrong_len = 0
    n_dupes = 0
    n_bad_module = 0
    bad_rows: list[str] = []

    for _, row in df.iterrows():
        sid = str(row["id"])
        mods = str(row["recommendations"]).split()
        if len(mods) != k:
            n_wrong_len += 1
            if len(bad_rows) < 5:
                bad_rows.append(f"{sid}: {len(mods)} модулей")
        if len(set(mods)) != len(mods):
            n_dupes += 1
        for m in mods:
            if m not in valid_modules:
                n_bad_module += 1
                break

    if n_wrong_len:
        errors.append(f"Строк с неверным числом модулей: {n_wrong_len}")
    if n_dupes:
        errors.append(f"Строк с дубликатами: {n_dupes}")
    if n_bad_module:
        errors.append(f"Строк с модулем не из каталога: {n_bad_module}")

    if bad_rows:
        print("\n  Примеры проблемных строк:")
        for row in bad_rows:
            print(f"    {row}")

    # --- Итог ---
    print("\n" + "-" * 72)
    if errors:
        print("[FAIL] Найдены проблемы:")
        for e in errors:
            print(f"  - {e}")
        print("\n  ЗАГРУЖАТЬ НЕЛЬЗЯ — исправь и пересобери submission.")
        return

    print("[OK] Валидация пройдена.")
    print(f"\n  Строк:         {len(df)}")
    print(f"  Уникальных id: {df['id'].nunique()}")
    print(f"  Все по {k} модулей, все из каталога.")
    print("\n" + "=" * 72)
    print("ГОТОВО К ЗАГРУЗКЕ:")
    print(f"  {sub_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()