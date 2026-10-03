"""
Проверка, что окружение настроено корректно.
Запускается один раз после установки.
"""
import sys
import importlib

REQUIRED = [
    "numpy",
    "pandas",
    "sklearn",
    "lightgbm",
    "xgboost",
    "catboost",
    "scipy",
    "matplotlib",
    "seaborn",
    "tqdm",
    "joblib",
]

def check() -> None:
    print(f"Python: {sys.version}\n")
    print(f"{'Библиотека':<15} {'Статус':<10} Версия")
    print("-" * 45)
    all_ok = True
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "?")
            print(f"{name:<15} {'OK':<10} {version}")
        except ImportError:
            print(f"{name:<15} {'FAIL':<10} -")
            all_ok = False
    print("-" * 45)
    if all_ok:
        print("\n[+] Все библиотеки на месте. Окружение готово к работе.")
    else:
        print("\n[!] Часть библиотек не установлена. Проверь pip install.")

if __name__ == "__main__":
    check()