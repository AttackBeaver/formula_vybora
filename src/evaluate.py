"""
CLI: оценка pNDCG@5 по submission-файлу.

Считает:
    1. Наш pNDCG@5 (src.metric) — с разбивкой по студентам.
    2. Официальный pNDCG@5 (metric.py в корне) — как контрольный скор.

Если оба числа совпадают — реализация верна.

Запуск из корня проекта:
    python -m src.evaluate
    python -m src.evaluate --submission submissions/val_submission_popularity.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_loader import load_all
from src.metric import (
    evaluate_pndcg5,
    hit_at_k,
    parse_ground_truth,
    pndcg_at_k,
    recall_at_k,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUBMISSION = PROJECT_ROOT / "submissions" / "val_submission_popularity.csv"


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------
def load_predictions(path: Path) -> dict[str, list[str]]:
    """Читает submission.csv формата id,recommendations."""
    df = pd.read_csv(path)
    return {
        row["id"]: str(row["recommendations"]).split()
        for _, row in df.iterrows()
    }


def load_ground_truth(df: pd.DataFrame) -> dict[str, str]:
    """{id: 'm_XXX:priority ...'}"""
    return dict(zip(df["id"], df["ground_truth"]))


# ---------------------------------------------------------------------------
# Сверка с официальным metric.py
# ---------------------------------------------------------------------------
def official_score(sub_path: Path, sol_path: Path) -> str:
    """
    Вызывает официальный metric.py через subprocess.
    Возвращает его stdout как строку (одна или две строки).
    """
    import subprocess
    import sys

    official = PROJECT_ROOT / "metric.py"
    if not official.exists():
        return "[официальный metric.py не найден — пропуск сверки]"

    result = subprocess.run(
        [sys.executable, str(official),
         "--solution", str(sol_path),
         "--submission", str(sub_path)],
        capture_output=True, text=True, encoding="utf-8",
    )
    if result.returncode != 0:
        return f"[ошибка официального metric.py]\n{result.stderr.strip()}"
    return result.stdout.strip()


# ---------------------------------------------------------------------------
# Основной сценарий
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка pNDCG@5")
    parser.add_argument(
        "--submission", type=str, default=str(DEFAULT_SUBMISSION),
        help="Путь к submission (по умолчанию: val_submission_popularity.csv)",
    )
    args = parser.parse_args()

    sub_path = Path(args.submission)
    sol_path = PROJECT_ROOT / "data" / "val_solution.csv"

    print("=" * 64)
    print("ОЦЕНКА pNDCG@5")
    print("=" * 64)

    # ---- 1. Данные ------------------------------------------------------
    data = load_all()
    val_sol = data["val_solution"]

    print(f"\n[1/4] Submission: {sub_path.name}")
    preds = load_predictions(sub_path)
    gt = load_ground_truth(val_sol)
    print(f"      Студентов в submission:   {len(preds)}")
    print(f"      Студентов в ground truth: {len(gt)}")

    # ---- 2. Наша метрика ------------------------------------------------
    print("\n[2/4] Наша метрика (src.metric):")
    mean, per_student = evaluate_pndcg5(preds, gt)

    values = np.array(list(per_student.values()))
    n_perfect = int((values >= 0.999).sum())
    n_zero = int((values <= 0.001).sum())

    print(f"      mean    = {mean:.4f}")
    print(f"      median  = {np.median(values):.4f}")
    print(f"      std     = {values.std():.4f}")
    print(f"      min     = {values.min():.4f}")
    print(f"      p25     = {np.percentile(values, 25):.4f}")
    print(f"      p75     = {np.percentile(values, 75):.4f}")
    print(f"      max     = {values.max():.4f}")
    print(f"      идеальных (=1.0): {n_perfect:>5}  "
          f"({100 * n_perfect / len(values):.1f}%)")
    print(f"      нулевых   (=0.0): {n_zero:>5}  "
          f"({100 * n_zero / len(values):.1f}%)")

    # ---- 3. Справочные метрики ------------------------------------------
    print("\n[3/4] Справочно:")
    hits, recalls = [], []
    for sid, gt_str in gt.items():
        gt_parsed = parse_ground_truth(gt_str)
        if not gt_parsed:
            continue
        ranked = preds.get(sid, [])
        hits.append(hit_at_k(ranked, gt_parsed))
        recalls.append(recall_at_k(ranked, gt_parsed))
    print(f"      hit@5    = {np.mean(hits):.4f}")
    print(f"      recall@5 = {np.mean(recalls):.4f}")

    # ---- 4. Официальный metric.py ---------------------------------------
    print("\n[4/4] Официальный metric.py:")
    print("      " + official_score(sub_path, sol_path).replace("\n", "\n      "))

    print("\n" + "=" * 64)
    print(f"ИТОГ: pNDCG@5 = {mean:.4f}")
    print("=" * 64)


if __name__ == "__main__":
    main()