"""
estimate_test_score.py — композиционно-корректная оценка public.

ЗАЧЕМ:
    Наивная оценка в run_ltr_v* берёт warm = среднее по ВСЕМ warm-val
    студентам. Но в val warm на 50.3% состоит из val_p1 (лёгкий сегмент,
    ~0.55), а в test доля p1 во всём множестве = 2%. Значит, наивная
    оценка систематически завышена.

    Правильные веса test (из ТЗ):
        cold = 0.352
        p1   = 0.020
        p2   = 0.324
        p3   = 0.304   ← proxy = val_p2, т.к. после max_history_periods=2
                         test_p3 структурно совпадает с val_p2
                         (2 периода истории → предсказать 3-й).

ИСПОЛЬЗОВАНИЕ:
    python -m src.estimate_test_score submissions/val_submission_ltr_v29.csv
    python -m src.estimate_test_score submissions/val_submission_ltr_v7.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.cold_start import identify_cold_students
from src.data_loader import load_all
from src.metric import evaluate_pndcg5

# Доли сегментов в test (из ТЗ, § 2).
TEST_SHARES: dict[str, float] = {
    "cold": 0.352,
    "p1":   0.020,
    "p2":   0.324,
    "p3":   0.304,
}


def read_submission(path: Path) -> dict[str, list[str]]:
    """Читает CSV-сабмит, возвращает {student_id: [module_id, ...]}.

    Поддерживает форматы:
        * ('id', 'recommendations')      — модули через пробел/запятую
        * ('id', 'prediction')           — модули через пробел
        * ('id', 'module_1'..'module_5') — по колонке на позицию
        * ('student_id', ...)            — синоним 'id'
    """
    df = pd.read_csv(path)
    cols = list(df.columns)

    id_col = None
    for cand in ("id", "student_id", "sid"):
        if cand in cols:
            id_col = cand
            break
    if id_col is None:
        raise ValueError(f"Нет колонки с id. Колонки: {cols}")

    # --- Формат "одна текстовая колонка с рекомендациями" ---
    for cand in ("recommendations", "prediction", "recommendation", "top5"):
        if cand in cols:
            rec: dict[str, list[str]] = {}
            for sid, raw in zip(df[id_col], df[cand]):
                if pd.isna(raw):
                    rec[str(sid)] = []
                    continue
                # Разделители: пробел, запятая, точка с запятой, '|'.
                s = str(raw).replace(",", " ").replace(";", " ").replace("|", " ")
                rec[str(sid)] = [tok for tok in s.split() if tok]
            return rec

    # --- Формат "module_1".."module_k" ---
    mod_cols = sorted(c for c in cols if c.startswith("module_"))
    if mod_cols:
        rec = {}
        for _, row in df.iterrows():
            rec[str(row[id_col])] = [
                str(row[c]) for c in mod_cols if pd.notna(row[c])
            ]
        return rec

    raise ValueError(
        f"Не распознан формат submission. Колонки: {cols}. "
        f"Ожидалось ('id','recommendations') / ('id','prediction') / "
        f"('id','module_1'..'module_5')."
    )


def split_val_by_periods(val: list[dict]) -> dict[str, list[str]]:
    """cold / p1 (1 период) / p2 (2+ периодов) по val.jsonl."""
    cold_ids = identify_cold_students(val)
    p1: list[str] = []
    p2: list[str] = []
    for s in val:
        sid = s["id"]
        if sid in cold_ids:
            continue
        n_periods = len(s.get("history") or [])
        if n_periods == 1:
            p1.append(sid)
        else:
            p2.append(sid)
    return {
        "cold": sorted(cold_ids),
        "p1":   sorted(p1),
        "p2":   sorted(p2),
    }


def _seg_mean(per_student: dict[str, float], ids: list[str]) -> tuple[float, int]:
    vals = [per_student[sid] for sid in ids if sid in per_student]
    if not vals:
        return 0.0, 0
    return float(np.mean(vals)), len(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "submission", type=Path,
        help="CSV на val (например submissions/val_submission_ltr_v29.csv)",
    )
    args = ap.parse_args()

    data = load_all()
    val = data["val"]
    val_solution = data["val_solution"]
    gt_raw = dict(zip(val_solution["id"], val_solution["ground_truth"]))

    rec = read_submission(args.submission)
    seg_ids = split_val_by_periods(val)

    print("=" * 78)
    print(f"Композиционно-корректная оценка: {args.submission.name}")
    print("=" * 78)
    print(f"val: cold={len(seg_ids['cold'])}, "
          f"p1={len(seg_ids['p1'])}, p2={len(seg_ids['p2'])}")

    _, per_student = evaluate_pndcg5(rec, gt_raw, k=5)

    seg_means: dict[str, float] = {}
    print("\nСегментные pNDCG@5 на val:")
    for name in ("cold", "p1", "p2"):
        m, n = _seg_mean(per_student, seg_ids[name])
        seg_means[name] = m
        print(f"  {name:<5}: n={n:>4}  pNDCG@5 = {m:.4f}")

    # p3 проксируется val_p2.
    est_correct = (
        TEST_SHARES["cold"] * seg_means["cold"]
        + TEST_SHARES["p1"] * seg_means["p1"]
        + (TEST_SHARES["p2"] + TEST_SHARES["p3"]) * seg_means["p2"]
    )

    # Наивная — как в run_ltr_v*.
    naive_warm, _ = _seg_mean(per_student, seg_ids["p1"] + seg_ids["p2"])
    est_naive = 0.352 * seg_means["cold"] + 0.648 * naive_warm

    print()
    print(f"Наивная (как в run_ltr_v*):        {est_naive:.4f}")
    print(f"Корректная (test-композиция):      {est_correct:.4f}")
    print(f"  ├─ cold  × {TEST_SHARES['cold']:.3f} = "
          f"{TEST_SHARES['cold'] * seg_means['cold']:.4f}")
    print(f"  ├─ p1    × {TEST_SHARES['p1']:.3f} = "
          f"{TEST_SHARES['p1'] * seg_means['p1']:.4f}")
    print(f"  └─ p2+p3 × {TEST_SHARES['p2'] + TEST_SHARES['p3']:.3f} = "
          f"{(TEST_SHARES['p2'] + TEST_SHARES['p3']) * seg_means['p2']:.4f}")
    print(f"\nНаивная завышает на: {est_naive - est_correct:+.4f}")

    print(
        "\nNB: val cold n=88 — оценка cold-сегмента сама по себе шумит "
        "(±0.02–0.03). Корректная оценка всё ещё оптимистична относительно "
        "public v7=0.3348, но её сдвиг между моделями уже сопоставим с "
        "реальным."
    )


if __name__ == "__main__":
    main()