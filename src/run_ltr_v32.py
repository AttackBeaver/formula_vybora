"""
run_ltr_v32.py — v7 (warm) + cold-optimized ranking.

ИДЕЯ:
    Cold = 35% test, нет истории, нет профиля, available_modules одинаков
    для всех. Вся cold-рекомендация — ОДИН глобальный ranking 158 модулей.
    Значит, его можно оптимизировать НАПРЯМУЮ по метрике (greedy по
    позициям), а не через LGBM.

    В v7 cold использует pop_weighted_global (fallback). Это доминируется
    P2-популярностью. Для cold логичнее P1-популярность (их первый выбор)
    или прямая оптимизация pNDCG.

ЧТО СРАВНИВАЕМ (все на val cold n=88):
    (a) pop_weighted_global      — baseline = v7 fallback
    (b) pop_by_period[1] только  — train-P1 popularity
    (c) greedy по train p0       — прямая оптимизация метрики на P1 targets

    Greedy — жадно: на позиции p берём модуль, максимизирующий
    Σ_s weight(target_s[m]) / log2(p+1) по train p0 target'ам.

ПРАВИЛА:
    Warm (p1/p2/p3) не трогаем — v7 проверен.
    Если лучший ranking даёт на val cold ≥ baseline + 0.015 → применяем
    к test cold, заменяем cold-строки в v7 submissions → v32.

ЗАПУСК:
    python -m src.run_ltr_v32           # только диагностика
    python -m src.run_ltr_v32 --apply    # + сохранить submission_v32.csv
"""
from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.cold_start import identify_cold_students
from src.data_loader import load_all
from src.metric import evaluate_pndcg5, parse_ground_truth
from src.popularity import compute_popularity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

PRIORITY_WEIGHTS = {1: 1.00, 2: 0.80, 3: 0.60, 4: 0.45, 5: 0.35}


# ---------------------------------------------------------------------------
# I/O submission
# ---------------------------------------------------------------------------
def read_submission(path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(path)
    id_col = "id" if "id" in df.columns else "student_id"
    if "recommendations" in df.columns:
        col = "recommendations"
    elif "prediction" in df.columns:
        col = "prediction"
    else:
        raise ValueError(f"Неизвестный формат: {list(df.columns)}")
    rec: dict[str, list[str]] = {}
    for sid, raw in zip(df[id_col], df[col]):
        if pd.isna(raw):
            rec[str(sid)] = []
        else:
            s = str(raw).replace(",", " ").replace(";", " ").replace("|", " ")
            rec[str(sid)] = [t for t in s.split() if t]
    return rec


def write_submission(rec: dict[str, list[str]], path: Path) -> None:
    df = pd.DataFrame({
        "id": list(rec.keys()),
        "recommendations": [" ".join(mods) for mods in rec.values()],
    })
    df.to_csv(path, index=False)
    print(f"      Сохранено: {path}  ({len(df)} строк)")


# ---------------------------------------------------------------------------
# Работа с таргетами
# ---------------------------------------------------------------------------
def target_weight(target: dict[str, int], module_id: str) -> float:
    prio = target.get(module_id, 0)
    if prio <= 0:
        return 0.0
    return PRIORITY_WEIGHTS.get(prio, 0.0)


def score_ranking_on_targets(
    ranking: list[str],
    targets: list[dict[str, int]],
    avails: list[set[str]] | None = None,
) -> float:
    """Средний pNDCG@5 ranking'а по target'ам (с учётом available, если задан)."""
    if not targets:
        return 0.0
    scores = []
    for i, tgt in enumerate(targets):
        if avails is not None:
            avail = avails[i]
            r = [m for m in ranking if m in avail][:5]
        else:
            r = ranking[:5]
        dcg = 0.0
        for pos, m in enumerate(r, start=1):
            w = target_weight(tgt, m)
            if w > 0:
                dcg += w / math.log2(pos + 1)
        scores.append(min(1.0, dcg))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Стратегии построения ranking
# ---------------------------------------------------------------------------
def ranking_global_pop(train: list[dict], pool: set[str]) -> list[str]:
    """Baseline v7: pop_weighted_global."""
    pop_weighted, _ = compute_popularity(train)
    return sorted(pool, key=lambda m: (-pop_weighted.get(m, 0.0), m))


def ranking_p1_popularity(train: list[dict], pool: set[str]) -> list[str]:
    """Популярность модуля в P1 (первый выбор)."""
    counts: dict[str, float] = defaultdict(float)
    for s in train:
        for h in s.get("history", []):
            if h.get("period") != 1:
                continue
            for c in h.get("choices", []):
                counts[c["module_id"]] += PRIORITY_WEIGHTS.get(c.get("priority", 5), 0.0)
    return sorted(pool, key=lambda m: (-counts.get(m, 0.0), m))


def ranking_greedy(
    train: list[dict],
    pool: set[str],
    k: int = 5,
) -> list[str]:
    """
    Жадная оптимизация pNDCG@5 по train P1-таргетам.

    На позиции p выбираем модуль m, максимизирующий:
        Σ_s weight(target_s[m]) / log2(p+1)
    """
    # Собираем таргеты train-студентов по P1.
    p1_targets: list[dict[str, int]] = []
    for s in train:
        for h in s.get("history", []):
            if h.get("period") == 1:
                tgt = {c["module_id"]: c.get("priority", 5)
                       for c in h.get("choices", [])}
                p1_targets.append(tgt)
                break
    print(f"      [greedy] train P1 targets: {len(p1_targets)}")

    ranked: list[str] = []
    remaining = set(pool)
    for pos in range(k):
        discount = 1.0 / math.log2(pos + 2)
        best_m = None
        best_score = -1.0
        for m in remaining:
            s = 0.0
            for tgt in p1_targets:
                w = target_weight(tgt, m)
                if w > 0:
                    s += w
            s *= discount
            if s > best_score:
                best_score = s
                best_m = m
        if best_m is None:
            break
        ranked.append(best_m)
        remaining.discard(best_m)
        print(f"      [greedy] pos={pos + 1}: {best_m}  "
              f"(expected contrib = {best_score:.4f})")
    return ranked


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="v32: cold-optimized ranking")
    ap.add_argument("--apply", action="store_true",
                    help="Применить лучший ranking и сохранить submission_v32.csv")
    ap.add_argument("--min-gain", type=float, default=0.015,
                    help="Минимальный прирост над baseline для применения")
    args = ap.parse_args()

    print("=" * 78)
    print("v32: cold-optimized ranking поверх v7 (warm не трогаем)")
    print("=" * 78)

    data = load_all()
    train = data["train"]
    val = data["val"]
    test = data["test"]
    modules = data["modules"]
    val_solution = data["val_solution"]
    val_gt = dict(zip(val_solution["id"], val_solution["ground_truth"]))

    # --- Available-пулы ---
    val_avail = set(val[0]["available_modules"])
    test_avail = set(test[0]["available_modules"])
    print(f"\nval avail: {len(val_avail)} модулей, test avail: {len(test_avail)}")
    print(f"Jaccard(val_avail, test_avail) = "
          f"{len(val_avail & test_avail) / len(val_avail | test_avail):.4f}")
    print(f"test-модулей нет в val: {len(test_avail - val_avail)}")

    # --- Cold-студенты ---
    val_cold_ids = identify_cold_students(val)
    test_cold_ids = identify_cold_students(test)
    print(f"\nval cold: {len(val_cold_ids)}, test cold: {len(test_cold_ids)}")

    val_cold_students = [s for s in val if s["id"] in val_cold_ids]
    test_cold_students = [s for s in test if s["id"] in test_cold_ids]

    # --- Val cold targets и avails (для оценки) ---
    val_cold_targets: list[dict[str, int]] = []
    val_cold_avails: list[set[str]] = []
    for s in val_cold_students:
        raw = val_gt.get(s["id"])
        if not raw:
            continue
        val_cold_targets.append(parse_ground_truth(raw))
        val_cold_avails.append(set(s.get("available_modules") or []))

    print(f"val cold с непустым target: {len(val_cold_targets)}")

    # Показать распределение target размеров
    tgt_sizes = [len(t) for t in val_cold_targets]
    print(f"  target size: mean={np.mean(tgt_sizes):.2f}, "
          f"med={np.median(tgt_sizes):.0f}, "
          f"min={min(tgt_sizes)}, max={max(tgt_sizes)}")

    # Кол-во cold с priority-1 в target
    n_with_p1 = sum(1 for t in val_cold_targets if any(p == 1 for p in t.values()))
    print(f"  cold с priority-1 в target: {n_with_p1}/{len(val_cold_targets)}")

    # ============================================================
    # Стратегии
    # ============================================================
    print("\n" + "=" * 78)
    print("Стратегии cold-ranking (все строятся по train, кроме baseline)")
    print("=" * 78)

    print("\n(a) global pop (baseline v7)...")
    r_a = ranking_global_pop(train, test_avail)
    s_a = score_ranking_on_targets(r_a, val_cold_targets, val_cold_avails)
    print(f"      val cold pNDCG = {s_a:.4f}")
    print(f"      top-5: {r_a[:5]}")

    print("\n(b) P1 popularity (только train period 1)...")
    r_b = ranking_p1_popularity(train, test_avail)
    s_b = score_ranking_on_targets(r_b, val_cold_targets, val_cold_avails)
    print(f"      val cold pNDCG = {s_b:.4f}  (Δ vs a: {s_b - s_a:+.4f})")
    print(f"      top-5: {r_b[:5]}")

    print("\n(c) greedy по train P1 targets (прямая оптимизация метрики)...")
    r_c = ranking_greedy(train, test_avail, k=5)
    s_c = score_ranking_on_targets(r_c, val_cold_targets, val_cold_avails)
    print(f"      val cold pNDCG = {s_c:.4f}  (Δ vs a: {s_c - s_a:+.4f})")
    print(f"      top-5: {r_c[:5]}")

    # --- Блендинг (a)+(c): первые 5 из c, остальное — для добивки ---
    # Для холодных top-5 достаточно; блендинг не нужен.

    # ============================================================
    # Итог
    # ============================================================
    print("\n" + "=" * 78)
    print("ИТОГ")
    print("=" * 78)
    candidates = [
        ("(a) global pop", r_a, s_a),
        ("(b) P1 popularity", r_b, s_b),
        ("(c) greedy-train", r_c, s_c),
    ]
    best_name, best_ranking, best_score = max(candidates, key=lambda kv: kv[2])
    print(f"\nBaseline (a): {s_a:.4f}")
    print(f"Лучший:       {best_name}: {best_score:.4f}  "
          f"(Δ = {best_score - s_a:+.4f})")
    print(f"\nПорог для применения: Δ ≥ {args.min_gain:.4f}")

    if best_score - s_a < args.min_gain:
        print(f"\n>>> НЕ ПРИМЕНЯЕМ: прирост {best_score - s_a:+.4f} "
              f"меньше порога {args.min_gain:.4f}.")
        print(f">>> Ничего не сабмитим. Возвращаемся к v7 = 0.3348.")
        return

    print(f"\n>>> ПРИМЕНЯЕМ: {best_name} (Δ = {best_score - s_a:+.4f}).")

    if not args.apply:
        print(f"\n[DRY-RUN] Флаг --apply не указан — submission не пишем.")
        print(f"          Запустите: python -m src.run_ltr_v32 --apply")
        return

    # ============================================================
    # Применяем: заменяем cold-строки в v7 submissions
    # ============================================================
    v7_sub_path = SUBMISSIONS_DIR / "submission_v7.csv"
    v7_val_path = SUBMISSIONS_DIR / "val_submission_ltr_v7.csv"

    if not v7_sub_path.exists() or not v7_val_path.exists():
        print(f"\n!!! Не найдены v7 submissions:")
        print(f"    {v7_sub_path} exists={v7_sub_path.exists()}")
        print(f"    {v7_val_path} exists={v7_val_path.exists()}")
        print(f"    Сначала запустите: python -m src.run_ltr_v7")
        return

    print(f"\nЧитаем v7 submissions...")
    v7_test_rec = read_submission(v7_sub_path)
    v7_val_rec = read_submission(v7_val_path)
    print(f"  test: {len(v7_test_rec)} студентов")
    print(f"  val:  {len(v7_val_rec)} студентов")

    # Заменяем cold
    for sid in test_cold_ids:
        v7_test_rec[sid] = best_ranking[:5]
    for sid in val_cold_ids:
        v7_val_rec[sid] = best_ranking[:5]

    test_path = SUBMISSIONS_DIR / "submission_v32.csv"
    val_path = SUBMISSIONS_DIR / "val_submission_ltr_v32.csv"
    write_submission(v7_test_rec, test_path)
    write_submission(v7_val_rec, val_path)

    # Проверка на val
    _, per_student = evaluate_pndcg5(v7_val_rec, val_gt, k=5)
    cold_scores = [per_student[sid] for sid in val_cold_ids if sid in per_student]
    warm_scores = [per_student[sid] for sid in per_student
                   if sid not in val_cold_ids]
    m_cold = float(np.mean(cold_scores))
    m_warm = float(np.mean(warm_scores))
    composite = 0.352 * m_cold + 0.648 * m_warm
    print(f"\nФинальная оценка v32 (на val):")
    print(f"  cold   = {m_cold:.4f}")
    print(f"  warm   = {m_warm:.4f}")
    print(f"  composite (0.352·cold + 0.648·warm) = {composite:.4f}")

    print(f"\n[OK] v32 готов. Файлы:")
    print(f"     {test_path}")
    print(f"     {val_path}")


if __name__ == "__main__":
    main()