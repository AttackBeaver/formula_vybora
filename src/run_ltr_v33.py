"""
run_ltr_v33.py — cold-ranking из val cold targets поверх v7.

ИДЕЯ:
    У нас есть 88 val cold студентов с известными таргетами. Если
    val cold ~ test cold по распределению предпочтений, то оптимальный
    ФИКСИРОВАННЫЙ top-5 для холодных = top-5 модулей по сумме весов
    приоритетов среди val cold targets.

    Оптимальность: метрика pNDCG@5 линейна по позициям, позиционные
    веса 1/log2(p+1) не зависят от модуля. Значит сортировка модулей
    по Σ_s weight(target_s[m]) — точный оптимум для top-5.

    Оценка честности — 5-fold CV по val cold:
      строим ranking на 4/5 (70 студентов), оцениваем на 1/5 (18).
      Это даёт несмещённую оценку переноса.

ЧТО ПРОВЕРЯЕМ:
    (a) global pop       — baseline (то, что v7 использует сейчас)
    (b) val cold pop     — Σ weight по val cold targets
    (c) train P1 pop     — Σ weight по train P1 targets (диагностика)
    (d) blend 0.8·b + 0.2·c
    (e) blend 0.8·b + 0.2·a

ПРАВИЛА:
    Если CV(best) - CV(a) ≥ 0.02 → применяем к test cold, заменяем
    cold-строки в v7 submissions → submission_v33.csv.
    Иначе не тратим сабмит.

ЗАПУСК:
    python -m src.run_ltr_v33              # только CV
    python -m src.run_ltr_v33 --apply      # + сохранить submission_v33.csv
"""
from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

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
# I/O
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
        "recommendations": [" ".join(m) for m in rec.values()],
    })
    df.to_csv(path, index=False)
    print(f"    Сохранено: {path} ({len(df)} строк)")


# ---------------------------------------------------------------------------
# Ядро: ranking + оценка
# ---------------------------------------------------------------------------
def weight_sum_ranking(targets: list[dict[str, int]],
                       pool: set[str] | None = None) -> list[str]:
    """Ranking по Σ_s weight(target_s[m]). Точный оптимум top-5."""
    counts: dict[str, float] = defaultdict(float)
    for tgt in targets:
        for m, prio in tgt.items():
            counts[m] += PRIORITY_WEIGHTS.get(prio, 0.0)
    modules = list(counts.keys())
    if pool is not None:
        modules = [m for m in modules if m in pool]
    modules.sort(key=lambda m: (-counts[m], m))
    return modules


def evaluate_ranking(ranking: list[str],
                     targets: list[dict[str, int]],
                     avails: list[set[str]],
                     k: int = 5) -> float:
    scores = []
    for tgt, avail in zip(targets, avails):
        r = [m for m in ranking if m in avail][:k]
        dcg = 0.0
        for pos, m in enumerate(r, 1):
            w = PRIORITY_WEIGHTS.get(tgt.get(m, 0), 0.0)
            if w > 0:
                dcg += w / math.log2(pos + 1)
        scores.append(min(1.0, dcg))
    return float(np.mean(scores)) if scores else 0.0


def cv_evaluate(strategy_fn,
                targets: list[dict[str, int]],
                avails: list[set[str]],
                pool: set[str],
                n_folds: int = 5,
                seed: int = 42,
                k: int = 5) -> tuple[float, float]:
    n = len(targets)
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    fold_size = n // n_folds
    scores = []
    for f in range(n_folds):
        start = f * fold_size
        end = start + fold_size if f < n_folds - 1 else n
        hold_idx = idx[start:end]
        train_idx = [i for i in idx if i not in hold_idx]
        train_targets = [targets[i] for i in train_idx]
        hold_targets = [targets[i] for i in hold_idx]
        hold_avails = [avails[i] for i in hold_idx]
        ranking = strategy_fn(train_targets, pool)
        scores.append(evaluate_ranking(ranking, hold_targets, hold_avails, k=k))
    return float(np.mean(scores)), float(np.std(scores, ddof=1))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="v33: cold-ranking из val cold targets")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--min-gain", type=float, default=0.02)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    print("=" * 78)
    print("v33: cold-ranking из val cold targets (с 5-fold CV)")
    print("=" * 78)

    data = load_all()
    train = data["train"]
    val = data["val"]
    test = data["test"]
    val_solution = data["val_solution"]
    val_gt = dict(zip(val_solution["id"], val_solution["ground_truth"]))

    val_cold_ids = identify_cold_students(val)
    test_cold_ids = identify_cold_students(test)
    val_cold_students = [s for s in val if s["id"] in val_cold_ids]
    test_cold_students = [s for s in test if s["id"] in test_cold_ids]
    print(f"\nval cold: {len(val_cold_students)}, test cold: {len(test_cold_students)}")

    val_avail = set(val[0]["available_modules"])
    test_avail = set(test[0]["available_modules"])
    common_avail = val_avail & test_avail
    print(f"val_avail: {len(val_avail)}, test_avail: {len(test_avail)}, "
          f"common: {len(common_avail)}")

    # --- Таргеты val cold ---
    val_cold_targets: list[dict[str, int]] = []
    val_cold_avails: list[set[str]] = []
    for s in val_cold_students:
        raw = val_gt.get(s["id"])
        if not raw:
            continue
        val_cold_targets.append(parse_ground_truth(raw))
        val_cold_avails.append(set(s.get("available_modules") or []))
    print(f"val cold с таргетами: {len(val_cold_targets)}")

    # --- Train P1 targets ---
    train_p1_targets: list[dict[str, int]] = []
    for s in train:
        for h in s.get("history", []):
            if h.get("period") == 1:
                train_p1_targets.append({
                    c["module_id"]: c.get("priority", 5)
                    for c in h.get("choices", [])
                })
                break
    print(f"train P1 targets: {len(train_p1_targets)}")

    # --- Global pop ---
    pop_weighted, _ = compute_popularity(train)

    # --- Диагностика: топ модулей в val cold ---
    print(f"\nТоп-10 модулей по Σ weight в val cold targets:")
    agg: dict[str, float] = defaultdict(float)
    for t in val_cold_targets:
        for m, p in t.items():
            agg[m] += PRIORITY_WEIGHTS.get(p, 0.0)
    for m, w in sorted(agg.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {m}: total_weight = {w:.2f}")

    # ============================================================
    # Стратегии
    # ============================================================
    def strat_global_pop(_train_targets, pool):
        return sorted(pool, key=lambda m: (-pop_weighted.get(m, 0.0), m))

    def strat_val_cold(train_targets, pool):
        return weight_sum_ranking(train_targets, pool=pool)

    def strat_train_p1(_train_targets, pool):
        return weight_sum_ranking(train_p1_targets, pool=pool)

    def strat_blend_vc_tp1(train_targets, pool):
        vc = defaultdict(float)
        for t in train_targets:
            for m, p in t.items():
                vc[m] += PRIORITY_WEIGHTS.get(p, 0.0)
        tp = defaultdict(float)
        for t in train_p1_targets:
            for m, p in t.items():
                tp[m] += PRIORITY_WEIGHTS.get(p, 0.0)
        vc_s = sum(vc.values()) or 1.0
        tp_s = sum(tp.values()) or 1.0
        blend = {m: 0.8 * vc[m] / vc_s + 0.2 * tp[m] / tp_s for m in pool}
        return sorted(pool, key=lambda m: (-blend[m], m))

    def strat_blend_vc_gp(train_targets, pool):
        vc = defaultdict(float)
        for t in train_targets:
            for m, p in t.items():
                vc[m] += PRIORITY_WEIGHTS.get(p, 0.0)
        vc_s = sum(vc.values()) or 1.0
        gp_max = max(pop_weighted.values()) if pop_weighted else 1.0
        blend = {m: 0.8 * vc.get(m, 0.0) / vc_s
                 + 0.2 * pop_weighted.get(m, 0.0) / gp_max for m in pool}
        return sorted(pool, key=lambda m: (-blend[m], m))

    strategies = [
        ("(a) global pop (baseline v7)", strat_global_pop),
        ("(b) val-cold popularity", strat_val_cold),
        ("(c) train-P1 popularity", strat_train_p1),
        ("(d) blend 0.8·valcold + 0.2·trainp1", strat_blend_vc_tp1),
        ("(e) blend 0.8·valcold + 0.2·globalpop", strat_blend_vc_gp),
    ]

    # ============================================================
    # CV
    # ============================================================
    print("\n" + "=" * 78)
    print(f"5-fold CV на val cold (pool = common_avail, {len(common_avail)} модулей)")
    print("=" * 78)

    results: list[tuple[str, float, float, callable]] = []
    for name, fn in strategies:
        mean, std = cv_evaluate(
            fn, val_cold_targets, val_cold_avails,
            pool=common_avail, n_folds=args.n_folds,
        )
        results.append((name, mean, std, fn))
        print(f"  {name:<38}  CV = {mean:.4f} ± {std:.4f}")

    baseline_name, baseline_mean, _, _ = results[0]
    best_name, best_mean, _, best_fn = max(results, key=lambda x: x[1])
    delta = best_mean - baseline_mean

    print(f"\nBaseline: {baseline_name}: {baseline_mean:.4f}")
    print(f"Лучший:   {best_name}: {best_mean:.4f}")
    print(f"Δ = {delta:+.4f}   (порог: {args.min_gain:+.4f})")

    if delta < args.min_gain:
        print(f"\n>>> НЕ ПРИМЕНЯЕМ: прирост {delta:+.4f} < {args.min_gain:.4f}.")
        print(f">>> Ничего не сабмитим.")
        return

    print(f"\n>>> ПРИМЕНЯЕМ: {best_name}  (Δ = {delta:+.4f}).")

    if not args.apply:
        print(f"\n[DRY-RUN] Без --apply ничего не пишем.")
        return

    # ============================================================
    # Применение: заменяем cold в v7 submission
    # ============================================================
    v7_test_path = SUBMISSIONS_DIR / "submission_v7.csv"
    v7_val_path = SUBMISSIONS_DIR / "val_submission_ltr_v7.csv"
    if not v7_test_path.exists() or not v7_val_path.exists():
        print(f"\n!!! Нет v7 submissions. Сначала: python -m src.run_ltr_v7")
        return

    print(f"\nЧитаем v7 submissions...")
    v7_test = read_submission(v7_test_path)
    v7_val = read_submission(v7_val_path)

    # Финальный ranking — на ВСЕХ val cold, pool = test_avail
    final_ranking = best_fn(val_cold_targets, test_avail)
    print(f"Cold top-5 (pool=test_avail): {final_ranking[:5]}")

    for sid in test_cold_ids:
        v7_test[sid] = final_ranking[:5]
    for sid in val_cold_ids:
        v7_val[sid] = final_ranking[:5]

    test_out = SUBMISSIONS_DIR / "submission_v33.csv"
    val_out = SUBMISSIONS_DIR / "val_submission_ltr_v33.csv"
    write_submission(v7_test, test_out)
    write_submission(v7_val, val_out)

    # Оценка на val
    _, per_student = evaluate_pndcg5(v7_val, val_gt, k=5)
    cold_scores = [per_student[sid] for sid in val_cold_ids if sid in per_student]
    warm_scores = [per_student[sid] for sid in per_student
                   if sid not in val_cold_ids]
    m_cold = float(np.mean(cold_scores))
    m_warm = float(np.mean(warm_scores))
    composite = 0.352 * m_cold + 0.648 * m_warm
    print(f"\nФинальная оценка v33 (на val, in-sample для cold):")
    print(f"  cold = {m_cold:.4f}  (in-sample — CV дал {best_mean:.4f})")
    print(f"  warm = {m_warm:.4f}")
    print(f"  composite (в val) = {composite:.4f}")

    print(f"\n[OK] v33 готов. Кандидат на сабмит: {test_out}")


if __name__ == "__main__":
    main()