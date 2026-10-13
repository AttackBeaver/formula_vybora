"""
diag_val_vs_test_available.py — почему val-оценки завышены на ~0.095.

ПРОВЕРЯЕМ ГИПОТЕЗЫ:
  (A) Состав available_modules в test отличается от val → rank-фичи
      (pop_rank_in_available, pop_rank_frac, pop_ratio_in_institute)
      считаются в разных контекстах.
  (B) Cold-val (n=88) не представляет cold-test (n=1516).
  (C) test_p3 после max_history_periods=2 структурно отличается от val_p2:
      n_choices, was_available_before, n_unique_modules.

ВЫВОД (что печатает):
  Часть 1. Сводка по сегментам: n, n_available quantiles,
           доля available, встречавшихся в train.
  Часть 2. Jaccard available-множеств: within-val / within-test / val↔test.
  Часть 3. Cold deep-dive: профиль, n_available, инсты/специальности.
  Часть 4. test_p3 (после trim) vs val_p2: n_choices, was_available_before.
  Часть 5. Сдвиг распределения топ-фич между val и test по сегментам.
  Часть 6. ВЕРДИКТ: какие гипотезы подтверждаются.

ЗАПУСК:
  python -m src.diag_val_vs_test_available
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from src.data_loader import load_all
from src.features import FeatureStats, build_dataset, candidates_available


# ============================================================
# Сегментация и утилиты
# ============================================================
def _n_periods(s: dict[str, Any]) -> int:
    return len(s.get("history") or [])


def _has_profile(s: dict[str, Any]) -> bool:
    prof = s.get("profile") or {}
    return bool(prof.get("institute") or prof.get("specialty"))


def segment_of(s: dict[str, Any]) -> str:
    n = _n_periods(s)
    if n == 0:
        return "cold" if not _has_profile(s) else "no_hist_with_prof"
    return f"p{n}"


def group_by_segment(students: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {}
    for s in students:
        buckets.setdefault(segment_of(s), []).append(s)
    return buckets


def q(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p)) if len(arr) else 0.0


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def fmt_pct(x: float) -> str:
    return f"{100 * x:.2f}%"


# ============================================================
# Часть 1: сводка по сегментам
# ============================================================
def part1_segment_summary(
    name: str,
    students: list[dict],
    train_seen: set[str],
) -> None:
    print(f"\n{'=' * 78}")
    print(f"ЧАСТЬ 1. Сводка по сегментам: {name.upper()} (n={len(students)})")
    print("=" * 78)
    groups = group_by_segment(students)
    for seg in sorted(groups):
        bucket = groups[seg]
        n_avail = np.array([len(s.get("available_modules") or []) for s in bucket])
        seen_rates: list[float] = []
        for s in bucket:
            avail = s.get("available_modules") or []
            if not avail:
                continue
            n_seen = sum(1 for m in avail if m in train_seen)
            seen_rates.append(n_seen / len(avail))
        seen_arr = np.array(seen_rates) if seen_rates else np.array([0.0])
        print(
            f"  {seg:<18} n={len(bucket):>5} | "
            f"n_avail mean={n_avail.mean():>6.1f} "
            f"p10={q(n_avail, 10):>5.0f} "
            f"med={q(n_avail, 50):>5.0f} "
            f"p90={q(n_avail, 90):>5.0f} | "
            f"seen_in_train={seen_arr.mean():.3f}"
        )


# ============================================================
# Часть 2: Jaccard available-множеств
# ============================================================
def part2_jaccard(
    val_students: list[dict],
    test_students: list[dict],
    n_pairs: int = 3000,
) -> dict[str, float]:
    print(f"\n{'=' * 78}")
    print("ЧАСТЬ 2. Jaccard available_modules (попарные сэмплы)")
    print("=" * 78)
    rng = random.Random(42)

    val_sets = [set(s.get("available_modules") or []) for s in val_students]
    test_sets = [set(s.get("available_modules") or []) for s in test_students]

    def _within(sets: list[set], k: int) -> float:
        vals = []
        for _ in range(k):
            i = rng.randrange(len(sets))
            j = rng.randrange(len(sets))
            if i == j:
                continue
            vals.append(jaccard(sets[i], sets[j]))
        return float(np.mean(vals)) if vals else 0.0

    def _cross(a: list[set], b: list[set], k: int) -> float:
        vals = []
        for _ in range(k):
            i = rng.randrange(len(a))
            j = rng.randrange(len(b))
            vals.append(jaccard(a[i], b[j]))
        return float(np.mean(vals)) if vals else 0.0

    j_vv = _within(val_sets, n_pairs)
    j_tt = _within(test_sets, n_pairs)
    j_vt = _cross(val_sets, test_sets, n_pairs)

    print(f"  within val   (попарно, k≈{n_pairs}):  {j_vv:.4f}")
    print(f"  within test  (попарно, k≈{n_pairs}):  {j_tt:.4f}")
    print(f"  val ↔ test   (k≈{n_pairs}):           {j_vt:.4f}")

    val_union = set().union(*val_sets) if val_sets else set()
    test_union = set().union(*test_sets) if test_sets else set()
    j_union = jaccard(val_union, test_union)
    print(f"\n  union val ({len(val_union)} модулей) ↔ union test "
          f"({len(test_union)} модулей): {j_union:.4f}")
    print(f"  модулей только в val: {len(val_union - test_union)}")
    print(f"  модулей только в test: {len(test_union - val_union)}")
    print(f"  общих: {len(val_union & test_union)}")

    return {"j_vv": j_vv, "j_tt": j_tt, "j_vt": j_vt, "j_union": j_union}


# ============================================================
# Часть 3: Cold deep-dive
# ============================================================
def part3_cold(
    val_groups: dict[str, list[dict]],
    test_groups: dict[str, list[dict]],
    pop_weighted_global: dict[str, float],
) -> dict[str, float]:
    print(f"\n{'=' * 78}")
    print("ЧАСТЬ 3. Cold deep-dive (val vs test)")
    print("=" * 78)

    result: dict[str, float] = {}

    for tag, groups in (("val", val_groups), ("test", test_groups)):
        cold = groups.get("cold", [])
        if not cold:
            print(f"  {tag}: cold-сегмент пустой")
            continue
        n_avail = np.array([len(s.get("available_modules") or []) for s in cold])
        has_prof = sum(1 for s in cold if _has_profile(s))
        has_hist = sum(1 for s in cold if _n_periods(s) > 0)

        print(f"\n  {tag} cold (n={len(cold)}):")
        print(f"    n_available: mean={n_avail.mean():.1f} "
              f"p10={q(n_avail, 10):.0f} med={q(n_avail, 50):.0f} "
              f"p90={q(n_avail, 90):.0f}")
        print(f"    с профилем: {has_prof} / {len(cold)} (ожидаем 0)")
        print(f"    с историей: {has_hist} / {len(cold)} (ожидаем 0)")

        insts = Counter()
        specs = Counter()
        for s in cold:
            p = s.get("profile") or {}
            insts[p.get("institute") or "NONE"] += 1
            specs[p.get("specialty") or "NONE"] += 1
        print(f"    top-3 institute: {insts.most_common(3)}")
        print(f"    top-3 specialty: {specs.most_common(3)}")

        # Доля модулей, попадающих в топ-50 глобально популярных,
        # среди доступных cold-студенту.
        top50 = set(
            m for m, _ in sorted(
                pop_weighted_global.items(), key=lambda kv: -kv[1]
            )[:50]
        )
        in_top50: list[float] = []
        for s in cold:
            avail = s.get("available_modules") or []
            if not avail:
                continue
            in_top50.append(sum(1 for m in avail if m in top50) / len(avail))
        in_top50_arr = np.array(in_top50) if in_top50 else np.array([0.0])
        print(f"    доля available ∈ top-50 global pop: "
              f"mean={in_top50_arr.mean():.3f} "
              f"med={np.median(in_top50_arr):.3f}")
        result[f"{tag}_cold_in_top50_mean"] = float(in_top50_arr.mean())

    return result


# ============================================================
# Часть 4: test_p3 (после trim) vs val_p2
# ============================================================
def part4_p3_vs_p2(
    val_groups: dict[str, list[dict]],
    test_groups: dict[str, list[dict]],
    stats: FeatureStats,
) -> dict[str, float]:
    print(f"\n{'=' * 78}")
    print("ЧАСТЬ 4. test_p3 (после trim=2) vs val_p2")
    print("=" * 78)

    def _history_stats(students: list[dict], past_periods: list[int] | None,
                       trim_to: int | None = None) -> dict[str, float]:
        """Считает n_choices / n_unique / was_available_before для среза."""
        n_choices_list: list[int] = []
        n_unique_list: list[int] = []
        was_avail_rates: list[float] = []
        n_periods_kept_list: list[int] = []
        for s in students:
            hist = list(s.get("history") or [])
            all_periods = sorted({h.get("period") for h in hist
                                  if h.get("period") is not None})
            # trim: оставляем последние N периодов
            if trim_to is not None and len(all_periods) > trim_to:
                keep = set(all_periods[-trim_to:])
                hist = [h for h in hist if h.get("period") in keep]
            # past_periods фильтр
            if past_periods is not None:
                hist = [h for h in hist if h.get("period") in past_periods]
            mods: list[str] = []
            for h in hist:
                for c in h.get("choices", []):
                    mods.append(c["module_id"])
            n_choices_list.append(len(mods))
            n_unique_list.append(len(set(mods)))
            n_periods_kept_list.append(
                len({h.get("period") for h in hist if h.get("period") is not None})
            )
            avail = s.get("available_modules") or []
            if avail:
                # original periods (не обрезанные) — для was_available_before
                orig_periods = set(all_periods)
                n_wa = 0
                for m in avail:
                    mp = stats.module_periods.get(m, set())
                    if any(p in mp for p in orig_periods):
                        n_wa += 1
                was_avail_rates.append(n_wa / len(avail))
        return {
            "n_choices_mean": float(np.mean(n_choices_list))
                if n_choices_list else 0.0,
            "n_choices_med": float(np.median(n_choices_list))
                if n_choices_list else 0.0,
            "n_unique_mean": float(np.mean(n_unique_list))
                if n_unique_list else 0.0,
            "was_avail_mean": float(np.mean(was_avail_rates))
                if was_avail_rates else 0.0,
            "n_periods_kept_med": float(np.median(n_periods_kept_list))
                if n_periods_kept_list else 0.0,
        }

    val_p2 = val_groups.get("p2", [])
    test_p3 = test_groups.get("p3", [])

    if not val_p2 or not test_p3:
        print("  Нет val_p2 или test_p3 — пропускаем.")
        return {}

    # val_p2 — как есть (past_periods=[1,2]).
    v = _history_stats(val_p2, past_periods=[1, 2], trim_to=None)
    # test_p3 — с обрезкой до 2 периодов (в features.py это max_history_periods=2).
    t = _history_stats(test_p3, past_periods=None, trim_to=2)

    print(f"  val_p2  (n={len(val_p2):>5}): "
          f"n_choices mean={v['n_choices_mean']:.2f} "
          f"med={v['n_choices_med']:.0f} | "
          f"n_unique mean={v['n_unique_mean']:.2f} | "
          f"was_available_before rate={v['was_avail_mean']:.3f}")
    print(f"  test_p3 (n={len(test_p3):>5}, после trim=2): "
          f"n_choices mean={t['n_choices_mean']:.2f} "
          f"med={t['n_choices_med']:.0f} | "
          f"n_unique mean={t['n_unique_mean']:.2f} | "
          f"was_available_before rate={t['was_avail_mean']:.3f}")

    delta_was = t["was_avail_mean"] - v["was_avail_mean"]
    delta_choices = t["n_choices_mean"] - v["n_choices_mean"]
    print(f"\n  Δ(was_available_before) = {delta_was:+.4f}")
    print(f"  Δ(n_choices mean)        = {delta_choices:+.3f}")
    return {
        "v_p2_was": v["was_avail_mean"],
        "t_p3_was": t["was_avail_mean"],
        "delta_was": delta_was,
        "delta_choices": delta_choices,
    }


# ============================================================
# Часть 5: сдвиг распределения топ-фич
# ============================================================
FEATURES_TO_CHECK: list[str] = [
    "n_periods", "n_choices", "n_unique_modules", "avg_history_priority",
    "n_available", "pop_weighted_global", "pop_by_institute",
    "pop_rank_frac", "pop_rank_in_available", "pop_ratio_in_institute",
    "cooc_score", "cooc_ppmi_norm",
    "was_available_before", "already_chosen",
]


def part5_feature_shift(
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_segments: dict[str, str],   # sid → segment (val)
    test_segments: dict[str, str],  # sid → segment (test)
) -> None:
    print(f"\n{'=' * 78}")
    print("ЧАСТЬ 5. Сдвиг распределения фич по сегментам (val → test)")
    print("=" * 78)

    val_df = val_df.copy()
    test_df = test_df.copy()
    val_df["segment"] = val_df["student_id"].map(val_segments)
    test_df["segment"] = test_df["student_id"].map(test_segments)

    # Пары: (val_segment, test_segment) — какие сопоставляем напрямую.
    pairs = [
        ("cold", "cold"),
        ("p1", "p1"),
        ("p2", "p2"),
        ("p2", "p3"),   # val_p2 как proxy для test_p3 после trim
    ]

    for feat in FEATURES_TO_CHECK:
        if feat not in val_df.columns:
            continue
        print(f"\n  {feat}")
        for vseg, tseg in pairs:
            vsub = val_df.loc[val_df["segment"] == vseg, feat].to_numpy()
            tsub = test_df.loc[test_df["segment"] == tseg, feat].to_numpy()
            if len(vsub) == 0 or len(tsub) == 0:
                continue
            # Доля ненулевых — важно для cooc / rank / already_chosen.
            v_nz = (vsub != 0).mean()
            t_nz = (tsub != 0).mean()
            print(
                f"    val_{vseg:<4} → test_{tseg:<4} | "
                f"mean {vsub.mean():>8.3f} → {tsub.mean():>8.3f} | "
                f"std {vsub.std():>7.3f} → {tsub.std():>7.3f} | "
                f"nz {fmt_pct(v_nz):>7} → {fmt_pct(t_nz):>7}"
            )


# ============================================================
# Main
# ============================================================
def main() -> None:
    print("=" * 78)
    print("Диагностика: val vs test — состав available, cold, p2/p3")
    print("=" * 78)

    data = load_all()
    train = data["train"]
    val = data["val"]
    test = data["test"]
    modules = data["modules"]

    # Множество модулей, встречавшихся в train-историях.
    train_seen: set[str] = set()
    for s in train:
        for h in s.get("history") or []:
            for c in h.get("choices", []):
                train_seen.add(c["module_id"])
    print(f"\nМодулей встречено в train-истории: {len(train_seen)} / {len(modules)}")

    val_groups = group_by_segment(val)
    test_groups = group_by_segment(test)

    # --- Части 1-2 ---
    part1_segment_summary("val", val, train_seen)
    part1_segment_summary("test", test, train_seen)
    jac = part2_jaccard(val, test)

    # --- FeatureStats нужен для части 3-5 ---
    print(f"\n{'=' * 78}")
    print("Строю FeatureStats (content_backend='none', только pop/cooc)...")
    print("=" * 78)
    stats = FeatureStats.fit(train, modules, content_backend="none")

    # --- Часть 3 ---
    part3_cold(val_groups, test_groups, stats.pop_weighted)

    # --- Часть 4 ---
    p4 = part4_p3_vs_p2(val_groups, test_groups, stats)

    # --- Часть 5 ---
    print(f"\n{'=' * 78}")
    print("Строю feature-датасеты для val (без trim) и test (trim=2)...")
    print("=" * 78)
    val_df = build_dataset(
        val, stats,
        candidates_fn=candidates_available,
        target_fn=None,
        past_periods=None,
    )
    test_df = build_dataset(
        test, stats,
        candidates_fn=candidates_available,
        target_fn=None,
        past_periods=None,
        max_history_periods=2,
    )
    print(f"  val_df: {len(val_df):,} строк")
    print(f"  test_df: {len(test_df):,} строк")

    val_segments = {s["id"]: segment_of(s) for s in val}
    test_segments = {s["id"]: segment_of(s) for s in test}
    part5_feature_shift(val_df, test_df, val_segments, test_segments)

    # --- Часть 6: вердикт ---
    print(f"\n{'=' * 78}")
    print("ЧАСТЬ 6. ВЕРДИКТ")
    print("=" * 78)
    v_cold_n = len(val_groups.get("cold", []))
    t_cold_n = len(test_groups.get("cold", []))
    cold_ratio = t_cold_n / max(1, v_cold_n)

    print(f"\n  (A) Состав available:")
    print(f"      Jaccard(union val, union test)   = {jac['j_union']:.4f}")
    print(f"      Jaccard(val↔test, попарно)       = {jac['j_vt']:.4f}")
    if jac["j_vt"] < 0.6:
        print(f"      → ПОДТВЕРЖДАЕТСЯ: available в val и test сильно различны.")
        print(f"        Rank-фичи (pop_rank_in_available/frac) в разных контекстах.")
    else:
        print(f"      → Слабо подтверждается (j_vt ≥ 0.6).")

    print(f"\n  (B) Cold-сегмент:")
    print(f"      val cold n={v_cold_n}, test cold n={t_cold_n} "
          f"(в {cold_ratio:.1f}× больше)")
    if t_cold_n >= 500 and v_cold_n <= 200:
        print(f"      → ПОДТВЕРЖДАЕТСЯ: val cold не представляет test cold.")
        print(f"        Оценка cold-качества по val (n={v_cold_n}) нестабильна.")
    else:
        print(f"      → Не критично.")

    print(f"\n  (C) test_p3 после trim:")
    if p4:
        print(f"      Δ(was_available_before) = {p4['delta_was']:+.4f}")
        print(f"      Δ(n_choices)            = {p4['delta_choices']:+.3f}")
        if abs(p4["delta_was"]) > 0.05 or abs(p4["delta_choices"]) > 1.0:
            print(f"      → ПОДТВЕРЖДАЕТСЯ: test_p3 (trim=2) ≠ val_p2.")
        else:
            print(f"      → Структурно совпадают (Δ малы).")

    print(f"\n[OK] Диагностика завершена.")


if __name__ == "__main__":
    main()