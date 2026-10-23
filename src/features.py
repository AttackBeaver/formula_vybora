"""
Feature engineering для LTR-модели: S (студент), M (модуль), S×M (взаимодействия).

Все агрегаты считаются один раз на train (fit) и переиспользуются для
val/test (transform) — без training-serving skew.

build_dataset всегда добавляет `label` (для test — target_fn=None → label=0),
единый интерфейс без ветвлений.

mask_profile / mask_student_ids — обучение на cold-подобных примерах.
ВНИМАНИЕ: p0_mask (mask_profile=True + past_periods=[]) НЕ ПЕРЕНОСИТСЯ на
test cold (v43: cold=0.028 на public против 0.34 на val). Не использовать
в production для cold.

Относительные периоды (pop_last/prev/first_period) устраняют сдвиг между
train (история до P2) и test (до P3/P4). max_history_periods обрезает
историю до последних N и перенумеровывает к (1, 2, ...).

content_backend: "tfidf" | "embed" | "both" | "none".
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.content import ContentModel
from src.content_embed import ContentEmbedModel

PRIORITY_WEIGHTS: dict[int, float] = {
    1: 1.00, 2: 0.80, 3: 0.60, 4: 0.45, 5: 0.35,
}
DEFAULT_INSTITUTE = "UNKNOWN"
DEFAULT_SPECIALTY = "UNKNOWN"

VALID_CONTENT_BACKENDS = ("tfidf", "embed", "both", "none")


class FeatureStats:
    """Агрегаты, посчитанные один раз на train; для val/test не пересчитываются."""

    def __init__(self, content_backend: str = "tfidf") -> None:
        if content_backend not in VALID_CONTENT_BACKENDS:
            raise ValueError(
                f"content_backend={content_backend!r}, ожидалось одно из "
                f"{VALID_CONTENT_BACKENDS}"
            )
        self.content_backend: str = content_backend

        self.pop_weighted: dict[str, float] = {}
        self.pop_raw: dict[str, int] = {}
        self.pop_by_institute: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.pop_by_specialty: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.pop_by_period: dict[int, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.pop_rank_global: dict[str, int] = {}
        self.n_modules_global: int = 0
        self.cooc: dict[str, Counter] = defaultdict(Counter)
        self.module_doc_freq: dict[str, int] = {}
        self.n_students_total: int = 0
        self.module_title_len: dict[str, int] = {}
        self.module_desc_len: dict[str, int] = {}
        self.module_periods: dict[str, set[int]] = {}
        self.content_model_tfidf: ContentModel | None = None
        self.content_model_embed: ContentEmbedModel | None = None
        self.median_pop_weighted: float = 0.0
        self._has_period_3: bool = False

    @classmethod
    def fit(
        cls,
        train: list[dict[str, Any]],
        modules: list[dict[str, Any]],
        content_backend: str = "tfidf",
    ) -> "FeatureStats":
        stats = cls(content_backend=content_backend)

        # 1. Метаданные модулей
        for m in modules:
            mid = m["module_id"]
            stats.module_title_len[mid] = len(m.get("title") or "")
            stats.module_desc_len[mid] = len(m.get("description") or "")
            stats.module_periods[mid] = set(m.get("available_in_periods") or [])
        stats.n_modules_global = len(stats.module_periods)

        # 2. Content-модели
        if content_backend in ("tfidf", "both"):
            print(f"[FeatureStats] TF-IDF: fit на {len(modules)} модулях...")
            stats.content_model_tfidf = ContentModel.fit(modules)
        if content_backend in ("embed", "both"):
            print(f"[FeatureStats] Embeddings: fit на {len(modules)} модулях...")
            stats.content_model_embed = ContentEmbedModel.fit(modules)

        # 3. Популярность и co-occurrence
        stats.n_students_total = len(train)
        for student in train:
            inst = (student.get("profile") or {}).get("institute") or DEFAULT_INSTITUTE
            spec = (student.get("profile") or {}).get("specialty") or DEFAULT_SPECIALTY
            all_modules_of_student: list[str] = []
            for h in student.get("history", []):
                period = h.get("period")
                for choice in h.get("choices", []):
                    mid = choice["module_id"]
                    prio = choice.get("priority", 5)
                    w = PRIORITY_WEIGHTS.get(prio, 0.0)
                    stats.pop_weighted[mid] = stats.pop_weighted.get(mid, 0.0) + w
                    stats.pop_raw[mid] = stats.pop_raw.get(mid, 0) + 1
                    stats.pop_by_institute[inst][mid] += w
                    stats.pop_by_specialty[spec][mid] += w
                    stats.pop_by_period[period][mid] += w
                    all_modules_of_student.append(mid)

            unique_modules = sorted(set(all_modules_of_student))
            for mid in unique_modules:
                stats.module_doc_freq[mid] = stats.module_doc_freq.get(mid, 0) + 1

            for i, m1 in enumerate(unique_modules):
                for m2 in unique_modules[i + 1:]:
                    stats.cooc[m1][m2] += 1
                    stats.cooc[m2][m1] += 1

        if stats.pop_weighted:
            stats.median_pop_weighted = float(
                np.median(list(stats.pop_weighted.values()))
            )

        # 4. Глобальный ранг
        sorted_modules = sorted(
            stats.module_periods.keys(),
            key=lambda m: (-stats.pop_weighted.get(m, 0.0), m),
        )
        stats.pop_rank_global = {m: r for r, m in enumerate(sorted_modules)}
        return stats

    def add_val_solution_popularity(self, val_solution: pd.DataFrame) -> None:
        """
        Добавляет pop_by_period[3] из val_solution (схема v7: p2-срез для warm).
        pop_rank_global НЕ пересчитывает.
        """
        n_rows = 0
        for _, row in val_solution.iterrows():
            gt_str = str(row["ground_truth"]).strip()
            if gt_str.startswith('"') and gt_str.endswith('"'):
                gt_str = gt_str[1:-1]
            for tok in gt_str.split():
                if ":" not in tok:
                    continue
                mid, prio_s = tok.split(":", 1)
                try:
                    prio = int(prio_s)
                except ValueError:
                    continue
                w = PRIORITY_WEIGHTS.get(prio, 0.0)
                self.pop_by_period[3][mid] += w
            n_rows += 1
        self._has_period_3 = True
        print(f"[FeatureStats] pop_by_period[3] построен из val_solution: "
              f"{n_rows} студентов, {len(self.pop_by_period[3])} модулей")

    def ppmi(self, a: str, b: str) -> float:
        """PPMI между модулями: max(0, log( P(a,b) / (P(a)·P(b)) ))."""
        cooc_ab = self.cooc.get(a, Counter()).get(b, 0)
        if cooc_ab == 0:
            return 0.0
        freq_a = self.module_doc_freq.get(a, 0)
        freq_b = self.module_doc_freq.get(b, 0)
        if freq_a <= 0 or freq_b <= 0:
            return 0.0
        pmi = math.log(cooc_ab * self.n_students_total / (freq_a * freq_b))
        return max(0.0, pmi)


def _safe_get(d: dict[str, Any] | None, key: str, default: str = "UNKNOWN") -> str:
    if not d:
        return default
    return d.get(key) or default


def build_dataset(
    students: list[dict[str, Any]],
    stats: FeatureStats,
    candidates_fn: Callable[[dict[str, Any]], list[str]],
    target_fn: Callable[[dict[str, Any]], dict[str, int]] | None = None,
    past_periods: list[int] | None = None,
    mask_profile: bool = False,
    mask_student_ids: set[str] | None = None,
    max_history_periods: int | None = None,
) -> pd.DataFrame:
    """
    Колонки: student_id, module_id, S/M/S×M признаки,
    tfidf_cos_mean/max, embed_cos_mean/max,
    pop_rank_global / _in_avail / _frac,
    pop_delta / _norm / _rank, priority, label.
    """
    rows: list[dict[str, Any]] = []
    tfidf_model = stats.content_model_tfidf
    embed_model = stats.content_model_embed
    use_id_filter = mask_student_ids is not None and len(mask_student_ids) > 0
    n_modules_global = stats.n_modules_global or 1

    for student in students:
        sid = student.get("id") or student.get("student_id")
        if sid is None:
            continue

        if use_id_filter:
            do_mask = sid in mask_student_ids  # type: ignore[operator]
        else:
            do_mask = mask_profile

        if do_mask:
            profile: dict[str, Any] = {}
            history_raw: list[dict[str, Any]] = []
        else:
            profile = student.get("profile") or {}
            history_raw = student.get("history", []) or []

        # Обрезка истории: последние N периодов, перенумерация 1..N.
        original_history_periods: set[int] = set()
        if max_history_periods is not None and history_raw:
            all_periods = sorted({
                h.get("period") for h in history_raw
                if h.get("period") is not None
            })
            original_history_periods = set(all_periods)
            if len(all_periods) > max_history_periods:
                keep_periods = set(all_periods[-max_history_periods:])
                filtered_history = [
                    h for h in history_raw if h.get("period") in keep_periods
                ]
                unique_kept_periods = sorted({
                    h.get("period") for h in filtered_history
                    if h.get("period") is not None
                })
                period_mapping = {
                    p: i + 1 for i, p in enumerate(unique_kept_periods)
                }
                history_raw = []
                for h in filtered_history:
                    new_h = dict(h)
                    new_h["period"] = period_mapping[h["period"]]
                    history_raw.append(new_h)
        else:
            original_history_periods = {
                h.get("period") for h in history_raw
                if h.get("period") is not None
            }

        inst = _safe_get(profile, "institute")
        spec = _safe_get(profile, "specialty")

        history_modules: list[str] = []
        history_priorities: list[int] = []
        history_periods_used: list[int] = []
        for h in history_raw:
            period = h.get("period")
            if past_periods is not None and period not in past_periods:
                continue
            history_periods_used.append(period)
            for c in h.get("choices", []):
                history_modules.append(c["module_id"])
                history_priorities.append(c.get("priority", 5))

        history_set = set(history_modules)
        n_periods = len(history_periods_used)
        n_choices = len(history_modules)
        is_cold = (
            n_choices == 0
            and not profile.get("institute")
            and not profile.get("specialty")
        )

        # Относительные периоды: last / prev / first.
        unique_periods = sorted(set(history_periods_used))
        if len(unique_periods) == 0:
            last_p: int | None = None
            prev_p: int | None = None
            first_p: int | None = None
        elif len(unique_periods) == 1:
            last_p = unique_periods[0]
            prev_p = None
            first_p = unique_periods[0]
        else:
            first_p = unique_periods[0]
            last_p = unique_periods[-1]
            prev_p = unique_periods[-2]

        def _pop_map(period: int | None) -> dict[str, float]:
            if period is None:
                return {}
            return stats.pop_by_period.get(period, {})  # type: ignore[return-value]

        pop_last_map = _pop_map(last_p)
        pop_prev_map = _pop_map(prev_p)
        pop_first_map = _pop_map(first_p)
        has_prev = prev_p is not None

        # S-признаки
        student_feats: dict[str, Any] = {
            "institute": inst,
            "specialty": spec,
            "n_periods": n_periods,
            "n_choices": n_choices,
            "n_unique_modules": len(history_set),
            "avg_history_priority": (
                float(np.mean(history_priorities)) if history_priorities else 0.0
            ),
            "is_cold": int(is_cold),
        }

        candidates = candidates_fn(student)
        n_available = len(candidates)

        if tfidf_model is not None:
            tfidf_profile = tfidf_model.student_profile_mean(history_modules)
            tfidf_cos_mean_arr = tfidf_model.cosine_mean(tfidf_profile, candidates)
            tfidf_cos_max_arr = tfidf_model.cosine_max(history_modules, candidates)
        else:
            tfidf_cos_mean_arr = np.zeros(len(candidates), dtype=np.float32)
            tfidf_cos_max_arr = np.zeros(len(candidates), dtype=np.float32)

        if embed_model is not None:
            embed_profile = embed_model.student_profile_mean(history_modules)
            embed_cos_mean_arr = embed_model.cosine_mean(embed_profile, candidates)
            embed_cos_max_arr = embed_model.cosine_max(history_modules, candidates)
        else:
            embed_cos_mean_arr = np.zeros(len(candidates), dtype=np.float32)
            embed_cos_max_arr = np.zeros(len(candidates), dtype=np.float32)

        inst_pop = stats.pop_by_institute.get(inst, {})
        sorted_by_inst_pop = sorted(candidates, key=lambda m: -inst_pop.get(m, 0.0))
        rank_by_inst_pop = {m: r for r, m in enumerate(sorted_by_inst_pop)}

        sorted_by_global_pop = sorted(
            candidates,
            key=lambda m: (-stats.pop_weighted.get(m, 0.0), m),
        )
        rank_by_global_pop_in_avail = {
            m: r for r, m in enumerate(sorted_by_global_pop)
        }

        # pop_delta = pop_last - pop_prev. При prev_p=None (один период) все
        # delta = 0, ранг — по module_id (нет сигнала).
        delta_scores: dict[str, float] = {}
        for m in candidates:
            if has_prev:
                d = pop_last_map.get(m, 0.0) - pop_prev_map.get(m, 0.0)
            else:
                d = 0.0
            delta_scores[m] = d
        sorted_by_delta = sorted(
            candidates,
            key=lambda m: (-delta_scores[m], m),
        )
        rank_by_delta = {m: r for r, m in enumerate(sorted_by_delta)}

        target_map: dict[str, int] = target_fn(student) if target_fn else {}

        for i, mid in enumerate(candidates):
            pop_last = pop_last_map.get(mid, 0.0)
            pop_prev = pop_prev_map.get(mid, 0.0)
            pop_first = pop_first_map.get(mid, 0.0)

            if has_prev:
                pop_delta = pop_last - pop_prev
                denom = pop_last + pop_prev + 1e-6
                pop_delta_norm = pop_delta / denom
            else:
                pop_delta = 0.0
                pop_delta_norm = 0.0

            module_feats: dict[str, Any] = {
                "pop_weighted_global": stats.pop_weighted.get(mid, 0.0),
                "pop_raw_global": stats.pop_raw.get(mid, 0),
                "pop_by_institute": inst_pop.get(mid, 0.0),
                "pop_by_specialty": stats.pop_by_specialty.get(spec, {}).get(mid, 0.0),
                "pop_period_1": stats.pop_by_period.get(1, {}).get(mid, 0.0),
                "pop_period_2": stats.pop_by_period.get(2, {}).get(mid, 0.0),
                "pop_last_period": pop_last,
                "pop_prev_period": pop_prev,
                "pop_first_period": pop_first,
                "pop_delta": pop_delta,
                "pop_delta_norm": pop_delta_norm,
                "n_periods_available": len(stats.module_periods.get(mid, set())),
                "title_len": stats.module_title_len.get(mid, 0),
                "desc_len": stats.module_desc_len.get(mid, 0),
                "pop_rank_global": stats.pop_rank_global.get(mid, n_modules_global),
            }

            cooc_score = 0.0
            cooc_max = 0.0
            ppmi_score = 0.0
            ppmi_max = 0.0
            for h_mod in history_set:
                c = stats.cooc.get(h_mod, Counter()).get(mid, 0)
                cooc_score += c
                if c > cooc_max:
                    cooc_max = c
                p = stats.ppmi(h_mod, mid)
                ppmi_score += p
                if p > ppmi_max:
                    ppmi_max = p

            module_periods_set = stats.module_periods.get(mid, set())
            was_available_before = int(
                any(p in module_periods_set for p in original_history_periods)
            )

            global_pop = stats.pop_weighted.get(mid, 0.0)
            inst_pop_val = inst_pop.get(mid, 0.0)
            global_rank_in_avail = rank_by_global_pop_in_avail.get(mid, n_available)

            interaction_feats: dict[str, Any] = {
                "already_chosen": int(mid in history_set),
                "cooc_score": cooc_score,
                "cooc_max": cooc_max,
                "cooc_norm": cooc_score / max(1, n_choices),
                "cooc_ppmi_score": ppmi_score,
                "cooc_ppmi_max": ppmi_max,
                "cooc_ppmi_norm": ppmi_score / max(1, n_choices),
                "tfidf_cos_mean": float(tfidf_cos_mean_arr[i]),
                "tfidf_cos_max": float(tfidf_cos_max_arr[i]),
                "embed_cos_mean": float(embed_cos_mean_arr[i]),
                "embed_cos_max": float(embed_cos_max_arr[i]),
                "pop_ratio_in_institute": (
                    inst_pop_val / global_pop if global_pop > 0 else 0.0
                ),
                "pop_rank_in_available": rank_by_inst_pop.get(mid, n_available),
                "pop_rank_frac": rank_by_inst_pop.get(mid, n_available)
                    / max(1, n_available),
                "n_available": n_available,
                "was_available_before": was_available_before,
                "pop_rank_global_in_avail": global_rank_in_avail,
                "pop_rank_global_frac": global_rank_in_avail
                    / max(1, n_available),
                "pop_delta_rank": rank_by_delta.get(mid, n_available),
            }

            row: dict[str, Any] = {"student_id": sid, "module_id": mid}
            row.update(student_feats)
            row.update(module_feats)
            row.update(interaction_feats)

            if target_fn is not None:
                priority = target_map.get(mid, 0)
                row["priority"] = priority
                row["label"] = (6 - priority) if priority > 0 else 0
            else:
                row["priority"] = 0
                row["label"] = 0

            rows.append(row)

    return pd.DataFrame(rows)


def candidates_available(student: dict[str, Any]) -> list[str]:
    return list(student.get("available_modules") or [])


def make_candidates_train_period2(
    stats: FeatureStats,
) -> Callable[[dict[str, Any]], list[str]]:
    period_2_modules = sorted(
        mid for mid, periods in stats.module_periods.items() if 2 in periods
    )
    def fn(student: dict[str, Any]) -> list[str]:
        return period_2_modules
    return fn


def make_candidates_train_period1(
    stats: FeatureStats,
) -> Callable[[dict[str, Any]], list[str]]:
    period_1_modules = sorted(
        mid for mid, periods in stats.module_periods.items() if 1 in periods
    )
    def fn(student: dict[str, Any]) -> list[str]:
        return period_1_modules
    return fn


def make_candidates_all(
    stats: FeatureStats,
) -> Callable[[dict[str, Any]], list[str]]:
    """Все 322 модуля. Экспериментально: holdout падает (v34)."""
    all_modules = sorted(stats.module_periods.keys())
    def fn(student: dict[str, Any]) -> list[str]:
        return all_modules
    return fn


def target_period_2(student: dict[str, Any]) -> dict[str, int]:
    for h in student.get("history", []):
        if h.get("period") == 2:
            return {
                c["module_id"]: c.get("priority", 5)
                for c in h.get("choices", [])
            }
    return {}


def target_period_1(student: dict[str, Any]) -> dict[str, int]:
    for h in student.get("history", []):
        if h.get("period") == 1:
            return {
                c["module_id"]: c.get("priority", 5)
                for c in h.get("choices", [])
            }
    return {}


def make_target_from_val_solution(
    val_gt: dict[str, str],
) -> Callable[[dict[str, Any]], dict[str, int]]:
    from src.metric import parse_ground_truth
    def fn(student: dict[str, Any]) -> dict[str, int]:
        raw = val_gt.get(student["id"])
        return parse_ground_truth(raw) if raw else {}
    return fn


if __name__ == "__main__":
    import argparse

    from src.data_loader import load_all

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--content-backend",
        choices=VALID_CONTENT_BACKENDS,
        default="tfidf",
    )
    args = parser.parse_args()

    data = load_all()
    train, val, test = data["train"], data["val"], data["test"]
    modules = data["modules"]
    val_solution = data["val_solution"]

    print("=" * 72)
    print(f"ПРОВЕРКА FEATURE ENGINEERING (content_backend={args.content_backend})")
    print("=" * 72)

    print("\n[1/3] Статистики...")
    stats = FeatureStats.fit(train, modules, content_backend=args.content_backend)
    stats.add_val_solution_popularity(val_solution)

    print("\n[2/3] Проверка pop_delta на примере warm-студента (val_p2):")
    val_p2 = [s for s in val if len(s.get("history") or []) == 2][:1]
    if val_p2:
        df_p2 = build_dataset(
            val_p2, stats,
            candidates_fn=candidates_available,
            target_fn=None,
            past_periods=[1, 2],
        )
        print(f"  студент: {val_p2[0]['id']}")
        print(f"  строк: {len(df_p2)}")
        print(f"  pop_delta:      "
              f"min={df_p2['pop_delta'].min():.2f}, "
              f"med={df_p2['pop_delta'].median():.2f}, "
              f"max={df_p2['pop_delta'].max():.2f}, "
              f"nonzero={100*(df_p2['pop_delta'] != 0).mean():.1f}%")
        print(f"  pop_delta_norm: "
              f"min={df_p2['pop_delta_norm'].min():.3f}, "
              f"med={df_p2['pop_delta_norm'].median():.3f}, "
              f"max={df_p2['pop_delta_norm'].max():.3f}")
        print(f"  pop_delta_rank: "
              f"min={df_p2['pop_delta_rank'].min()}, "
              f"med={df_p2['pop_delta_rank'].median():.0f}, "
              f"max={df_p2['pop_delta_rank'].max()}")

    print("\n[3/3] Проверка pop_delta на cold-студенте (должно быть 0):")
    test_cold = next((s for s in test if not (s.get("history") or [])), test[0])
    df_cold = build_dataset(
        [test_cold], stats,
        candidates_fn=candidates_available,
        target_fn=None,
        past_periods=None,
    )
    print(f"  pop_delta: nonzero={100*(df_cold['pop_delta'] != 0).mean():.1f}% "
          f"(ожидаем 0%)")
    print(f"  pop_delta_norm: nonzero={100*(df_cold['pop_delta_norm'] != 0).mean():.1f}%")

    print("\n[OK] FE работает.")