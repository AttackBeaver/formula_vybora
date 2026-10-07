"""
Feature engineering v2: добавлены "свежие" признаки на основе последнего периода.
"""
from __future__ import annotations
import math
from collections import Counter, defaultdict
from typing import Any, Callable
import numpy as np
import pandas as pd
from src.content import ContentModel

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
PRIORITY_WEIGHTS: dict[int, float] = {
    1: 1.00, 2: 0.80, 3: 0.60, 4: 0.45, 5: 0.35,
}
DEFAULT_INSTITUTE = "UNKNOWN"
DEFAULT_SPECIALTY = "UNKNOWN"

# ===========================================================================
# FeatureStats
# ===========================================================================
class FeatureStats:
    def __init__(self) -> None:
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
        self.cooc: dict[str, Counter] = defaultdict(Counter)
        self.module_doc_freq: dict[str, int] = {}
        self.n_students_total: int = 0
        self.module_title_len: dict[str, int] = {}
        self.module_desc_len: dict[str, int] = {}
        self.module_periods: dict[str, set[int]] = {}
        self.content_model: ContentModel | None = None
        self.median_pop_weighted: float = 0.0
        self._has_period_3: bool = False

    @classmethod
    def fit(
        cls,
        train: list[dict[str, Any]],
        modules: list[dict[str, Any]],
    ) -> FeatureStats:
        stats = cls()
        for m in modules:
            mid = m["module_id"]
            stats.module_title_len[mid] = len(m.get("title") or "")
            stats.module_desc_len[mid] = len(m.get("description") or "")
            stats.module_periods[mid] = set(m.get("available_in_periods") or [])
        
        stats.content_model = ContentModel.fit(modules)
        
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
        return stats

    def add_val_solution_popularity(self, val_solution: pd.DataFrame) -> None:
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
        cooc_ab = self.cooc.get(a, Counter()).get(b, 0)
        if cooc_ab == 0:
            return 0.0
        freq_a = self.module_doc_freq.get(a, 0)
        freq_b = self.module_doc_freq.get(b, 0)
        if freq_a <= 0 or freq_b <= 0:
            return 0.0
        pmi = math.log(cooc_ab * self.n_students_total / (freq_a * freq_b))
        return max(0.0, pmi)


# ===========================================================================
# Универсальный построитель датасета
# ===========================================================================
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
    rows: list[dict[str, Any]] = []
    content = stats.content_model
    use_id_filter = mask_student_ids is not None and len(mask_student_ids) > 0
    
    for student in students:
        sid = student.get("id") or student.get("student_id")
        if sid is None:
            continue
            
        if use_id_filter:
            do_mask = sid in mask_student_ids
        else:
            do_mask = mask_profile
            
        if do_mask:
            profile: dict[str, Any] = {}
            history_raw: list[dict[str, Any]] = []
        else:
            profile = student.get("profile") or {}
            history_raw = student.get("history", []) or []
            
        original_history_periods: set[int] = set()
        if max_history_periods is not None and history_raw:
            all_periods = sorted({h.get("period") for h in history_raw if h.get("period") is not None})
            original_history_periods = set(all_periods)
            if len(all_periods) > max_history_periods:
                keep_periods = set(all_periods[-max_history_periods:])
                filtered_history = [h for h in history_raw if h.get("period") in keep_periods]
                
                unique_kept_periods = sorted({h.get("period") for h in filtered_history if h.get("period") is not None})
                period_mapping = {p: i + 1 for i, p in enumerate(unique_kept_periods)}
                
                history_raw = []
                for h in filtered_history:
                    new_h = dict(h)
                    new_h["period"] = period_mapping[h["period"]]
                    history_raw.append(new_h)
        else:
            original_history_periods = {h.get("period") for h in history_raw if h.get("period") is not None}
            
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
        
        # === НОВОЕ: модули последнего периода ===
        last_period_modules: list[str] = []
        if history_raw:
            max_period = max(h.get("period", 0) for h in history_raw)
            for h in history_raw:
                if h.get("period") == max_period:
                    for c in h.get("choices", []):
                        last_period_modules.append(c["module_id"])
        last_period_set = set(last_period_modules)
        
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
            return stats.pop_by_period.get(period, {})
            
        pop_last_map = _pop_map(last_p)
        pop_prev_map = _pop_map(prev_p)
        pop_first_map = _pop_map(first_p)
        
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
        
        # === НОВОЕ: профиль по последнему периоду ===
        if content is not None and last_period_modules:
            last_period_profile_vec = content.student_profile_mean(last_period_modules)
        else:
            last_period_profile_vec = None
        
        if content is not None:
            profile_vec = content.student_profile_mean(history_modules)
            cos_mean_arr = content.cosine_mean(profile_vec, candidates)
            cos_max_arr = content.cosine_max(history_modules, candidates)
            
            # === НОВОЕ: cosine по последнему периоду ===
            if last_period_profile_vec is not None:
                cos_last_arr = content.cosine_mean(last_period_profile_vec, candidates)
            else:
                cos_last_arr = np.zeros(len(candidates), dtype=np.float32)
        else:
            cos_mean_arr = np.zeros(len(candidates), dtype=np.float32)
            cos_max_arr = np.zeros(len(candidates), dtype=np.float32)
            cos_last_arr = np.zeros(len(candidates), dtype=np.float32)
            
        inst_pop = stats.pop_by_institute.get(inst, {})
        sorted_by_inst_pop = sorted(candidates, key=lambda m: -inst_pop.get(m, 0.0))
        rank_by_inst_pop = {m: r for r, m in enumerate(sorted_by_inst_pop)}
        
        target_map: dict[str, int] = target_fn(student) if target_fn else {}
        
        for i, mid in enumerate(candidates):
            module_feats: dict[str, Any] = {
                "pop_weighted_global": stats.pop_weighted.get(mid, 0.0),
                "pop_raw_global": stats.pop_raw.get(mid, 0),
                "pop_by_institute": inst_pop.get(mid, 0.0),
                "pop_by_specialty": stats.pop_by_specialty.get(spec, {}).get(mid, 0.0),
                "pop_period_1": stats.pop_by_period.get(1, {}).get(mid, 0.0),
                "pop_period_2": stats.pop_by_period.get(2, {}).get(mid, 0.0),
                "pop_last_period": pop_last_map.get(mid, 0.0),
                "pop_prev_period": pop_prev_map.get(mid, 0.0),
                "pop_first_period": pop_first_map.get(mid, 0.0),
                "n_periods_available": len(stats.module_periods.get(mid, set())),
                "title_len": stats.module_title_len.get(mid, 0),
                "desc_len": stats.module_desc_len.get(mid, 0),
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
            
            # === НОВОЕ: cooc по последнему периоду ===
            cooc_last = 0.0
            for h_mod in last_period_set:
                c = stats.cooc.get(h_mod, Counter()).get(mid, 0)
                cooc_last += c
                    
            module_periods_set = stats.module_periods.get(mid, set())
            was_available_before = int(
                any(p in module_periods_set for p in original_history_periods)
            )
            
            global_pop = stats.pop_weighted.get(mid, 0.0)
            inst_pop_val = inst_pop.get(mid, 0.0)
            
            interaction_feats: dict[str, Any] = {
                "already_chosen": int(mid in history_set),
                "cooc_score": cooc_score,
                "cooc_max": cooc_max,
                "cooc_norm": cooc_score / max(1, n_choices),
                "cooc_ppmi_score": ppmi_score,
                "cooc_ppmi_max": ppmi_max,
                "cooc_ppmi_norm": ppmi_score / max(1, n_choices),
                "tfidf_cos_mean": float(cos_mean_arr[i]),
                "tfidf_cos_max": float(cos_max_arr[i]),
                "pop_ratio_in_institute": (
                    inst_pop_val / global_pop if global_pop > 0 else 0.0
                ),
                "pop_rank_in_available": rank_by_inst_pop.get(mid, n_available),
                "pop_rank_frac": rank_by_inst_pop.get(mid, n_available)
                    / max(1, n_available),
                "n_available": n_available,
                "was_available_before": was_available_before,
                # === НОВЫЕ ПРИЗНАКИ ===
                "cooc_with_last_period": cooc_last,
                "tfidf_cos_with_last_period": float(cos_last_arr[i]),
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


# ===========================================================================
# Обёртки
# ===========================================================================
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


def target_period_2(student: dict[str, Any]) -> dict[str, int]:
    for h in student.get("history", []):
        if h.get("period") == 2:
            return {
                c["module_id"]: c.get("priority", 5)
                for c in h.get("choices", [])
            }
    return {}


def make_candidates_train_period1(
    stats: FeatureStats,
) -> Callable[[dict[str, Any]], list[str]]:
    period_1_modules = sorted(
        mid for mid, periods in stats.module_periods.items() if 1 in periods
    )
    def fn(student: dict[str, Any]) -> list[str]:
        return period_1_modules
    return fn


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