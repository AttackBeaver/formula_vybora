"""
Feature engineering для LTR-модели.
Строит признаки трёх групп:
S   — признаки студента;
M   — признаки модуля;
S×M — взаимодействие (co-occurrence, TF-IDF косинус, already_chosen, ...).
Все агрегаты вычисляются один раз на train (fit) и переиспользуются
для val/test (transform) — это устраняет training-serving skew.
Ключевые свойства:
* build_dataset ВСЕГДА добавляет колонку `label`, даже когда target_fn=None
(для test). В этом случае label=0 — единый интерфейс без ветвлений.
* Есть режим маскирования профиля (mask_profile / mask_student_ids):
нужен для обучения LTR на cold-подобных примерах.
* Относительные периоды (pop_last_period, pop_prev_period, pop_first_period):
устраняют сдвиг распределения между train (история до P2) и test
(история до P3/P4). См. FeatureStats.add_val_solution_popularity.
* max_history_periods: обрезает историю студента до последних N периодов
И НОРМАЛИЗУЕТ их к относительным номерам (1, 2...), чтобы выровнять
распределение признаков pop_last_period/prev_period с обучающей выборкой.
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
    """
    Агрегаты, вычисленные один раз на train. НЕ пересчитываются для val/test.
    """
    def __init__(self) -> None:
        # --- Популярность ---
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
        # --- Co-occurrence ---
        self.cooc: dict[str, Counter] = defaultdict(Counter)
        self.module_doc_freq: dict[str, int] = {}
        self.n_students_total: int = 0
        # --- Метаданные модулей ---
        self.module_title_len: dict[str, int] = {}
        self.module_desc_len: dict[str, int] = {}
        self.module_periods: dict[str, set[int]] = {}
        # --- Content-based ---
        self.content_model: ContentModel | None = None
        # --- Служебное ---
        self.median_pop_weighted: float = 0.0
        self._has_period_3: bool = False

    # -----------------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        train: list[dict[str, Any]],
        modules: list[dict[str, Any]],
    ) -> FeatureStats:
        stats = cls()
        # --- 1. Метаданные модулей ---
        for m in modules:
            mid = m["module_id"]
            stats.module_title_len[mid] = len(m.get("title") or "")
            stats.module_desc_len[mid] = len(m.get("description") or "")
            stats.module_periods[mid] = set(m.get("available_in_periods") or [])
        
        # --- 2. Content-based модель ---
        stats.content_model = ContentModel.fit(modules)
        
        # --- 3. Популярность и co-occurrence ---
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

    # -----------------------------------------------------------------------
    def add_val_solution_popularity(self, val_solution: pd.DataFrame) -> None:
        """
        Добавляет в pop_by_period агрегат за период 3, посчитанный из val_solution.
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

    # -----------------------------------------------------------------------
    def ppmi(self, a: str, b: str) -> float:
        """
        PPMI между двумя модулями: max(0, log( P(a,b) / (P(a)·P(b)) )).
        """
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
    """
    Строит feature-датасет для списка студентов.
    """
    rows: list[dict[str, Any]] = []
    content = stats.content_model
    use_id_filter = mask_student_ids is not None and len(mask_student_ids) > 0
    
    for student in students:
        sid = student.get("id") or student.get("student_id")
        if sid is None:
            continue
            
        # --- Маскировать ли этого студента ---
        if use_id_filter:
            do_mask = sid in mask_student_ids  # type: ignore[operator]
        else:
            do_mask = mask_profile
            
        # --- Профиль и история (или стираем при маске) ---
        if do_mask:
            profile: dict[str, Any] = {}
            history_raw: list[dict[str, Any]] = []
        else:
            profile = student.get("profile") or {}
            history_raw = student.get("history", []) or []
            
        # --- Обрезка истории до последних N периодов И НОРМАЛИЗАЦИЯ к относительным периодам ---
        # КРИТИЧЕСКИ ВАЖНО для test-студентов с 3+ периодами: после обрезки до 2 периодов
        # (например, остались периоды 2 и 3), мы переименовываем их в 1 и 2.
        # Это устраняет сдвиг распределения для признаков pop_last_period и pop_prev_period,
        # так как модель в train (p2-срез) видела только периоды 1 и 2.
        original_history_periods: set[int] = set()
        if max_history_periods is not None and history_raw:
            all_periods = sorted({h.get("period") for h in history_raw if h.get("period") is not None})
            original_history_periods = set(all_periods)
            if len(all_periods) > max_history_periods:
                keep_periods = set(all_periods[-max_history_periods:])
                filtered_history = [h for h in history_raw if h.get("period") in keep_periods]
                
                # Нормализуем периоды: самый ранний из оставшихся становится 1, следующий 2 и т.д.
                unique_kept_periods = sorted({h.get("period") for h in filtered_history if h.get("period") is not None})
                period_mapping = {p: i + 1 for i, p in enumerate(unique_kept_periods)}
                
                history_raw = []
                for h in filtered_history:
                    new_h = dict(h)  # поверхностная копия, choices не мутируем
                    new_h["period"] = period_mapping[h["period"]]
                    history_raw.append(new_h)
        else:
            original_history_periods = {h.get("period") for h in history_raw if h.get("period") is not None}
            
        inst = _safe_get(profile, "institute")
        spec = _safe_get(profile, "specialty")
        
        # --- История студента (с фильтром по past_periods) ---
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
        
        # --- Относительные периоды: last / prev / first ---
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
        
        # --- Признаки студента (S) ---
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
        
        # --- Кандидаты ---
        candidates = candidates_fn(student)
        n_available = len(candidates)
        
        # --- TF-IDF косинусы ---
        if content is not None:
            profile_vec = content.student_profile_mean(history_modules)
            cos_mean_arr = content.cosine_mean(profile_vec, candidates)
            cos_max_arr = content.cosine_max(history_modules, candidates)
        else:
            cos_mean_arr = np.zeros(len(candidates), dtype=np.float32)
            cos_max_arr = np.zeros(len(candidates), dtype=np.float32)
            
        # --- Ранг среди кандидатов по институтской популярности ---
        inst_pop = stats.pop_by_institute.get(inst, {})
        sorted_by_inst_pop = sorted(candidates, key=lambda m: -inst_pop.get(m, 0.0))
        rank_by_inst_pop = {m: r for r, m in enumerate(sorted_by_inst_pop)}
        
        # --- Ground truth ---
        target_map: dict[str, int] = target_fn(student) if target_fn else {}
        
        # --- Пробегаем по кандидатам ---
        for i, mid in enumerate(candidates):
            # ---------- M: признаки модуля ----------
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
            
            # ---------- S×M ----------
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
            # Используем original_history_periods, чтобы корректно проверить доступность модуля 
            # в оригинальных периодах, несмотря на нормализацию истории студента.
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
            }
            
            # ---------- Сборка ----------
            row: dict[str, Any] = {"student_id": sid, "module_id": mid}
            row.update(student_feats)
            row.update(module_feats)
            row.update(interaction_feats)
            
            # ---------- Target (всегда) ----------
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
    """Кандидаты для target=period 1: все модули, доступные в period 1."""
    period_1_modules = sorted(
        mid for mid, periods in stats.module_periods.items() if 1 in periods
    )
    def fn(student: dict[str, Any]) -> list[str]:
        return period_1_modules
    return fn


def target_period_1(student: dict[str, Any]) -> dict[str, int]:
    """Для train: {module_id: priority} за period 1."""
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


# ===========================================================================
# CLI: проверка
# ===========================================================================
if __name__ == "__main__":
    from src.data_loader import load_all
    data = load_all()
    train, val, test, modules = data["train"], data["val"], data["test"], data["modules"]
    val_solution = data["val_solution"]
    
    print("=" * 72)
    print("ПРОВЕРКА FEATURE ENGINEERING: max_history_periods + нормализация")
    print("=" * 72)
    
    print("\n[1/3] Статистики...")
    stats = FeatureStats.fit(train, modules)
    stats.add_val_solution_popularity(val_solution)
    
    print("\n[2/3] test: без обрезки vs с max_history_periods=2")
    test_no_trim = build_dataset(
        test, stats,
        candidates_fn=candidates_available,
        target_fn=None,
        past_periods=None,
    )
    test_trim = build_dataset(
        test, stats,
        candidates_fn=candidates_available,
        target_fn=None,
        past_periods=None,
        max_history_periods=2,
    )
    
    for name, df in [("без обрезки", test_no_trim), ("max_history=2", test_trim)]:
        n3_mask = df["student_id"].isin(
            [s["id"] for s in test if len(s.get("history") or []) == 3]
        )
        df_p3 = df[n3_mask]
        print(f"\n  {name}:")
        print(f"      всего строк: {len(df):,}")
        print(f"      periods=3 студентов: {df_p3['student_id'].nunique():,}")
        print(f"      n_periods (по группам): "
              f"{dict(df.groupby('student_id')['n_periods'].first().value_counts().sort_index())}")
        print(f"      n_choices (среднее по всем): {df['n_choices'].mean():.3f}")
        
    # Проверка что для periods=3 обрезалось до 2 периодов И нормализовалось
    test_p3_ids = {s["id"] for s in test if len(s.get("history") or []) == 3}
    p3_trim = test_trim[test_trim["student_id"].isin(test_p3_ids)]
    n_periods_after = p3_trim.groupby("student_id")["n_periods"].first()
    print(f"\n  После обрезки+нормализации у periods=3 в test:")
    print(f"      n_periods распределение: {dict(n_periods_after.value_counts().sort_index())}")
    print(f"      Ожидаем все = 2: {'OK' if (n_periods_after == 2).all() else 'FAIL'}")
    
    # Проверка, что pop_last_period теперь берётся из pop_by_period[2], а не [3]
    # Для этого смотрим на уникальные значения n_periods в test_trim
    print(f"\n  Уникальные значения n_periods в test_trim: {sorted(test_trim['n_periods'].unique())}")
    print(f"  Ожидаем [0, 1, 2] (не должно быть 3): {'OK' if 3 not in test_trim['n_periods'].unique() else 'FAIL'}")
    
    print("\n[3/3] Проверка, что обрезка не влияет на periods<=2 в test")
    test_p_le2_ids = {s["id"] for s in test if len(s.get("history") or []) <= 2}
    a = test_no_trim[test_no_trim["student_id"].isin(test_p_le2_ids)].sort_values(
        ["student_id", "module_id"],
    ).reset_index(drop=True)
    b = test_trim[test_trim["student_id"].isin(test_p_le2_ids)].sort_values(
        ["student_id", "module_id"],
    ).reset_index(drop=True)
    print(f"      Строк без обрезки: {len(a)}")
    print(f"      Строк с обрезкой:  {len(b)}")
    same = a.equals(b)
    print(f"      Фреймы идентичны: {same}")
    
    print("\n[OK] FE работает.")