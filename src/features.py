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
      (для test). В этом случае label=0 — чтобы prepare_xyg мог использовать
      единый интерфейс без ветвлений.
    * Есть режим маскирования профиля (mask_profile / mask_student_ids):
      нужен для обучения LTR на cold-подобных примерах. Когда маска активна,
      студент обрабатывается как cold: пустой профиль, пустая история,
      обнулённые S и S×M фичи. Остаются только M-признаки и глобальная
      популярность.
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
) -> pd.DataFrame:
    """
    Строит feature-датасет для списка студентов.

    Args:
        students: список записей (train/val/test).
        stats: обученные статистики.
        candidates_fn: функция (student) → список module_id-кандидатов.
        target_fn: функция (student) → {module_id: priority} для target.
                   None для test — тогда label=0 для всех строк.
        past_periods: список периодов, которые считаем «прошлым».
                     None — вся доступная история.
        mask_profile: если True — маскируем профиль у ВСЕХ студентов в партии.
        mask_student_ids: если задан — маскируем только студентов с этими sid.
                         Переопределяет mask_profile (то есть если задан
                         непустой set, mask_profile игнорируется).

    Returns:
        DataFrame со столбцами:
            student_id, module_id, S-фичи, M-фичи, S×M-фичи,
            priority, label.
    """
    rows: list[dict[str, Any]] = []
    content = stats.content_model

    # Определяем, кого маскируем.
    # mask_student_ids имеет приоритет: если задан непустой — маскируем только их.
    # Иначе — если mask_profile=True, маскируем всех.
    use_id_filter = mask_student_ids is not None and len(mask_student_ids) > 0

    for student in students:
        sid = student.get("id") or student.get("student_id")
        if sid is None:
            continue

        # --- Решаем, маскировать ли этого студента ---
        if use_id_filter:
            do_mask = sid in mask_student_ids  # type: ignore[operator]
        else:
            do_mask = mask_profile

        # --- Извлекаем профиль и историю (или стираем при маске) ---
        if do_mask:
            profile: dict[str, Any] = {}
            history_raw: list[dict[str, Any]] = []
        else:
            profile = student.get("profile") or {}
            history_raw = student.get("history", []) or []

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
        # При маске is_cold всегда 1. При естественных данных — как обычно.
        is_cold = (
            n_choices == 0
            and not profile.get("institute")
            and not profile.get("specialty")
        )

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
        # При маске history_modules=[] → profile_vec=None → нули.
        if content is not None:
            profile_vec = content.student_profile_mean(history_modules)
            cos_mean_arr = content.cosine_mean(profile_vec, candidates)
            cos_max_arr = content.cosine_max(history_modules, candidates)
        else:
            cos_mean_arr = np.zeros(len(candidates), dtype=np.float32)
            cos_max_arr = np.zeros(len(candidates), dtype=np.float32)

        # --- Ранг среди кандидатов по институтской популярности ---
        # При маске inst=UNKNOWN → inst_pop пустой → все нули, tie-break по id.
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
                "n_periods_available": len(stats.module_periods.get(mid, set())),
                "title_len": stats.module_title_len.get(mid, 0),
                "desc_len": stats.module_desc_len.get(mid, 0),
            }

            # ---------- S×M ----------
            # При маске history_set пустой → все суммы 0.
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
                any(p in module_periods_set for p in history_periods_used)
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
    train, val, modules = data["train"], data["val"], data["modules"]

    print("=" * 64)
    print("ПРОВЕРКА FEATURE ENGINEERING v3 (с маскированием)")
    print("=" * 64)

    print("\n[1/4] Обучаю статистики...")
    stats = FeatureStats.fit(train, modules)
    print(f"      Модулей: {len(stats.pop_weighted)}")

    print("\n[2/4] Train-датасет БЕЗ маски...")
    train_students = [
        s for s in train
        if any(h.get("period") == 2 for h in s.get("history", []))
    ]
    df_plain = build_dataset(
        train_students, stats,
        candidates_fn=make_candidates_train_period2(stats),
        target_fn=target_period_2,
        past_periods=[1],
        mask_profile=False,
    )
    print(f"      Строк: {len(df_plain):,}")

    print("\n[3/4] Тот же датасет С маской...")
    df_masked = build_dataset(
        train_students, stats,
        candidates_fn=make_candidates_train_period2(stats),
        target_fn=target_period_2,
        past_periods=[1],
        mask_profile=True,
    )
    print(f"      Строк: {len(df_masked):,}")

    print("\n[4/4] Сравнение ключевых фич (plain vs masked):")
    check_cols = [
        "is_cold", "n_choices", "cooc_score", "cooc_ppmi_score",
        "tfidf_cos_mean", "tfidf_cos_max", "already_chosen",
        "pop_weighted_global", "pop_rank_in_available",
    ]
    for col in check_cols:
        p_mean = df_plain[col].mean()
        m_mean = df_masked[col].mean()
        print(f"      {col:<24}  plain={p_mean:>10.4f}  masked={m_mean:>10.4f}")

    # Проверка инвариантов
    assert df_masked["is_cold"].min() == 1, "masked: is_cold должен быть везде 1"
    assert df_masked["n_choices"].max() == 0, "masked: n_choices должен быть 0"
    assert df_masked["cooc_score"].max() == 0, "masked: cooc_score должен быть 0"
    assert df_masked["already_chosen"].max() == 0, "masked: already_chosen = 0"
    # M-признаки должны сохраниться
    assert (df_masked["pop_weighted_global"] == df_plain["pop_weighted_global"]).all(), \
        "masked: pop_weighted_global должен совпадать с plain"

    print("\n[OK] Маскирование работает: S и S×M обнулены, M сохранены.")