"""
Feature engineering для LTR-модели.

Строит признаки трёх групп:
    S   — признаки студента;
    M   — признаки модуля;
    S×M — взаимодействие (co-occurrence, content-косинус, already_chosen, ...).

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

Content-backends (v29):
* "tfidf" — только TF-IDF (как в v7).
* "embed" — только семантические эмбеддинги (замена TF-IDF).
* "both"  — оба набора фичей.
* "none"  — без content-фичей (диагностика).

ВАЖНО: build_dataset ВСЕГДА пишет 4 колонки — `tfidf_cos_mean/max` и
`embed_cos_mean/max`. Если соответствующий backend не подключён, колонки
заполняются нулями. Это позволяет использовать один и тот же DataFrame
в разных feature-set'ах без ветвлений.

Кандидаты (v34+):
* make_candidates_train_period2(stats) — 174 модуля (2 ∈ available_in_periods).
* make_candidates_train_period1(stats) — 196 модулей (1 ∈ available_in_periods).
* make_candidates_all(stats)           — все 322 модуля из modules.jsonl.

Глобальные rank-фичи (v37):
* pop_rank_global          — ранг модуля по pop_weighted среди всех 322 модулей.
                             Одинаков для всех студентов. Строится в fit.
* pop_rank_global_in_avail — ранг модуля по pop_weighted среди candidates
                             конкретного студента. Per-student.
* pop_rank_global_frac     — нормализованный (rank / n_available).

Зачем: для cold-студентов pop_rank_in_available вырождается в алфавитную
сортировку (pop_by_institute = 0). Глобальные rank-фичи остаются
осмысленными и для cold, и для warm.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.content import ContentModel
from src.content_embed import ContentEmbedModel

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------
PRIORITY_WEIGHTS: dict[int, float] = {
    1: 1.00, 2: 0.80, 3: 0.60, 4: 0.45, 5: 0.35,
}
DEFAULT_INSTITUTE = "UNKNOWN"
DEFAULT_SPECIALTY = "UNKNOWN"

VALID_CONTENT_BACKENDS = ("tfidf", "embed", "both", "none")


# ===========================================================================
# FeatureStats
# ===========================================================================
class FeatureStats:
    """
    Агрегаты, вычисленные один раз на train. НЕ пересчитываются для val/test.
    """

    def __init__(self, content_backend: str = "tfidf") -> None:
        if content_backend not in VALID_CONTENT_BACKENDS:
            raise ValueError(
                f"content_backend={content_backend!r}, ожидалось одно из "
                f"{VALID_CONTENT_BACKENDS}"
            )
        self.content_backend: str = content_backend

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
        # --- Глобальный ранг по pop_weighted (v37) ---
        self.pop_rank_global: dict[str, int] = {}
        self.n_modules_global: int = 0
        # --- Co-occurrence ---
        self.cooc: dict[str, Counter] = defaultdict(Counter)
        self.module_doc_freq: dict[str, int] = {}
        self.n_students_total: int = 0
        # --- Метаданные модулей ---
        self.module_title_len: dict[str, int] = {}
        self.module_desc_len: dict[str, int] = {}
        self.module_periods: dict[str, set[int]] = {}
        # --- Content-based (могут быть None, если backend не подключён) ---
        self.content_model_tfidf: ContentModel | None = None
        self.content_model_embed: ContentEmbedModel | None = None
        # --- Служебное ---
        self.median_pop_weighted: float = 0.0
        self._has_period_3: bool = False

    # -----------------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        train: list[dict[str, Any]],
        modules: list[dict[str, Any]],
        content_backend: str = "tfidf",
    ) -> "FeatureStats":
        stats = cls(content_backend=content_backend)

        # --- 1. Метаданные модулей ---
        for m in modules:
            mid = m["module_id"]
            stats.module_title_len[mid] = len(m.get("title") or "")
            stats.module_desc_len[mid] = len(m.get("description") or "")
            stats.module_periods[mid] = set(m.get("available_in_periods") or [])
        stats.n_modules_global = len(stats.module_periods)

        # --- 2. Content-based модели (по требованию) ---
        if content_backend in ("tfidf", "both"):
            print(f"[FeatureStats] TF-IDF: fit на {len(modules)} модулях...")
            stats.content_model_tfidf = ContentModel.fit(modules)
        if content_backend in ("embed", "both"):
            print(f"[FeatureStats] Embeddings: fit на {len(modules)} модулях...")
            stats.content_model_embed = ContentEmbedModel.fit(modules)

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

        # --- 4. Глобальный ранг модулей по pop_weighted ---
        # Все модули из stats.module_periods получают ранг; модули без
        # популярности идут в конец, отсортированные по module_id.
        sorted_modules = sorted(
            stats.module_periods.keys(),
            key=lambda m: (-stats.pop_weighted.get(m, 0.0), m),
        )
        stats.pop_rank_global = {m: r for r, m in enumerate(sorted_modules)}
        return stats

    # -----------------------------------------------------------------------
    def add_val_solution_popularity(self, val_solution: pd.DataFrame) -> None:
        """
        Добавляет в pop_by_period агрегат за период 3, посчитанный из val_solution.

        ВНИМАНИЕ: это создаёт утечку val→модель, если val_solution используется
        при обучении на val-срезах. Для v7 это ожидаемое поведение, но при
        построении honest-валидации (KFold с чистым фолдом) этот вызов нужно
        либо убрать, либо обернуть в проверку.

        ВАЖНО (v37): НЕ пересчитывает pop_rank_global — глобальный ранг
        зафиксирован на train и не зависит от val-таргетов.
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

    Возвращает DataFrame с колонками:
        student_id, module_id,
        <S-признаки>, <M-признаки>, <S×M-признаки>,
        tfidf_cos_mean, tfidf_cos_max, embed_cos_mean, embed_cos_max,
        priority, label
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

        # --- Обрезка истории до последних N периодов + нормализация номеров ---
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

        # --- Content-based фичи: TF-IDF ---
        if tfidf_model is not None:
            tfidf_profile = tfidf_model.student_profile_mean(history_modules)
            tfidf_cos_mean_arr = tfidf_model.cosine_mean(tfidf_profile, candidates)
            tfidf_cos_max_arr = tfidf_model.cosine_max(history_modules, candidates)
        else:
            tfidf_cos_mean_arr = np.zeros(len(candidates), dtype=np.float32)
            tfidf_cos_max_arr = np.zeros(len(candidates), dtype=np.float32)

        # --- Content-based фичи: Embeddings ---
        if embed_model is not None:
            embed_profile = embed_model.student_profile_mean(history_modules)
            embed_cos_mean_arr = embed_model.cosine_mean(embed_profile, candidates)
            embed_cos_max_arr = embed_model.cosine_max(history_modules, candidates)
        else:
            embed_cos_mean_arr = np.zeros(len(candidates), dtype=np.float32)
            embed_cos_max_arr = np.zeros(len(candidates), dtype=np.float32)

        # --- Ранг среди кандидатов по институтской популярности ---
        inst_pop = stats.pop_by_institute.get(inst, {})
        sorted_by_inst_pop = sorted(candidates, key=lambda m: -inst_pop.get(m, 0.0))
        rank_by_inst_pop = {m: r for r, m in enumerate(sorted_by_inst_pop)}

        # --- Ранг среди кандидатов по ГЛОБАЛЬНОЙ популярности (v37) ---
        # Для cold (inst_pop = {}) это даёт осмысленный ранк вместо
        # алфавитной сортировки, которая получается в rank_by_inst_pop.
        sorted_by_global_pop = sorted(
            candidates,
            key=lambda m: (-stats.pop_weighted.get(m, 0.0), m),
        )
        rank_by_global_pop_in_avail = {
            m: r for r, m in enumerate(sorted_by_global_pop)
        }

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
                # --- v37: глобальный ранг модуля (одинаков для всех) ---
                "pop_rank_global": stats.pop_rank_global.get(mid, n_modules_global),
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
                # --- Content: TF-IDF ---
                "tfidf_cos_mean": float(tfidf_cos_mean_arr[i]),
                "tfidf_cos_max": float(tfidf_cos_max_arr[i]),
                # --- Content: Embeddings ---
                "embed_cos_mean": float(embed_cos_mean_arr[i]),
                "embed_cos_max": float(embed_cos_max_arr[i]),
                # --- Прочее ---
                "pop_ratio_in_institute": (
                    inst_pop_val / global_pop if global_pop > 0 else 0.0
                ),
                "pop_rank_in_available": rank_by_inst_pop.get(mid, n_available),
                "pop_rank_frac": rank_by_inst_pop.get(mid, n_available)
                    / max(1, n_available),
                "n_available": n_available,
                "was_available_before": was_available_before,
                # --- v37: глобальный ранг среди кандидатов ---
                "pop_rank_global_in_avail": global_rank_in_avail,
                "pop_rank_global_frac": global_rank_in_avail
                    / max(1, n_available),
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
# Обёртки: кандидаты
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


def make_candidates_all(
    stats: FeatureStats,
) -> Callable[[dict[str, Any]], list[str]]:
    """
    Все 322 модуля из modules.jsonl (по stats.module_periods).

    Зачем: train-пул (174 модуля для P2) и test-available (158 модулей)
    пересекаются лишь на Jaccard ≈ 0.42. Значит 57 test-модулей модель
    в обучении не видела, а 88 train-модулей в test не встретит.

    Эксперимент v34 показал: holdout падает (0.4707 против 0.5134).
    Функция оставлена для совместимости и дальнейших ablation.
    """
    all_modules = sorted(stats.module_periods.keys())
    def fn(student: dict[str, Any]) -> list[str]:
        return all_modules
    return fn


# ===========================================================================
# Обёртки: таргеты
# ===========================================================================
def target_period_2(student: dict[str, Any]) -> dict[str, int]:
    for h in student.get("history", []):
        if h.get("period") == 2:
            return {
                c["module_id"]: c.get("priority", 5)
                for c in h.get("choices", [])
            }
    return {}


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
    import argparse

    from src.data_loader import load_all

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--content-backend",
        choices=VALID_CONTENT_BACKENDS,
        default="tfidf",
        help="Какой content-сигнал использовать (default: tfidf)",
    )
    args = parser.parse_args()

    data = load_all()
    train, val, test = data["train"], data["val"], data["test"]
    modules = data["modules"]
    val_solution = data["val_solution"]

    print("=" * 72)
    print(f"ПРОВЕРКА FEATURE ENGINEERING (content_backend={args.content_backend})")
    print("=" * 72)

    print("\n[1/4] Статистики...")
    stats = FeatureStats.fit(train, modules, content_backend=args.content_backend)
    stats.add_val_solution_popularity(val_solution)

    # Проверка глобального ранга
    print(f"\n[2/4] pop_rank_global:")
    print(f"      всего модулей: {stats.n_modules_global}")
    top5_global = sorted(stats.pop_rank_global.items(), key=lambda kv: kv[1])[:5]
    print(f"      топ-5 по pop_weighted: {[m for m, _ in top5_global]}")
    print(f"      pop_rank_global топ-5: {[r for _, r in top5_global]}")

    # Проверка кандидатов
    cand_all_fn = make_candidates_all(stats)
    sample_candidates = cand_all_fn(test[0])
    print(f"\n[3/4] make_candidates_all: {len(sample_candidates)} модулей")

    test_avail_sample = set(test[0].get("available_modules") or [])
    cand_all_set = set(sample_candidates)
    print(f"      test_avail[0] ⊆ cand_all: {test_avail_sample.issubset(cand_all_set)}")

    # Проверка новых фич на холодном студенте
    print(f"\n[4/4] v37-фичи на cold-студенте test[0]:")
    cold_student = next((s for s in test if not (s.get("history") or [])), test[0])
    df_cold = build_dataset(
        [cold_student], stats,
        candidates_fn=candidates_available,
        target_fn=None,
        past_periods=None,
    )
    print(f"      строк: {len(df_cold)}")
    print(f"      pop_rank_global       : min={df_cold['pop_rank_global'].min()}, "
          f"med={df_cold['pop_rank_global'].median():.0f}, "
          f"max={df_cold['pop_rank_global'].max()}")
    print(f"      pop_rank_global_in_avail: min={df_cold['pop_rank_global_in_avail'].min()}, "
          f"med={df_cold['pop_rank_global_in_avail'].median():.0f}, "
          f"max={df_cold['pop_rank_global_in_avail'].max()}")
    print(f"      pop_rank_global_frac  : mean={df_cold['pop_rank_global_frac'].mean():.3f}")
    print(f"      --- для сравнения ---")
    print(f"      pop_rank_in_available : mean={df_cold['pop_rank_in_available'].mean():.1f} "
          f"(алфавитный шум)")
    print(f"      pop_rank_global_in_avail коррелирует с pop_weighted_global: "
          f"{df_cold[['pop_rank_global_in_avail', 'pop_weighted_global']].corr().iloc[0,1]:.3f}")

    print("\n[OK] FE работает.")