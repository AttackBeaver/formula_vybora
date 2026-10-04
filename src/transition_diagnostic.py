"""
Transition diagnostic: проверка полезности матрицы переходов между соседними периодами.

Гипотеза:
    Test periods=3 (30.4% test) требует предсказать P4, для которого в
    train/val нет прямого обучающего сигнала. Модель B (без p2-среза)
    показывает на holdout 0.4050 против 0.4987 у A — теряет 0.094.

    Чтобы закрыть эту дыру, нужно добавить признак, который обобщается
    между train (P1→P2), val (P1,P2→P3) и test (P2,P3→P4).
    Transition-матрица «модуль X в P_t → модуль Y в P_{t+1}» даёт такой сигнал.

Что делаем:
    1. Считаем матрицу переходов из train (для пар (P1, P2)).
    2. Считаем ту же матрицу из val periods=2 (для пар (P2, P3), где P3 = val_solution).
       Это даёт нам «обобщённый паттерн перехода» без привязки к номеру периода.
    3. Добавляем признаки:
        transition_from_last_sum  — сумма переходов из модулей последнего периода
        transition_from_last_max  — максимум по модулям последнего периода
        transition_from_prev_sum  — сумма переходов из модуля предпоследнего периода
    4. Обучаем модель на 3 срезах (p0, p1, pmask) + p2 с новыми фичами.
       Сравниваем с A (0.4987) и B (0.4050) на holdout.

Запуск:
    python -m src.transition_diagnostic
"""
from __future__ import annotations

import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_loader import load_all
from src.features import (
    FeatureStats,
    candidates_available,
    make_candidates_train_period1,
    make_candidates_train_period2,
    target_period_1,
    target_period_2,
)
from src.ltr_model import (
    CATEGORICAL_FEATS,
    FEATURE_COLS_V4,
    encode_categoricals,
    predict_scores,
    prepare_xygw,
    top_k_per_student,
    train_lgbm_ranker,
)
from src.metric import evaluate_pndcg5, parse_ground_truth
from src.content import ContentModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent

WEIGHT_P1 = 1.0
WEIGHT_P0 = 0.5
WEIGHT_PMASK = 1.0
WEIGHT_P2 = 1.0

HOLDOUT_FRACTION = 0.20
HOLDOUT_SEED = 42

FAST_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
}

# Итоговый набор: FEATURE_COLS_V4 + 3 новых transition-фичи
FEATURE_COLS_V5: list[str] = FEATURE_COLS_V4 + [
    "transition_from_last_sum",
    "transition_from_last_max",
    "transition_from_prev_sum",
]


def _n_periods(r: dict[str, Any]) -> int:
    return len(r.get("history") or [])


# ---------------------------------------------------------------------------
# Transition-матрица
# ---------------------------------------------------------------------------
def build_transition_matrix(
    students: list[dict[str, Any]],
    val_gt: dict[str, str] | None = None,
) -> tuple[dict[str, Counter], dict[str, int]]:
    """
    Считает матрицу переходов между соседними периодами.

    Для train (у студентов 2 периода):
        История [(P1, [mods]), (P2, [mods])].
        Для каждой пары (m1 in P1, m2 in P2): trans[m1][m2] += 1.

    Для val (у студентов 1-2 периода + target в val_solution):
        Если history = [(P1, ...)], target = val_solution.
        Тогда для каждой пары (m1 in P1, m2 in target): trans[m1][m2] += 1.
        Если history = [(P1, ...), (P2, ...)], target = val_solution.
        Тогда для каждой пары (m2 in P2, m3 in target): trans[m2][m3] += 1.

    Возвращает:
        trans: dict[X][Y] = count
        freq: dict[X] = количество вхождений X (по всем «первым» позициям пар)
    """
    trans: dict[str, Counter] = defaultdict(Counter)
    freq: Counter = Counter()

    for s in students:
        history = s.get("history") or []
        if not history:
            continue

        # Смотрим на последние два периода (если есть) — или на последний + target.
        last_period_mods = [c["module_id"] for c in history[-1].get("choices", [])]
        prev_period_mods: list[str] = []
        if len(history) >= 2:
            prev_period_mods = [c["module_id"] for c in history[-2].get("choices", [])]

        # Целевые модули
        target_mods: list[str] = []
        if val_gt is not None:
            sid = s.get("id") or s.get("student_id")
            gt_str = val_gt.get(sid) if sid else None
            if gt_str:
                gt = parse_ground_truth(gt_str)
                target_mods = list(gt.keys())

        # Определяем пары (source_period_mods, target_period_mods)
        pairs_to_add: list[tuple[list[str], list[str]]] = []

        if prev_period_mods and last_period_mods:
            # Есть два периода в истории → пара (prev, last)
            pairs_to_add.append((prev_period_mods, last_period_mods))

        if target_mods:
            if last_period_mods:
                # Целевой период известен отдельно → пара (last, target)
                pairs_to_add.append((last_period_mods, target_mods))
            elif prev_period_mods:
                pairs_to_add.append((prev_period_mods, target_mods))

        for sources, targets in pairs_to_add:
            for m1 in sources:
                freq[m1] += 1
                for m2 in targets:
                    if m1 != m2:
                        trans[m1][m2] += 1

    return dict(trans), dict(freq)


# ---------------------------------------------------------------------------
# Применение transition-фичей
# ---------------------------------------------------------------------------
def attach_transition_features(
    df: pd.DataFrame,
    students: list[dict[str, Any]],
    trans: dict[str, Counter],
    freq: dict[str, int],
) -> pd.DataFrame:
    """
    Добавляет к df колонки transition_from_last_sum/max и transition_from_prev_sum.

    Логика по студенту:
        - Определяем модули в последнем и предпоследнем периодах истории
          (по номерам периодов, без прошлого).
        - Для каждого кандидата mid:
            transition_from_last_sum = Σ_{h in last_mods} trans[h].get(mid, 0)
            transition_from_last_max = max_{h in last_mods} trans[h].get(mid, 0)
            transition_from_prev_sum = Σ_{h in prev_mods} trans[h].get(mid, 0)

    ВАЖНО: df.dtypes совместимы с pd.merge. Соединяем по (student_id, module_id).

    Args:
        df: DataFrame с колонками student_id, module_id. Префиксы
            (p1__, p0__, ho__) должны быть УЖЕ удалены или, наоборот,
            student_id должен соответствовать ключам в students_by_id.
        students: список записей студентов.
        trans, freq: матрица переходов.

    Returns:
        df с тремя новыми колонками.
    """
    # Индекс: student_id → (last_mods, prev_mods)
    by_id: dict[str, tuple[list[str], list[str]]] = {}
    for s in students:
        sid = s.get("id") or s.get("student_id")
        if sid is None:
            continue
        history = s.get("history") or []
        last_mods = [c["module_id"] for c in history[-1].get("choices", [])] if history else []
        prev_mods = (
            [c["module_id"] for c in history[-2].get("choices", [])]
            if len(history) >= 2 else []
        )
        by_id[sid] = (last_mods, prev_mods)

    # Извлекаем «чистый» id (без префикса p1__/p0__/pmask__/p2__/ho__/val__)
    def clean_id(raw: str) -> str:
        if "__" in raw:
            return raw.split("__", 1)[1]
        return raw

    tls = np.zeros(len(df), dtype=np.float32)
    tlm = np.zeros(len(df), dtype=np.float32)
    tps = np.zeros(len(df), dtype=np.float32)

    student_ids = df["student_id"].values
    module_ids = df["module_id"].values

    for i in range(len(df)):
        sid = clean_id(str(student_ids[i]))
        mid = str(module_ids[i])
        last_mods, prev_mods = by_id.get(sid, ([], []))

        s_last = 0
        m_last = 0
        for h in last_mods:
            v = trans.get(h, Counter()).get(mid, 0)
            s_last += v
            if v > m_last:
                m_last = v
        s_prev = 0
        for h in prev_mods:
            s_prev += trans.get(h, Counter()).get(mid, 0)

        tls[i] = s_last
        tlm[i] = m_last
        tps[i] = s_prev

    out = df.copy()
    out["transition_from_last_sum"] = tls
    out["transition_from_last_max"] = tlm
    out["transition_from_prev_sum"] = tps
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print("TRANSITION DIAGNOSTIC")
    print("=" * 78)

    t0 = time.time()
    data = load_all()
    train = data["train"]
    val = data["val"]
    modules = data["modules"]
    val_solution = data["val_solution"]
    val_gt_raw = dict(zip(val_solution["id"], val_solution["ground_truth"]))

    print("\n[1/6] Считаю статистики и transition-матрицу...")
    stats = FeatureStats.fit(train, modules)
    stats.add_val_solution_popularity(val_solution)

    # Матрица переходов: train (P1→P2) + val (последний→target)
    trans_train, freq_train = build_transition_matrix(train, val_gt=None)
    trans_val, freq_val = build_transition_matrix(val, val_gt=val_gt_raw)

    # Объединяем
    trans: dict[str, Counter] = defaultdict(Counter)
    for src, cnt in trans_train.items():
        for dst, c in cnt.items():
            trans[src][dst] += c
    for src, cnt in trans_val.items():
        for dst, c in cnt.items():
            trans[src][dst] += c
    trans = dict(trans)
    freq = {**freq_train}
    for k, v in freq_val.items():
        freq[k] = freq.get(k, 0) + v

    print(f"      transition: {len(trans)} модулей-источников, "
          f"{sum(len(v) for v in trans.values())} рёбер")
    top_src = max(trans.items(), key=lambda kv: sum(kv[1].values()))
    print(f"      топ источник: {top_src[0]} → "
          f"{sum(top_src[1].values())} переходов")

    # ==== 2. Сырые срезы (без transition) ====
    print("\n[2/6] Строю базовые датасеты...")
    # Используем те же утилиты, что в v4
    from src.features import build_dataset, make_target_from_val_solution

    train_students_p1 = [
        s for s in train
        if any(h.get("period") == 2 for h in s.get("history", []))
    ]
    train_df_p1 = build_dataset(
        train_students_p1, stats,
        candidates_fn=make_candidates_train_period2(stats),
        target_fn=target_period_2,
        past_periods=[1],
    )
    train_df_p0 = build_dataset(
        [s for s in train if any(h.get("period") == 1 for h in s.get("history", []))],
        stats,
        candidates_fn=make_candidates_train_period1(stats),
        target_fn=target_period_1,
        past_periods=[],
    )
    train_df_pmask = build_dataset(
        train_students_p1, stats,
        candidates_fn=make_candidates_train_period2(stats),
        target_fn=target_period_2,
        past_periods=[1],
        mask_profile=True,
    )

    val_p2 = [s for s in val if _n_periods(s) == 2]
    rng = random.Random(HOLDOUT_SEED)
    shuffled = list(val_p2)
    rng.shuffle(shuffled)
    n_holdout = int(len(shuffled) * HOLDOUT_FRACTION)
    holdout_records = shuffled[:n_holdout]
    p2_records = shuffled[n_holdout:]

    p2_df = build_dataset(
        p2_records, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )
    holdout_df = build_dataset(
        holdout_records, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )

    # ==== 3. Применяем transition-фичи ====
    print("\n[3/6] Добавляю transition-фичи...")

    train_df_p1 = attach_transition_features(train_df_p1, train_students_p1, trans, freq)
    train_df_p0 = attach_transition_features(
        train_df_p0,
        [s for s in train if any(h.get("period") == 1 for h in s.get("history", []))],
        trans, freq,
    )
    train_df_pmask = attach_transition_features(train_df_pmask, train_students_p1, trans, freq)
    p2_df = attach_transition_features(p2_df, p2_records, trans, freq)
    holdout_df = attach_transition_features(holdout_df, holdout_records, trans, freq)

    # Проверим, что фичи не нулевые
    for name, df in [
        ("p1", train_df_p1), ("p0", train_df_p0), ("pmask", train_df_pmask),
        ("p2", p2_df), ("holdout", holdout_df),
    ]:
        nz = int((df["transition_from_last_sum"] > 0).sum())
        print(f"      {name:<10} transition_from_last_sum>0: {nz:>7} / {len(df):>7}")

    # ==== 4. Сборка + кодирование ====
    print("\n[4/6] Сборка + кодирование...")

    train_df_p1["weight"] = np.float32(WEIGHT_P1)
    train_df_p0["weight"] = np.float32(WEIGHT_P0)
    train_df_pmask["weight"] = np.float32(WEIGHT_PMASK)
    p2_df["weight"] = np.float32(WEIGHT_P2)
    holdout_df["weight"] = np.float32(1.0)

    train_df_p1["student_id"] = "p1__" + train_df_p1["student_id"].astype(str)
    train_df_p0["student_id"] = "p0__" + train_df_p0["student_id"].astype(str)
    train_df_pmask["student_id"] = "pmask__" + train_df_pmask["student_id"].astype(str)
    p2_df["student_id"] = "p2__" + p2_df["student_id"].astype(str)
    holdout_df["student_id"] = "ho__" + holdout_df["student_id"].astype(str)

    train_df = pd.concat([train_df_p1, train_df_p0, train_df_pmask, p2_df], ignore_index=True)

    n_train = len(train_df)
    n_ho = len(holdout_df)
    all_df = pd.concat([train_df, holdout_df], ignore_index=True)
    all_df, _ = encode_categoricals(all_df, all_df, CATEGORICAL_FEATS)

    train_df = all_df.iloc[:n_train].reset_index(drop=True)
    holdout_df = all_df.iloc[n_train:].reset_index(drop=True)

    # ==== 5. Обучение ====
    print("\n[5/6] Обучение с FEATURE_COLS_V5 (31 + 3 transition)...")

    X_tr, y_tr, g_tr, w_tr, _ = prepare_xygw(train_df, feature_cols=FEATURE_COLS_V5)
    X_ho, y_ho, g_ho, w_ho, ho_sorted = prepare_xygw(holdout_df, feature_cols=FEATURE_COLS_V5)

    print(f"      Train: X={X_tr.shape}, групп={len(g_tr)}")

    t_train = time.time()
    model = train_lgbm_ranker(
        X_tr, y_tr, g_tr, X_ho, y_ho, g_ho,
        params=FAST_PARAMS, num_boost_round=2500, early_stopping_rounds=100,
        log_every=100, sample_weight_train=w_tr, sample_weight_val=w_ho,
    )
    print(f"      Обучено за {time.time() - t_train:.1f} сек, "
          f"best_iter={model.best_iteration}")

    # ==== 6. Оценка ====
    print("\n[6/6] Оценка:")

    scores = predict_scores(model, ho_sorted, feature_cols=FEATURE_COLS_V5)
    rec = top_k_per_student(ho_sorted, scores, k=5)
    rec_clean = {sid.replace("ho__", ""): v for sid, v in rec.items()}
    ho_ids = {s["id"] for s in holdout_records}
    gt_ho = {sid: gt for sid, gt in val_gt_raw.items() if sid in ho_ids}

    mean_ho, _ = evaluate_pndcg5(rec_clean, gt_ho, k=5)
    print(f"\n      Holdout (n={len(holdout_records)}): {mean_ho:.4f}")
    print(f"      --- сравнение:")
    print(f"          A (с p2, без transition):  0.4987")
    print(f"          B (без p2):                0.4050")
    print(f"          v5 (с p2 + transition):    {mean_ho:.4f}")

    # Feature importance
    print("\n[Feature importance — топ-15]")
    gain = model.feature_importance(importance_type="gain")
    imp = sorted(zip(FEATURE_COLS_V5, gain), key=lambda t: -t[1])
    for name, g in imp[:15]:
        print(f"      {name:<32}  gain = {g:>12.0f}")

    print(f"\n[OK] Диагностика завершена за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()