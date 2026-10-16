"""
LightGBM Ranker — основная LTR-модель.

Использует objective='lambdarank' с кастомным label_gain, соответствующим
весам приоритетов из метрики pNDCG@5.

Отличительная особенность: используется CUSTOM EVAL `pndcg5_eval`,
считающий ИМЕННО нашу метрику (norm=1, clip at 1.0) — а не классический NDCG
LightGBM. Это позволяет early stopping останавливать обучение по правильной
цели.

Поддержка sample weights и НЕСКОЛЬКИХ наборов признаков:

    FEATURE_COLS     — v1-v3, 30 признаков с абсолютными pop_period_1/2.
    FEATURE_COLS_V4  — v4, 31 признак с ОТНОСИТЕЛЬНЫМИ pop_last/prev/first.
                       Используется в v7 (champion).
    FEATURE_COLS_V5  — 31 признак. TF-IDF заменён на embeddings.
    FEATURE_COLS_V6  — 33 признака. TF-IDF + embeddings.
    FEATURE_COLS_V7  — v4 без 3 rank-фич (pop_rank_in_available,
                       pop_rank_frac, pop_ratio_in_institute).
    FEATURE_COLS_V8  — v6 без 3 rank-фич.
    FEATURE_COLS_V11 — v4 + 3 глобальные rank-фичи:
                       pop_rank_global, pop_rank_global_in_avail,
                       pop_rank_global_frac.
                       Гипотеза: pop_rank_in_available вырождается для cold
                       (нет института → pop_by_institute=0 → алфавитный
                       порядок). Глобальные rank-фичи остаются
                       осмысленными и для cold, и для warm.
    FEATURE_COLS_V12 — v7 + те же 3 глобальные rank-фичи.
                       Это ЗАМЕНА вырождающихся rank-3 на глобальные.

ВАЖНО: набор фичей должен соответствовать тому, как посчитан датасет.
При content_backend="embed" в FeatureStats.build_dataset колонки
tfidf_cos_* заполнены нулями — значит, использовать нужно FEATURE_COLS_V5.
При content_backend="both" — оба набора заполнены, использовать FEATURE_COLS_V6/V8.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Наборы признаков
# ---------------------------------------------------------------------------
# v1-v3: 30 признаков с абсолютными периодами.
FEATURE_COLS: list[str] = [
    # S — студент
    "institute", "specialty",
    "n_periods", "n_choices", "n_unique_modules",
    "avg_history_priority", "is_cold",
    # M — модуль
    "pop_weighted_global", "pop_raw_global", "pop_by_institute",
    "pop_by_specialty", "pop_period_1", "pop_period_2",
    "n_periods_available", "title_len", "desc_len",
    # S×M — взаимодействие (raw co-occurrence)
    "already_chosen", "cooc_score", "cooc_max", "cooc_norm",
    # S×M — взаимодействие (PPMI co-occurrence)
    "cooc_ppmi_score", "cooc_ppmi_max", "cooc_ppmi_norm",
    # S×M — content-based (TF-IDF cosine)
    "tfidf_cos_mean", "tfidf_cos_max",
    # S×M — прочее
    "pop_ratio_in_institute", "pop_rank_in_available", "pop_rank_frac",
    "n_available", "was_available_before",
]

# v4: 31 признак. Абсолютные pop_period_1/2 заменены относительными
# pop_last_period / pop_prev_period / pop_first_period.
FEATURE_COLS_V4: list[str] = [
    # S — студент (7)
    "institute", "specialty",
    "n_periods", "n_choices", "n_unique_modules",
    "avg_history_priority", "is_cold",
    # M — модуль (10)
    "pop_weighted_global", "pop_raw_global", "pop_by_institute",
    "pop_by_specialty",
    "pop_last_period", "pop_prev_period", "pop_first_period",
    "n_periods_available", "title_len", "desc_len",
    # S×M — взаимодействие (14)
    "already_chosen", "cooc_score", "cooc_max", "cooc_norm",
    "cooc_ppmi_score", "cooc_ppmi_max", "cooc_ppmi_norm",
    "tfidf_cos_mean", "tfidf_cos_max",
    "pop_ratio_in_institute", "pop_rank_in_available", "pop_rank_frac",
    "n_available", "was_available_before",
]

# v5: 31 признак. TF-IDF заменён на семантические эмбеддинги.
FEATURE_COLS_V5: list[str] = [
    # S — студент (7)
    "institute", "specialty",
    "n_periods", "n_choices", "n_unique_modules",
    "avg_history_priority", "is_cold",
    # M — модуль (10)
    "pop_weighted_global", "pop_raw_global", "pop_by_institute",
    "pop_by_specialty",
    "pop_last_period", "pop_prev_period", "pop_first_period",
    "n_periods_available", "title_len", "desc_len",
    # S×M — взаимодействие (14, content теперь embed)
    "already_chosen", "cooc_score", "cooc_max", "cooc_norm",
    "cooc_ppmi_score", "cooc_ppmi_max", "cooc_ppmi_norm",
    "embed_cos_mean", "embed_cos_max",
    "pop_ratio_in_institute", "pop_rank_in_available", "pop_rank_frac",
    "n_available", "was_available_before",
]

# v6: 33 признака. Оба content-сигнала: TF-IDF и embeddings.
FEATURE_COLS_V6: list[str] = [
    # S — студент (7)
    "institute", "specialty",
    "n_periods", "n_choices", "n_unique_modules",
    "avg_history_priority", "is_cold",
    # M — модуль (10)
    "pop_weighted_global", "pop_raw_global", "pop_by_institute",
    "pop_by_specialty",
    "pop_last_period", "pop_prev_period", "pop_first_period",
    "n_periods_available", "title_len", "desc_len",
    # S×M — взаимодействие (16, оба content-сигнала)
    "already_chosen", "cooc_score", "cooc_max", "cooc_norm",
    "cooc_ppmi_score", "cooc_ppmi_max", "cooc_ppmi_norm",
    "tfidf_cos_mean", "tfidf_cos_max",
    "embed_cos_mean", "embed_cos_max",
    "pop_ratio_in_institute", "pop_rank_in_available", "pop_rank_frac",
    "n_available", "was_available_before",
]

# v7: 28 признаков. v4 без 3 rank-фич (для ablation).
FEATURE_COLS_V7: list[str] = [
    # S — студент (7)
    "institute", "specialty",
    "n_periods", "n_choices", "n_unique_modules",
    "avg_history_priority", "is_cold",
    # M — модуль (10)
    "pop_weighted_global", "pop_raw_global", "pop_by_institute",
    "pop_by_specialty",
    "pop_last_period", "pop_prev_period", "pop_first_period",
    "n_periods_available", "title_len", "desc_len",
    # S×M — взаимодействие (11, без rank-3)
    "already_chosen", "cooc_score", "cooc_max", "cooc_norm",
    "cooc_ppmi_score", "cooc_ppmi_max", "cooc_ppmi_norm",
    "tfidf_cos_mean", "tfidf_cos_max",
    "n_available", "was_available_before",
]

# v8: 30 признаков. v6 (оба content) без 3 rank-фич.
FEATURE_COLS_V8: list[str] = [
    # S — студент (7)
    "institute", "specialty",
    "n_periods", "n_choices", "n_unique_modules",
    "avg_history_priority", "is_cold",
    # M — модуль (10)
    "pop_weighted_global", "pop_raw_global", "pop_by_institute",
    "pop_by_specialty",
    "pop_last_period", "pop_prev_period", "pop_first_period",
    "n_periods_available", "title_len", "desc_len",
    # S×M — взаимодействие (13, оба content, без rank-3)
    "already_chosen", "cooc_score", "cooc_max", "cooc_norm",
    "cooc_ppmi_score", "cooc_ppmi_max", "cooc_ppmi_norm",
    "tfidf_cos_mean", "tfidf_cos_max",
    "embed_cos_mean", "embed_cos_max",
    "n_available", "was_available_before",
]

# v11: 34 признака. v4 + 3 глобальные rank-фичи.
# Гипотеза: pop_rank_in_available вырождается для cold (нет института).
# Глобальные rank-фичи (по pop_weighted) остаются осмысленными.
FEATURE_COLS_V11: list[str] = [
    # S — студент (7)
    "institute", "specialty",
    "n_periods", "n_choices", "n_unique_modules",
    "avg_history_priority", "is_cold",
    # M — модуль (10)
    "pop_weighted_global", "pop_raw_global", "pop_by_institute",
    "pop_by_specialty",
    "pop_last_period", "pop_prev_period", "pop_first_period",
    "n_periods_available", "title_len", "desc_len",
    # S×M — взаимодействие (14 старых + 3 новых = 17)
    "already_chosen", "cooc_score", "cooc_max", "cooc_norm",
    "cooc_ppmi_score", "cooc_ppmi_max", "cooc_ppmi_norm",
    "tfidf_cos_mean", "tfidf_cos_max",
    "pop_ratio_in_institute", "pop_rank_in_available", "pop_rank_frac",
    "n_available", "was_available_before",
    # --- v11: глобальные rank-фичи ---
    "pop_rank_global",
    "pop_rank_global_in_avail",
    "pop_rank_global_frac",
]

# v12: 31 признак. v7 + 3 глобальные rank-фичи.
# = ЗАМЕНА вырождающихся rank-3 на глобальные.
FEATURE_COLS_V12: list[str] = [
    # S — студент (7)
    "institute", "specialty",
    "n_periods", "n_choices", "n_unique_modules",
    "avg_history_priority", "is_cold",
    # M — модуль (10)
    "pop_weighted_global", "pop_raw_global", "pop_by_institute",
    "pop_by_specialty",
    "pop_last_period", "pop_prev_period", "pop_first_period",
    "n_periods_available", "title_len", "desc_len",
    # S×M — взаимодействие (11 без rank-3 + 3 новых = 14)
    "already_chosen", "cooc_score", "cooc_max", "cooc_norm",
    "cooc_ppmi_score", "cooc_ppmi_max", "cooc_ppmi_norm",
    "tfidf_cos_mean", "tfidf_cos_max",
    "n_available", "was_available_before",
    # --- v12: глобальные rank-фичи ---
    "pop_rank_global",
    "pop_rank_global_in_avail",
    "pop_rank_global_frac",
]

# Удобный словарь для CLI: имя набора → список колонок.
FEATURE_SETS: dict[str, list[str]] = {
    "v4": FEATURE_COLS_V4,
    "v5": FEATURE_COLS_V5,
    "v6": FEATURE_COLS_V6,
    "v7": FEATURE_COLS_V7,
    "v8": FEATURE_COLS_V8,
    "v11": FEATURE_COLS_V11,
    "v12": FEATURE_COLS_V12,
}

# Соответствие content_backend → какие feature-set'ы с ним совместимы.
COMPATIBLE_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "tfidf": ("v4", "v6", "v7", "v8", "v11", "v12"),
    "embed": ("v5", "v6", "v8"),
    "both":  ("v4", "v5", "v6", "v7", "v8", "v11", "v12"),
    "none":  ("v4", "v7", "v11", "v12"),
}

CATEGORICAL_FEATS: list[str] = ["institute", "specialty"]

# label_gain[i] = вес для label i
LABEL_GAIN: list[float] = [0.0, 0.35, 0.45, 0.60, 0.80, 1.00]

WEIGHTS_BY_LABEL: np.ndarray = np.array(LABEL_GAIN, dtype=np.float64)

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "lambdarank",
    "metric": "None",
    "label_gain": LABEL_GAIN,
    "lambdarank_truncation_level": 15,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "max_bin": 255,
    "verbosity": -1,
    "seed": 42,
    "num_threads": 0,
}


# ---------------------------------------------------------------------------
# Custom eval: наш pNDCG@5
# ---------------------------------------------------------------------------
def _pndcg5_from_arrays(
    preds: np.ndarray,
    labels: np.ndarray,
    group_sizes: np.ndarray,
) -> float:
    n_groups = len(group_sizes)

    if np.all(group_sizes == group_sizes[0]):
        n_per_group = int(group_sizes[0])
        preds_2d = preds.reshape(n_groups, n_per_group)
        labels_2d = labels.reshape(n_groups, n_per_group)

        top_idx = np.argsort(-preds_2d, axis=1, kind="stable")[:, :5]
        top_labels = np.take_along_axis(labels_2d, top_idx, axis=1)

        gains = WEIGHTS_BY_LABEL[top_labels]

        positions = np.arange(1, 6, dtype=np.float64)
        discounts = 1.0 / np.log2(positions + 1)

        dcg = (gains * discounts[None, :]).sum(axis=1)
        dcg = np.minimum(dcg, 1.0)
        return float(dcg.mean())

    total = 0.0
    start = 0
    for size in group_sizes:
        end = start + int(size)
        p = preds[start:end]
        y = labels[start:end]
        order = np.argsort(-p, kind="stable")[:5]
        dcg = 0.0
        for pos, idx in enumerate(order, start=1):
            lab = int(y[idx])
            if lab > 0:
                dcg += WEIGHTS_BY_LABEL[lab] / math.log2(pos + 1)
        total += min(1.0, dcg)
        start = end
    return total / n_groups


def pndcg5_eval(
    preds: np.ndarray,
    eval_data: lgb.Dataset,
) -> tuple[str, float, bool]:
    labels = eval_data.get_label().astype(np.int32)
    group_sizes = np.asarray(eval_data.get_group(), dtype=np.int64)
    score = _pndcg5_from_arrays(preds, labels, group_sizes)
    return "pndcg@5", score, True


# ---------------------------------------------------------------------------
# Кодирование категориальных
# ---------------------------------------------------------------------------
def encode_categoricals(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    cat_cols: list[str] = CATEGORICAL_FEATS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Приводит категориальные колонки к общему набору категорий.
    """
    for c in cat_cols:
        all_vals = sorted(set(train_df[c].astype(str)) | set(val_df[c].astype(str)))
        dtype = pd.CategoricalDtype(categories=all_vals)
        train_df[c] = train_df[c].astype(str).astype(dtype)
        val_df[c] = val_df[c].astype(str).astype(dtype)
    return train_df, val_df


# ---------------------------------------------------------------------------
# Подготовка X, y, group, w
# ---------------------------------------------------------------------------
def prepare_xyg(
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    """Сортирует по student_id, отделяет X, y, group."""
    df_sorted = df.sort_values("student_id", kind="stable").reset_index(drop=True)
    X = df_sorted[feature_cols].copy()
    y = df_sorted["label"].to_numpy(dtype=np.int32)
    group_sizes = (
        df_sorted.groupby("student_id", sort=False).size().to_numpy(dtype=np.int32)
    )
    return X, y, group_sizes, df_sorted


def prepare_xygw(
    df: pd.DataFrame,
    weight_col: str = "weight",
    feature_cols: list[str] = FEATURE_COLS,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Как prepare_xyg, но возвращает ещё и веса."""
    df_sorted = df.sort_values("student_id", kind="stable").reset_index(drop=True)
    X = df_sorted[feature_cols].copy()
    y = df_sorted["label"].to_numpy(dtype=np.int32)
    group_sizes = (
        df_sorted.groupby("student_id", sort=False).size().to_numpy(dtype=np.int32)
    )
    if weight_col in df_sorted.columns:
        w = df_sorted[weight_col].to_numpy(dtype=np.float32)
    else:
        w = np.ones(len(df_sorted), dtype=np.float32)
    return X, y, group_sizes, w, df_sorted


# ---------------------------------------------------------------------------
# Обучение
# ---------------------------------------------------------------------------
def train_lgbm_ranker(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    group_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    group_val: np.ndarray,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 3000,
    early_stopping_rounds: int = 100,
    log_every: int = 25,
    sample_weight_train: np.ndarray | None = None,
    sample_weight_val: np.ndarray | None = None,
) -> lgb.Booster:
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)

    train_ds = lgb.Dataset(
        X_train, label=y_train, group=group_train,
        weight=sample_weight_train,
        categorical_feature=CATEGORICAL_FEATS,
        free_raw_data=False,
    )
    val_ds = lgb.Dataset(
        X_val, label=y_val, group=group_val,
        weight=sample_weight_val,
        categorical_feature=CATEGORICAL_FEATS,
        reference=train_ds,
        free_raw_data=False,
    )

    callbacks = [
        lgb.log_evaluation(period=log_every),
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
    ]

    booster = lgb.train(
        params=p,
        train_set=train_ds,
        num_boost_round=num_boost_round,
        valid_sets=[val_ds],
        valid_names=["val"],
        feval=pndcg5_eval,
        callbacks=callbacks,
    )
    return booster


# ---------------------------------------------------------------------------
# Предсказание
# ---------------------------------------------------------------------------
def predict_scores(
    model: lgb.Booster,
    df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLS,
) -> np.ndarray:
    """Возвращает массив score (float), выровненный по строкам df.

    ВАЖНО: feature_cols должен совпадать с тем, на котором обучалась модель.
    """
    X = df[feature_cols]
    return model.predict(X, num_iteration=model.best_iteration)


def top_k_per_student(
    df: pd.DataFrame,
    scores: np.ndarray,
    k: int = 5,
) -> dict[str, list[str]]:
    assert len(df) == len(scores), "df и scores разошлись по длине"

    work = pd.DataFrame({
        "student_id": df["student_id"].values,
        "module_id": df["module_id"].values,
        "score": scores,
    })
    work = work.sort_values(
        ["student_id", "score", "module_id"],
        ascending=[True, False, True],
        kind="stable",
    )

    result: dict[str, list[str]] = {}
    for sid, group in work.groupby("student_id", sort=False):
        result[sid] = group["module_id"].head(k).tolist()
    return result


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
def feature_importance_report(
    model: lgb.Booster,
    feature_cols: list[str] = FEATURE_COLS,
    top_n: int = 20,
) -> pd.DataFrame:
    gain = model.feature_importance(importance_type="gain")
    split = model.feature_importance(importance_type="split")
    df = pd.DataFrame({
        "feature": feature_cols,
        "gain": gain,
        "split": split,
    }).sort_values("gain", ascending=False).reset_index(drop=True)
    return df.head(top_n)


# ---------------------------------------------------------------------------
# Сохранение / загрузка
# ---------------------------------------------------------------------------
def save_model(model: lgb.Booster, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path), num_iteration=model.best_iteration)
    print(f"[ltr_model] Модель сохранена: {path}")


def load_model(path: str | Path) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))