"""
LightGBM Ranker — основная LTR-модель.

Использует objective='lambdarank' с кастомным label_gain и custom eval
pndcg5_eval (наша метрика с clip-1.0).

ДВЕ КОНЦЕПЦИИ (v47+):
    * LABEL_GAIN — веса для lambdarank objective (градиенты).
      v47 показал, что усиление priority-1 до 1.5 ВРЕДИТ (cold −0.031,
      warm −0.006). Откат к каноничному [0, 0.35, 0.45, 0.60, 0.80, 1.0].
    * METRIC_WEIGHTS_BY_LABEL — веса для custom eval и pndcg5_eval.
      Должны совпадать с официальной метрикой. НЕЛЬЗЯ менять.

Наборы признаков:
    v4, v5, v6, v7, v8, v11, v12, v14 — основные
    v15 = V11 − pop_rank_global_frac  (убрать дубль in_avail)
    v16 = V11 − title_len − desc_len  (убрать прокси ID модуля)
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
FEATURE_COLS: list[str] = [
    "institute", "specialty",
    "n_periods", "n_choices", "n_unique_modules",
    "avg_history_priority", "is_cold",
    "pop_weighted_global", "pop_raw_global", "pop_by_institute",
    "pop_by_specialty", "pop_period_1", "pop_period_2",
    "n_periods_available", "title_len", "desc_len",
    "already_chosen", "cooc_score", "cooc_max", "cooc_norm",
    "cooc_ppmi_score", "cooc_ppmi_max", "cooc_ppmi_norm",
    "tfidf_cos_mean", "tfidf_cos_max",
    "pop_ratio_in_institute", "pop_rank_in_available", "pop_rank_frac",
    "n_available", "was_available_before",
]

FEATURE_COLS_V4: list[str] = [
    "institute", "specialty",
    "n_periods", "n_choices", "n_unique_modules",
    "avg_history_priority", "is_cold",
    "pop_weighted_global", "pop_raw_global", "pop_by_institute",
    "pop_by_specialty",
    "pop_last_period", "pop_prev_period", "pop_first_period",
    "n_periods_available", "title_len", "desc_len",
    "already_chosen", "cooc_score", "cooc_max", "cooc_norm",
    "cooc_ppmi_score", "cooc_ppmi_max", "cooc_ppmi_norm",
    "tfidf_cos_mean", "tfidf_cos_max",
    "pop_ratio_in_institute", "pop_rank_in_available", "pop_rank_frac",
    "n_available", "was_available_before",
]

FEATURE_COLS_V5: list[str] = [
    "institute", "specialty",
    "n_periods", "n_choices", "n_unique_modules",
    "avg_history_priority", "is_cold",
    "pop_weighted_global", "pop_raw_global", "pop_by_institute",
    "pop_by_specialty",
    "pop_last_period", "pop_prev_period", "pop_first_period",
    "n_periods_available", "title_len", "desc_len",
    "already_chosen", "cooc_score", "cooc_max", "cooc_norm",
    "cooc_ppmi_score", "cooc_ppmi_max", "cooc_ppmi_norm",
    "embed_cos_mean", "embed_cos_max",
    "pop_ratio_in_institute", "pop_rank_in_available", "pop_rank_frac",
    "n_available", "was_available_before",
]

FEATURE_COLS_V6: list[str] = [
    "institute", "specialty",
    "n_periods", "n_choices", "n_unique_modules",
    "avg_history_priority", "is_cold",
    "pop_weighted_global", "pop_raw_global", "pop_by_institute",
    "pop_by_specialty",
    "pop_last_period", "pop_prev_period", "pop_first_period",
    "n_periods_available", "title_len", "desc_len",
    "already_chosen", "cooc_score", "cooc_max", "cooc_norm",
    "cooc_ppmi_score", "cooc_ppmi_max", "cooc_ppmi_norm",
    "tfidf_cos_mean", "tfidf_cos_max",
    "embed_cos_mean", "embed_cos_max",
    "pop_ratio_in_institute", "pop_rank_in_available", "pop_rank_frac",
    "n_available", "was_available_before",
]

FEATURE_COLS_V7: list[str] = [
    "institute", "specialty",
    "n_periods", "n_choices", "n_unique_modules",
    "avg_history_priority", "is_cold",
    "pop_weighted_global", "pop_raw_global", "pop_by_institute",
    "pop_by_specialty",
    "pop_last_period", "pop_prev_period", "pop_first_period",
    "n_periods_available", "title_len", "desc_len",
    "already_chosen", "cooc_score", "cooc_max", "cooc_norm",
    "cooc_ppmi_score", "cooc_ppmi_max", "cooc_ppmi_norm",
    "tfidf_cos_mean", "tfidf_cos_max",
    "n_available", "was_available_before",
]

FEATURE_COLS_V8: list[str] = [
    "institute", "specialty",
    "n_periods", "n_choices", "n_unique_modules",
    "avg_history_priority", "is_cold",
    "pop_weighted_global", "pop_raw_global", "pop_by_institute",
    "pop_by_specialty",
    "pop_last_period", "pop_prev_period", "pop_first_period",
    "n_periods_available", "title_len", "desc_len",
    "already_chosen", "cooc_score", "cooc_max", "cooc_norm",
    "cooc_ppmi_score", "cooc_ppmi_max", "cooc_ppmi_norm",
    "tfidf_cos_mean", "tfidf_cos_max",
    "embed_cos_mean", "embed_cos_max",
    "n_available", "was_available_before",
]

FEATURE_COLS_V11: list[str] = [
    "institute", "specialty",
    "n_periods", "n_choices", "n_unique_modules",
    "avg_history_priority", "is_cold",
    "pop_weighted_global", "pop_raw_global", "pop_by_institute",
    "pop_by_specialty",
    "pop_last_period", "pop_prev_period", "pop_first_period",
    "n_periods_available", "title_len", "desc_len",
    "already_chosen", "cooc_score", "cooc_max", "cooc_norm",
    "cooc_ppmi_score", "cooc_ppmi_max", "cooc_ppmi_norm",
    "tfidf_cos_mean", "tfidf_cos_max",
    "pop_ratio_in_institute", "pop_rank_in_available", "pop_rank_frac",
    "n_available", "was_available_before",
    "pop_rank_global",
    "pop_rank_global_in_avail",
    "pop_rank_global_frac",
]

FEATURE_COLS_V12: list[str] = [
    "institute", "specialty",
    "n_periods", "n_choices", "n_unique_modules",
    "avg_history_priority", "is_cold",
    "pop_weighted_global", "pop_raw_global", "pop_by_institute",
    "pop_by_specialty",
    "pop_last_period", "pop_prev_period", "pop_first_period",
    "n_periods_available", "title_len", "desc_len",
    "already_chosen", "cooc_score", "cooc_max", "cooc_norm",
    "cooc_ppmi_score", "cooc_ppmi_max", "cooc_ppmi_norm",
    "tfidf_cos_mean", "tfidf_cos_max",
    "n_available", "was_available_before",
    "pop_rank_global",
    "pop_rank_global_in_avail",
    "pop_rank_global_frac",
]

FEATURE_COLS_V14: list[str] = [
    "institute", "specialty",
    "n_periods", "n_choices", "n_unique_modules",
    "avg_history_priority", "is_cold",
    "pop_weighted_global", "pop_raw_global", "pop_by_institute",
    "pop_by_specialty",
    "pop_last_period", "pop_prev_period", "pop_first_period",
    "pop_delta", "pop_delta_norm",
    "n_periods_available", "title_len", "desc_len",
    "already_chosen", "cooc_score", "cooc_max", "cooc_norm",
    "cooc_ppmi_score", "cooc_ppmi_max", "cooc_ppmi_norm",
    "tfidf_cos_mean", "tfidf_cos_max",
    "pop_ratio_in_institute", "pop_rank_in_available", "pop_rank_frac",
    "n_available", "was_available_before",
    "pop_rank_global",
    "pop_rank_global_in_avail",
    "pop_rank_global_frac",
    "pop_delta_rank",
]

# v15: V11 − pop_rank_global_frac
# Для фиксированного n_available (158/159) pop_rank_global_frac =
# pop_rank_global_in_avail / n_available. Это ПОЛНЫЙ дубль (линейное
# преобразование с константой). Одна из двух фич избыточна.
FEATURE_COLS_V15: list[str] = [
    "institute", "specialty",
    "n_periods", "n_choices", "n_unique_modules",
    "avg_history_priority", "is_cold",
    "pop_weighted_global", "pop_raw_global", "pop_by_institute",
    "pop_by_specialty",
    "pop_last_period", "pop_prev_period", "pop_first_period",
    "n_periods_available", "title_len", "desc_len",
    "already_chosen", "cooc_score", "cooc_max", "cooc_norm",
    "cooc_ppmi_score", "cooc_ppmi_max", "cooc_ppmi_norm",
    "tfidf_cos_mean", "tfidf_cos_max",
    "pop_ratio_in_institute", "pop_rank_in_available", "pop_rank_frac",
    "n_available", "was_available_before",
    "pop_rank_global",
    "pop_rank_global_in_avail",
]

# v16: V11 − title_len − desc_len
# Гипотеза: длины текста проксируют ID модуля (короткие title = узкие
# специальные модули). Модель может overfit на этот прокси вместо
# содержательного сигнала. Убираем — смотрим, улучшится ли warm.
FEATURE_COLS_V16: list[str] = [
    "institute", "specialty",
    "n_periods", "n_choices", "n_unique_modules",
    "avg_history_priority", "is_cold",
    "pop_weighted_global", "pop_raw_global", "pop_by_institute",
    "pop_by_specialty",
    "pop_last_period", "pop_prev_period", "pop_first_period",
    "n_periods_available",
    "already_chosen", "cooc_score", "cooc_max", "cooc_norm",
    "cooc_ppmi_score", "cooc_ppmi_max", "cooc_ppmi_norm",
    "tfidf_cos_mean", "tfidf_cos_max",
    "pop_ratio_in_institute", "pop_rank_in_available", "pop_rank_frac",
    "n_available", "was_available_before",
    "pop_rank_global",
    "pop_rank_global_in_avail",
    "pop_rank_global_frac",
]

FEATURE_SETS: dict[str, list[str]] = {
    "v4": FEATURE_COLS_V4,
    "v5": FEATURE_COLS_V5,
    "v6": FEATURE_COLS_V6,
    "v7": FEATURE_COLS_V7,
    "v8": FEATURE_COLS_V8,
    "v11": FEATURE_COLS_V11,
    "v12": FEATURE_COLS_V12,
    "v14": FEATURE_COLS_V14,
    "v15": FEATURE_COLS_V15,
    "v16": FEATURE_COLS_V16,
}

COMPATIBLE_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "tfidf": ("v4", "v6", "v7", "v8", "v11", "v12", "v14", "v15", "v16"),
    "embed": ("v5", "v6", "v8"),
    "both":  ("v4", "v5", "v6", "v7", "v8", "v11", "v12", "v14", "v15", "v16"),
    "none":  ("v4", "v7", "v11", "v12", "v14", "v15", "v16"),
}

CATEGORICAL_FEATS: list[str] = ["institute", "specialty"]

# ---------------------------------------------------------------------------
# МЕТРИКА (не менять!)
# ---------------------------------------------------------------------------
# Веса приоритетов в НАСТОЯЩЕЙ метрике pNDCG@5. Совпадают с официальной
# формулой. Используются в custom eval и pndcg5_eval.
METRIC_WEIGHTS_BY_LABEL: np.ndarray = np.array(
    [0.0, 0.35, 0.45, 0.60, 0.80, 1.00],
    dtype=np.float64,
)

# Алиас для обратной совместимости.
WEIGHTS_BY_LABEL: np.ndarray = METRIC_WEIGHTS_BY_LABEL

# ---------------------------------------------------------------------------
# LABEL_GAIN — ОТКАЧЕНО к каноничному (v47 показал: усиление 1.5 → регресс)
# ---------------------------------------------------------------------------
LABEL_GAIN: list[float] = [0.0, 0.35, 0.45, 0.60, 0.80, 1.0]

# ---------------------------------------------------------------------------
# Параметры LightGBM (truncation=15 откачено)
# ---------------------------------------------------------------------------
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
# Custom eval: pNDCG@5 (использует METRIC_WEIGHTS_BY_LABEL)
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