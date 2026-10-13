"""
Диагностика: используются ли embeddings в v6?

ЗАЧЕМ:
    KFold дал +0.0039 composite для v6 (v4 + embeddings), но std по
    фолдам = 0.014-0.025. Дельта меньше шума. Нужно проверить
    МЕХАНИСТИЧЕСКИ: реально ли LightGBM использует embed_cos_* для
    сплитов, или это перестановка из-за +2 фичей?

КАК:
    Строим ОДИН датасет со stats(content_backend="both"): заполнены
    все 4 content-колонки (tfidf_cos_mean/max и embed_cos_mean/max).
    Обучаем две модели на ОДНОМ датасете:
        - V4: 31 колонка (tfidf_cos_*), как базовая.
        - V6: 33 колонки (tfidf_cos_* + embed_cos_*).
    Печатаем top-15 features by gain для каждой + отдельно gain
    content-колонок.

ЧТО ЧИТАТЬ:
    * embed_cos_* в V6 имеют gain >= 100 и входят в top-15 → сигнал
      реален, дельта v6-v4 не шум.
    * embed_cos_* gain ~0 → LightGBM их проигнорировал, и дельта
      v6-v4 — это эффект «+2 фичей в пуле сплитов».
    * tfidf_cos_* в V6 имеют gain заметно меньше, чем в V4 →
      часть работы tfidf перетекла на embed (ожидаемо, комплементарность).

ОБУЧЕНИЕ:
    Только срез p1 (830k, 15 признаков + label) — этого достаточно,
    чтобы увидеть механику. Holdout-качество здесь НЕ важно, важны
    только gain'ы.

Запуск:
    python -m src.diag_content_importance
    python -m src.diag_content_importance --n-boost 400
"""
from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import pandas as pd

from src.data_loader import load_all
from src.features import (
    FeatureStats,
    build_dataset,
    make_candidates_train_period2,
    make_target_from_val_solution,
    target_period_2,
)
from src.ltr_model import (
    CATEGORICAL_FEATS,
    FEATURE_COLS_V4,
    FEATURE_COLS_V6,
    encode_categoricals,
    feature_importance_report,
    prepare_xygw,
    train_lgbm_ranker,
)

FAST_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambdarank_truncation_level": 15,
}


def _content_gains(model, feature_cols: list[str]) -> pd.DataFrame:
    """Gain'ы только по content-фичам, в порядке как в feature_cols."""
    gain = model.feature_importance(importance_type="gain")
    content_names = [
        "tfidf_cos_mean", "tfidf_cos_max",
        "embed_cos_mean", "embed_cos_max",
    ]
    rows = []
    for name in content_names:
        if name in feature_cols:
            idx = feature_cols.index(name)
            rows.append({"feature": name, "gain": float(gain[idx])})
    return pd.DataFrame(rows)


def _print_top(model, feature_cols: list[str], label: str, top_n: int = 15) -> None:
    print(f"\n  Top-{top_n} features by gain ({label}):")
    df = feature_importance_report(model, feature_cols=feature_cols, top_n=top_n)
    for _, row in df.iterrows():
        print(f"      {row['feature']:<28}  gain = {row['gain']:>14.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Диагностика gain content-фичей")
    parser.add_argument("--n-boost", type=int, default=400,
                        help="Фиксированное число бустингов (без early stopping)")
    args = parser.parse_args()

    print("=" * 78)
    print(f"Диагностика: используются ли embeddings в v6?")
    print(f"  content_backend = 'both' (заполнены и tfidf_cos_*, и embed_cos_*)")
    print(f"  обучаем V4 и V6 на ОДНОМ датасете, сравниваем gain'ы")
    print("=" * 78)

    t0 = time.time()
    data = load_all()
    train = data["train"]
    val = data["val"]
    modules = data["modules"]
    val_solution = data["val_solution"]
    val_gt_raw = dict(zip(val_solution["id"], val_solution["ground_truth"]))

    # ==== 1. Статистики (both) ====
    print("\n[1/4] Статистики (content_backend='both')...")
    stats = FeatureStats.fit(train, modules, content_backend="both")

    # ==== 2. Датасет: p1 (sanity check на 830k строк) ====
    print("\n[2/4] Строю p1-срез (830k строк)...")
    train_students_p1 = [
        s for s in train
        if any(h.get("period") == 2 for h in s.get("history", []))
    ]
    df_p1 = build_dataset(
        train_students_p1, stats,
        candidates_fn=make_candidates_train_period2(stats),
        target_fn=target_period_2,
        past_periods=[1],
    )
    df_p1["weight"] = np.float32(1.0)
    print(f"      p1 = {len(df_p1):,} строк, "
          f"уникальных студентов = {df_p1['student_id'].nunique():,}")

    # ==== 3. Единое кодирование категориальных ====
    df_p1, _ = encode_categoricals(df_p1, df_p1.copy(), CATEGORICAL_FEATS)

    # ==== 4. Обучение V4 и V6 на одном датасете ====
    print("\n[3/4] Обучаю V4 и V6 на ОДНОМ датасете...")

    X_full, y_full, g_full, w_full, sorted_df = prepare_xygw(
        df_p1, feature_cols=FEATURE_COLS_V4,  # набор колонок тут не важен
    )

    # Используем один и тот же train/val split (для early stopping),
    # но обучение фиксированное (n_boost), чтобы gain'ы были сравнимы.
    n = len(X_full)
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_val = max(1, n // 20)  # 5% на валидацию (только для reference)

    def _split_cols(cols: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        X = sorted_df[cols].copy()
        return X, y_full, g_full, w_full

    # --- V4 ---
    print("\n  --- V4 (31 признак, только tfidf_cos_*) ---")
    X_v4, y_v4, g_v4, w_v4 = _split_cols(FEATURE_COLS_V4)
    t_v4 = time.time()
    model_v4 = train_lgbm_ranker(
        X_v4, y_v4, g_v4,
        X_v4.iloc[:100].copy(), y_v4[:100], np.array([len(y_v4[:100])]),
        params={**FAST_PARAMS, "seed": 42},
        num_boost_round=args.n_boost,
        early_stopping_rounds=10000,  # фактически отключен
        log_every=200,
        sample_weight_train=w_v4,
        sample_weight_val=None,
    )
    print(f"      V4 обучен за {time.time() - t_v4:.1f} сек, "
          f"best_iter={model_v4.best_iteration}")

    # --- V6 ---
    print("\n  --- V6 (33 признака, tfidf_cos_* + embed_cos_*) ---")
    X_v6, y_v6, g_v6, w_v6 = _split_cols(FEATURE_COLS_V6)
    t_v6 = time.time()
    model_v6 = train_lgbm_ranker(
        X_v6, y_v6, g_v6,
        X_v6.iloc[:100].copy(), y_v6[:100], np.array([len(y_v6[:100])]),
        params={**FAST_PARAMS, "seed": 42},
        num_boost_round=args.n_boost,
        early_stopping_rounds=10000,
        log_every=200,
        sample_weight_train=w_v6,
        sample_weight_val=None,
    )
    print(f"      V6 обучен за {time.time() - t_v6:.1f} сек, "
          f"best_iter={model_v6.best_iteration}")

    # ==== 5. Отчёт ====
    print("\n[4/4] Feature importance")
    _print_top(model_v4, FEATURE_COLS_V4, "V4", top_n=15)
    _print_top(model_v6, FEATURE_COLS_V6, "V6", top_n=15)

    print("\n  --- Content-фичи: gain в каждой модели ---")
    g4 = _content_gains(model_v4, FEATURE_COLS_V4)
    g6 = _content_gains(model_v6, FEATURE_COLS_V6)

    print(f"\n      V4:")
    for _, row in g4.iterrows():
        print(f"          {row['feature']:<20}  gain = {row['gain']:>14.0f}")
    print(f"\n      V6:")
    for _, row in g6.iterrows():
        print(f"          {row['feature']:<20}  gain = {row['gain']:>14.0f}")

    # ==== 6. Вердикт ====
    print("\n" + "=" * 78)
    print("ВЕРДИКТ:")
    embed_in_v4 = g4[g4["feature"].str.startswith("embed")]["gain"].sum() if len(g4) else 0
    embed_in_v6 = g6[g6["feature"].str.startswith("embed")]["gain"].sum() if len(g6) else 0
    tfidf_in_v4 = g4[g4["feature"].str.startswith("tfidf")]["gain"].sum() if len(g4) else 0
    tfidf_in_v6 = g6[g6["feature"].str.startswith("tfidf")]["gain"].sum() if len(g6) else 0

    print(f"  embed gain:  V4={embed_in_v4:>14.0f}   V6={embed_in_v6:>14.0f}")
    print(f"  tfidf gain:  V4={tfidf_in_v4:>14.0f}   V6={tfidf_in_v6:>14.0f}")

    total_v6 = sum(model_v6.feature_importance(importance_type="gain"))
    embed_share = embed_in_v6 / max(1.0, total_v6)
    print(f"  доля embed в общем gain V6: {100 * embed_share:.3f}%")

    if embed_share > 0.005:
        print("\n  >> embed_cos_* РЕАЛЬНО используются (доля > 0.5% от gain).")
        print("  >> Гипотеза «+0.0039 — шум» опровергнута.")
        print("  >> Рекомендация: собрать финальный пайплайн v29 = V6 + multi-seed.")
    else:
        print("\n  >> embed_cos_* почти не используются (доля < 0.5%).")
        print("  >> Дельта v6-v4, вероятно, эффект «+2 фичей».")
        print("  >> Рекомендация: не спешить с сабмитом, копать глубже.")

    print(f"\n[OK] Диагностика завершена за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()