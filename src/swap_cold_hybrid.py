"""
swap_cold_hybrid.py — гибрид: cold от одной модели, warm от другой.

ПОЧЕМУ:
    v41 (champion, 0.3604): cold=0.3372, warm=0.4839 — баланс
    v43 (p0mask=1.0):       cold=0.3448 (+0.008), warm=0.4631 (−0.021)
    v44 (noGRA):            cold=0.2968 (−0.040), warm=0.4874 (+0.004)

    Каждая модель выигрывает в одном сегменте. Гибрид:
    cold от v43 (лучший cold), warm от v41 (стабильно хороший warm).

    Ожидание:
      cold: 0.3448 → public ≈ 0.314
      warm: 0.4839 → public ≈ 0.39
      composite ≈ 0.352·0.314 + 0.628·0.39 + 0.007 = 0.363

ЗАПУСК:
    python -m src.swap_cold_hybrid \\
        --cold-model lgbm_ranker_v42_p2w3p0_p0mask1p0.txt \\
        --warm-model lgbm_ranker_v41_v11_p2w3p0.txt

    По умолчанию — именно эти модели.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.cold_start import identify_cold_students
from src.data_loader import load_all
from src.features import FeatureStats, build_dataset, candidates_available
from src.ltr_model import (
    CATEGORICAL_FEATS,
    FEATURE_COLS_V11,
    encode_categoricals,
    load_model,
    predict_scores,
    prepare_xygw,
    top_k_per_student,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_COLD_MODEL = "lgbm_ranker_v42_p2w3p0_p0mask1p0.txt"
DEFAULT_WARM_MODEL = "lgbm_ranker_v41_v11_p2w3p0.txt"
OUTPUT_NAME = "submission_hybrid_coldV43_warmV41.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid: cold от одной модели, warm от другой")
    ap.add_argument("--cold-model", default=DEFAULT_COLD_MODEL)
    ap.add_argument("--warm-model", default=DEFAULT_WARM_MODEL)
    args = ap.parse_args()

    print("=" * 78)
    print("swap_cold_hybrid: cold от cold-model, warm от warm-model")
    print(f"  cold-model: {args.cold_model}")
    print(f"  warm-model: {args.warm_model}")
    print("=" * 78)

    t0 = time.time()
    data = load_all()
    train = data["train"]
    test = data["test"]
    modules = data["modules"]

    cold_path = MODELS_DIR / args.cold_model
    warm_path = MODELS_DIR / args.warm_model
    if not cold_path.exists():
        print(f"! Нет {cold_path}")
        return
    if not warm_path.exists():
        print(f"! Нет {warm_path}")
        return

    print(f"\n[1/5] Загрузка моделей...")
    model_cold = load_model(cold_path)
    model_warm = load_model(warm_path)

    print(f"\n[2/5] FeatureStats (tfidf)...")
    stats = FeatureStats.fit(train, modules, content_backend="tfidf")

    print(f"\n[3/5] Test датасет...")
    test_df = build_dataset(
        test, stats,
        candidates_fn=candidates_available,
        target_fn=None,
        past_periods=None,
        max_history_periods=2,
    )
    test_df["weight"] = 1.0
    print(f"  test={len(test_df):,}")

    print(f"\n[4/5] Инференс...")
    all_df, _ = encode_categoricals(test_df, test_df, CATEGORICAL_FEATS)
    test_df = all_df
    _, _, _, _, test_sorted = prepare_xygw(
        test_df, feature_cols=FEATURE_COLS_V11, weight_col="weight",
    )
    scores_cold = predict_scores(model_cold, test_sorted, feature_cols=FEATURE_COLS_V11)
    scores_warm = predict_scores(model_warm, test_sorted, feature_cols=FEATURE_COLS_V11)

    rec_cold = top_k_per_student(test_sorted, scores_cold, k=5)
    rec_warm = top_k_per_student(test_sorted, scores_warm, k=5)

    print(f"\n[5/5] Гибрид...")
    cold_ids = identify_cold_students(test)
    print(f"  cold-студентов в test: {len(cold_ids)}")

    hybrid: dict[str, list[str]] = {}
    n_cold_used = 0
    n_warm_used = 0
    for sid in rec_warm:
        if sid in cold_ids:
            hybrid[sid] = rec_cold[sid]
            n_cold_used += 1
        else:
            hybrid[sid] = rec_warm[sid]
            n_warm_used += 1
    print(f"  cold из {args.cold_model}: {n_cold_used}")
    print(f"  warm из {args.warm_model}: {n_warm_used}")

    out_path = SUBMISSIONS_DIR / OUTPUT_NAME
    df = pd.DataFrame({
        "id": list(hybrid.keys()),
        "recommendations": [" ".join(m) for m in hybrid.values()],
    })
    df.to_csv(out_path, index=False)
    print(f"\n  Сохранено: {out_path} ({len(df)} строк)")
    print(f"\n[OK] Завершено за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()