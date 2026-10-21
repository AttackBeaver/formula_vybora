"""
swap_cold_raw.py — заменить cold-строки в v41 сабмите на raw-LTR.

ЗАЧЕМ:
    v41 показал cold raw = 0.3372 (лучше fallback 0.2816 впервые в сессии).
    Fallback теперь ВРЕДИТ cold. Нужен сабмит без fallback.

ЧТО ДЕЛАЕТ:
    1. Загружает модель models/lgbm_ranker_v41_v11_p2w3p0.txt
    2. Строит test feature-датасет (V11-фичи)
    3. Инферирует raw-LTR предсказания
    4. Читает submission_v41_v11_p2w3p0.csv (где cold = fallback)
    5. Заменяет cold-строки на raw-LTR
    6. Сохраняет submission_v41_v11_p2w3p0_raw.csv

ЗАПУСК:
    python -m src.swap_cold_raw
"""
from __future__ import annotations

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

MODEL_NAME = "lgbm_ranker_v41_v11_p2w3p0.txt"
SUB_NAME = "submission_v41_v11_p2w3p0.csv"
OUT_NAME = "submission_v41_v11_p2w3p0_raw.csv"


def read_submission(path: Path) -> dict[str, list[str]]:
    df = pd.read_csv(path)
    id_col = "id" if "id" in df.columns else "student_id"
    col = "recommendations" if "recommendations" in df.columns else "prediction"
    return {
        str(sid): [t for t in str(raw).split() if t]
        for sid, raw in zip(df[id_col], df[col])
    }


def write_submission(rec: dict[str, list[str]], path: Path) -> None:
    df = pd.DataFrame({
        "id": list(rec.keys()),
        "recommendations": [" ".join(m) for m in rec.values()],
    })
    df.to_csv(path, index=False)
    print(f"  Сохранено: {path} ({len(df)} строк)")


def main() -> None:
    print("=" * 78)
    print("swap_cold_raw: v41 без fallback на cold")
    print("=" * 78)

    t0 = time.time()
    data = load_all()
    train = data["train"]
    test = data["test"]
    modules = data["modules"]

    model_path = MODELS_DIR / MODEL_NAME
    if not model_path.exists():
        print(f"! Нет {model_path}")
        return
    print(f"\n[1/5] Модель: {model_path.name}")
    model = load_model(model_path)

    print("\n[2/5] FeatureStats (tfidf)...")
    stats = FeatureStats.fit(train, modules, content_backend="tfidf")

    print("\n[3/5] Test датасет...")
    test_df = build_dataset(
        test, stats,
        candidates_fn=candidates_available,
        target_fn=None,
        past_periods=None,
        max_history_periods=2,
    )
    test_df["weight"] = 1.0
    print(f"  test={len(test_df):,}")

    print("\n[4/5] Кодирование + инференс...")
    all_df, _ = encode_categoricals(test_df, test_df, CATEGORICAL_FEATS)
    test_df = all_df
    _, _, _, _, test_sorted = prepare_xygw(
        test_df, feature_cols=FEATURE_COLS_V11, weight_col="weight",
    )
    scores = predict_scores(model, test_sorted, feature_cols=FEATURE_COLS_V11)
    rec_ltr = top_k_per_student(test_sorted, scores, k=5)

    print("\n[5/5] Замена cold-строк...")
    sub_path = SUBMISSIONS_DIR / SUB_NAME
    if not sub_path.exists():
        print(f"! Нет {sub_path}")
        return
    rec_sub = read_submission(sub_path)
    print(f"  v41 (с fallback): {len(rec_sub)} студентов")

    cold_ids = identify_cold_students(test)
    print(f"  cold в test: {len(cold_ids)}")

    n_replaced = 0
    for sid in cold_ids:
        if sid in rec_sub and sid in rec_ltr:
            rec_sub[sid] = rec_ltr[sid]
            n_replaced += 1
    print(f"  заменено: {n_replaced}")

    out_path = SUBMISSIONS_DIR / OUT_NAME
    write_submission(rec_sub, out_path)

    print(f"\n[OK] Завершено за {time.time() - t0:.1f} сек.")
    print(f"     Кандидат на сабмит: {out_path}")


if __name__ == "__main__":
    main()