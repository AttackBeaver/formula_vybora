"""
run_ltr_v46_two_stage.py — двухступенчатая модель: классификатор → ранкер.

АРХИТЕКТУРА:
    Stage 1: бинарный классификатор LightGBM на V11-фичах.
             Label = 1 если module_id попал в target (priority>0), иначе 0.
             Цель — отсеять явных «непопаданцев».
    Stage 2: LGBM lambdarank (как v41), но обучается и предсказывает
             только на top-K кандидатах по классификатору.

ЗАЧЕМ:
    У lambdarank 158 кандидатов на студента, из них 5 — target (3%).
    truncation_level=15 смотрит только top-15, а там много мусора.
    Классификатор вычищает пул до K=30 «осмысленных» → ранкер
    концентрируется на реальных кандидатах.

ПРОВЕРКА:
    Recall@K классификатора на target — если < 0.95, архитектура не
    может найти target в top-5 (риск потери). Если ≥ 0.95 — безопасно.

ЧТО СМОТРИМ:
    cold raw / warm raw / holdout двухступенчатой vs v41.
    Recall@K классификатора.

РЕШЕНИЕ:
    cold ≥ 0.337, warm ≥ 0.487, recall@K ≥ 0.95 → кандидат.
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.cold_start import identify_cold_students
from src.data_loader import load_all
from src.features import (
    FeatureStats,
    build_dataset,
    candidates_available,
    make_candidates_train_period1,
    make_candidates_train_period2,
    make_target_from_val_solution,
    target_period_1,
    target_period_2,
)
from src.ltr_model import (
    CATEGORICAL_FEATS,
    FEATURE_COLS_V11,
    WEIGHTS_BY_LABEL,
    encode_categoricals,
    predict_scores,
    prepare_xygw,
    save_model,
    top_k_per_student,
    train_lgbm_ranker,
)
from src.metric import evaluate_pndcg5
from src.popularity import compute_popularity
from src.submit import write_submission

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
MODELS_DIR = PROJECT_ROOT / "models"

WEIGHT_P1 = 1.0
WEIGHT_P0 = 0.5
WEIGHT_PMASK = 1.0
WEIGHT_P2 = 3.0

HOLDOUT_FRACTION = 0.20
HOLDOUT_SEED = 42
BASE_SEED = 42

FEATURE_COLS = FEATURE_COLS_V11
CONTENT_BACKEND = "tfidf"

RANKER_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
}
CLASSIFIER_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbosity": -1,
}
NUM_BOOST_RANKER = 2500
EARLY_STOP_RANKER = 100
NUM_BOOST_CLASSIFIER = 1000
EARLY_STOP_CLASSIFIER = 50

TOP_K = 30


def _n_periods(r: dict[str, Any]) -> int:
    return len(r.get("history") or [])


def _add_weight(df: pd.DataFrame, weight: float) -> pd.DataFrame:
    df = df.copy()
    df["weight"] = np.float32(weight)
    return df


def _segment_means(
    rec: dict[str, list[str]],
    gt_raw: dict[str, str],
    cold_ids: set[str],
    k: int = 5,
) -> tuple[float, float, float]:
    mean_all, per_student = evaluate_pndcg5(rec, gt_raw, k=k)
    cold_scores = [s for sid, s in per_student.items() if sid in cold_ids]
    warm_scores = [s for sid, s in per_student.items() if sid not in cold_ids]
    c = float(np.mean(cold_scores)) if cold_scores else 0.0
    w = float(np.mean(warm_scores)) if warm_scores else 0.0
    return mean_all, c, w


# ---------------------------------------------------------------------------
# Stage 1: классификатор
# ---------------------------------------------------------------------------
def train_classifier(
    X_train: pd.DataFrame,
    y_train_bin: np.ndarray,
    X_val: pd.DataFrame,
    y_val_bin: np.ndarray,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> lgb.Booster:
    params = dict(CLASSIFIER_PARAMS)
    params["seed"] = seed

    dtrain = lgb.Dataset(
        X_train, label=y_train_bin, weight=sample_weight,
        categorical_feature=CATEGORICAL_FEATS, free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_val, label=y_val_bin,
        categorical_feature=CATEGORICAL_FEATS,
        reference=dtrain, free_raw_data=False,
    )
    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=NUM_BOOST_CLASSIFIER,
        valid_sets=[dval],
        valid_names=["val"],
        callbacks=[
            lgb.log_evaluation(period=500),
            lgb.early_stopping(EARLY_STOP_CLASSIFIER, verbose=False),
        ],
    )
    return model


def recall_at_k_per_student(
    df: pd.DataFrame,
    scores: np.ndarray,
    k: int,
) -> float:
    """Recall@K: доля target-модулей, попавших в top-K кандидатов."""
    work = pd.DataFrame({
        "student_id": df["student_id"].values,
        "module_id": df["module_id"].values,
        "score": scores,
        "label": df["label"].values if "label" in df.columns else np.zeros(len(df)),
    })
    work = work.sort_values(
        ["student_id", "score"], ascending=[True, False], kind="stable",
    )
    recalls: list[float] = []
    for sid, grp in work.groupby("student_id", sort=False):
        top = grp.head(k)
        target_mods = set(grp[grp["label"] > 0]["module_id"].tolist())
        if not target_mods:
            continue
        hit = sum(1 for m in top["module_id"] if m in target_mods)
        recalls.append(hit / len(target_mods))
    return float(np.mean(recalls)) if recalls else 0.0


def filter_top_k_per_student(
    df: pd.DataFrame,
    scores: np.ndarray,
    k: int,
) -> pd.DataFrame:
    """Оставляет top-K строк per student по scores."""
    work = df.copy()
    work["_classifier_score"] = scores
    work = work.sort_values(
        ["student_id", "_classifier_score"],
        ascending=[True, False], kind="stable",
    )
    work = work.groupby("student_id", sort=False).head(k).reset_index(drop=True)
    return work.drop(columns=["_classifier_score"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="v46: two-stage classifier → ranker")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    args = ap.parse_args()

    n_seeds = max(1, int(args.n_seeds))
    seeds = [BASE_SEED + i for i in range(n_seeds)]
    top_k = int(args.top_k)

    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 78)
    print(f"LTR v46: two-stage (classifier → ranker)")
    print(f"  feature_set = V11 ({len(FEATURE_COLS)} признаков)")
    print(f"  content = '{CONTENT_BACKEND}', p2={WEIGHT_P2}")
    print(f"  Top-K классификатора: {top_k}")
    print(f"  Seeds: {seeds}")
    print("=" * 78)

    t0 = time.time()
    data = load_all()
    train = data["train"]
    val = data["val"]
    test = data["test"]
    modules = data["modules"]
    val_solution = data["val_solution"]
    val_gt_raw = dict(zip(val_solution["id"], val_solution["ground_truth"]))

    # ==== 1. FeatureStats ====
    print(f"\n[1/10] FeatureStats...")
    stats = FeatureStats.fit(train, modules, content_backend=CONTENT_BACKEND)
    stats.add_val_solution_popularity(val_solution)

    # ==== 2. Train-срезы ====
    print("\n[2/10] Train-срезы...")
    train_students_p1 = [
        s for s in train
        if any(h.get("period") == 2 for h in s.get("history", []))
    ]
    train_students_p0 = [
        s for s in train
        if any(h.get("period") == 1 for h in s.get("history", []))
    ]

    train_df_p1 = build_dataset(
        train_students_p1, stats,
        candidates_fn=make_candidates_train_period2(stats),
        target_fn=target_period_2,
        past_periods=[1],
    )
    train_df_p0 = build_dataset(
        train_students_p0, stats,
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
    print(f"      p1={len(train_df_p1):,}  p0={len(train_df_p0):,}  "
          f"pmask={len(train_df_pmask):,}")

    # ==== 3. Holdout + p2 ====
    print("\n[3/10] Holdout + p2...")
    val_p2 = [s for s in val if _n_periods(s) == 2]
    rng = random.Random(HOLDOUT_SEED)
    shuffled = list(val_p2)
    rng.shuffle(shuffled)
    n_holdout = int(len(shuffled) * HOLDOUT_FRACTION)
    holdout_records = shuffled[:n_holdout]
    p2_records = shuffled[n_holdout:]
    holdout_ids = {s["id"] for s in holdout_records}

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
    print(f"      p2={len(p2_records)}  holdout={len(holdout_records)}")

    # ==== 4. Val и test ====
    print("\n[4/10] Val и test...")
    val_df = build_dataset(
        val, stats,
        candidates_fn=candidates_available,
        target_fn=make_target_from_val_solution(val_gt_raw),
        past_periods=[1, 2],
    )
    test_df = build_dataset(
        test, stats,
        candidates_fn=candidates_available,
        target_fn=None,
        past_periods=None,
        max_history_periods=2,
    )
    print(f"      val={len(val_df):,}  test={len(test_df):,}")

    # ==== 5. Сборка ====
    print("\n[5/10] Сборка + кодирование...")
    train_df_p1 = _add_weight(train_df_p1, WEIGHT_P1)
    train_df_p0 = _add_weight(train_df_p0, WEIGHT_P0)
    train_df_pmask = _add_weight(train_df_pmask, WEIGHT_PMASK)
    p2_df = _add_weight(p2_df, WEIGHT_P2)
    holdout_df = _add_weight(holdout_df, 1.0)
    val_df = _add_weight(val_df, 1.0)
    test_df = _add_weight(test_df, 1.0)

    train_df_p1["student_id"] = "p1__" + train_df_p1["student_id"].astype(str)
    train_df_p0["student_id"] = "p0__" + train_df_p0["student_id"].astype(str)
    train_df_pmask["student_id"] = "pmask__" + train_df_pmask["student_id"].astype(str)
    p2_df["student_id"] = "p2__" + p2_df["student_id"].astype(str)
    holdout_df["student_id"] = "ho__" + holdout_df["student_id"].astype(str)

    train_df = pd.concat(
        [train_df_p1, train_df_p0, train_df_pmask, p2_df],
        ignore_index=True,
    )

    n_tr = len(train_df)
    n_ho = len(holdout_df)
    n_va = len(val_df)
    all_df = pd.concat([train_df, holdout_df, val_df, test_df], ignore_index=True)
    all_df, _ = encode_categoricals(all_df, all_df, CATEGORICAL_FEATS)

    train_df = all_df.iloc[:n_tr].reset_index(drop=True)
    holdout_df = all_df.iloc[n_tr:n_tr + n_ho].reset_index(drop=True)
    val_df = all_df.iloc[n_tr + n_ho:n_tr + n_ho + n_va].reset_index(drop=True)
    test_df = all_df.iloc[n_tr + n_ho + n_va:].reset_index(drop=True)

    # ==== 6. Матрицы ====
    print("\n[6/10] Матрицы X, y, g, w...")
    X_train, y_train, g_train, w_train, _ = prepare_xygw(
        train_df, feature_cols=FEATURE_COLS)
    X_ho, y_ho, g_ho, w_ho, ho_sorted = prepare_xygw(
        holdout_df, feature_cols=FEATURE_COLS)
    X_val, y_val, g_val, w_val, val_sorted = prepare_xygw(
        val_df, feature_cols=FEATURE_COLS)
    _, _, _, _, test_sorted = prepare_xygw(test_df, feature_cols=FEATURE_COLS)
    print(f"      Train X={X_train.shape}, групп={len(g_train)}")

    # ==== 7. Stage 1: classifier (n seeds) ====
    print(f"\n[7/10] Stage 1: classifier {n_seeds} seeds...")
    clf_scores_val_list: list[np.ndarray] = []
    clf_scores_ho_list: list[np.ndarray] = []
    clf_scores_test_list: list[np.ndarray] = []
    last_clf = None

    y_train_bin = (y_train > 0).astype(np.int32)
    y_ho_bin = (y_ho > 0).astype(np.int32)

    for i, seed in enumerate(seeds):
        print(f"\n      --- Classifier seed={seed} ---")
        t = time.time()
        clf = train_classifier(
            X_train, y_train_bin,
            X_ho, y_ho_bin,
            seed=seed,
            sample_weight=w_train,
        )
        print(f"      clf seed={seed}: {time.time() - t:.1f} сек, "
              f"best_iter={clf.best_iteration}")

        clf_scores_val_list.append(
            clf.predict(X_val, num_iteration=clf.best_iteration))
        clf_scores_ho_list.append(
            clf.predict(X_ho, num_iteration=clf.best_iteration))
        clf_scores_test_list.append(
            clf.predict(test_sorted[FEATURE_COLS], num_iteration=clf.best_iteration))
        last_clf = clf

    clf_scores_val = np.mean(clf_scores_val_list, axis=0)
    clf_scores_ho = np.mean(clf_scores_ho_list, axis=0)
    clf_scores_test = np.mean(clf_scores_test_list, axis=0)

    if last_clf is not None:
        save_model(last_clf, MODELS_DIR / "lgbm_classifier_v46.txt")

    # Проверка Recall@K
    recall_val = recall_at_k_per_student(val_df, clf_scores_val, k=top_k)
    recall_ho = recall_at_k_per_student(holdout_df, clf_scores_ho, k=top_k)
    print(f"\n      Recall@{top_k} (val)     = {recall_val:.4f}")
    print(f"      Recall@{top_k} (holdout) = {recall_ho:.4f}")
    if recall_val < 0.95:
        print(f"      ⚠️  Recall@{top_k} < 0.95 — риск потери target!")

    # ==== 8. Stage 2: ranker на top-K ====
    print(f"\n[8/10] Stage 2: ranker на top-{top_k}...")

    # Фильтруем train/val/holdout/test до top-K кандидатов
    train_sorted_full = train_df.sort_values("student_id", kind="stable").reset_index(drop=True)
    clf_scores_train = last_clf.predict(
        train_sorted_full[FEATURE_COLS],
        num_iteration=last_clf.best_iteration,
    )
    train_df_topk = filter_top_k_per_student(train_df, clf_scores_train, k=top_k)
    val_df_topk = filter_top_k_per_student(val_df, clf_scores_val, k=top_k)
    ho_df_topk = filter_top_k_per_student(holdout_df, clf_scores_ho, k=top_k)
    test_df_topk = filter_top_k_per_student(test_df, clf_scores_test, k=top_k)

    print(f"      top-{top_k}: "
          f"train={len(train_df_topk):,}  val={len(val_df_topk):,}  "
          f"ho={len(ho_df_topk):,}  test={len(test_df_topk):,}")

    X_train_k, y_train_k, g_train_k, w_train_k, _ = prepare_xygw(
        train_df_topk, feature_cols=FEATURE_COLS)
    X_ho_k, y_ho_k, g_ho_k, w_ho_k, ho_sorted_k = prepare_xygw(
        ho_df_topk, feature_cols=FEATURE_COLS)
    X_val_k, y_val_k, g_val_k, w_val_k, val_sorted_k = prepare_xygw(
        val_df_topk, feature_cols=FEATURE_COLS)
    _, _, _, _, test_sorted_k = prepare_xygw(
        test_df_topk, feature_cols=FEATURE_COLS)

    ranker_scores_val_list: list[np.ndarray] = []
    ranker_scores_test_list: list[np.ndarray] = []
    ranker_scores_ho_list: list[np.ndarray] = []

    for i, seed in enumerate(seeds):
        print(f"\n      --- Ranker seed={seed} ---")
        params = {**RANKER_PARAMS, "seed": seed}
        t = time.time()
        model = train_lgbm_ranker(
            X_train_k, y_train_k, g_train_k,
            X_ho_k, y_ho_k, g_ho_k,
            params=params,
            num_boost_round=NUM_BOOST_RANKER,
            early_stopping_rounds=EARLY_STOP_RANKER,
            log_every=500,
            sample_weight_train=w_train_k,
            sample_weight_val=w_ho_k,
        )
        print(f"      ranker seed={seed}: {time.time() - t:.1f} сек, "
              f"best_iter={model.best_iteration}")
        ranker_scores_ho_list.append(
            predict_scores(model, ho_sorted_k, feature_cols=FEATURE_COLS))
        ranker_scores_val_list.append(
            predict_scores(model, val_sorted_k, feature_cols=FEATURE_COLS))
        ranker_scores_test_list.append(
            predict_scores(model, test_sorted_k, feature_cols=FEATURE_COLS))

    ranker_scores_ho = np.mean(ranker_scores_ho_list, axis=0)
    ranker_scores_val = np.mean(ranker_scores_val_list, axis=0)
    ranker_scores_test = np.mean(ranker_scores_test_list, axis=0)

    # ==== 9. Оценка ====
    print("\n[9/10] Оценка two-stage...")

    cold_ids_val = identify_cold_students(val)
    gt_ho = {sid: gt for sid, gt in val_gt_raw.items() if sid in holdout_ids}

    rec_ho = top_k_per_student(ho_sorted_k, ranker_scores_ho, k=5)
    rec_ho_clean = {sid.replace("ho__", ""): v for sid, v in rec_ho.items()}
    m_ho, _ = evaluate_pndcg5(rec_ho_clean, gt_ho, k=5)

    rec_val = top_k_per_student(val_sorted_k, ranker_scores_val, k=5)
    _, m_cold, m_warm = _segment_means(rec_val, val_gt_raw, cold_ids_val)

    print(f"  Two-stage:")
    print(f"    holdout = {m_ho:.4f}  cold raw = {m_cold:.4f}  "
          f"warm raw = {m_warm:.4f}")

    # ==== 10. Test + сохранить ====
    print("\n[10/10] Инференс test + сохранение...")
    rec_test = top_k_per_student(test_sorted_k, ranker_scores_test, k=5)

    test_path = SUBMISSIONS_DIR / "submission_v46_two_stage_raw.csv"
    write_submission(rec_test, test_path)
    print(f"      Сохранено: {test_path}  ({len(rec_test)} строк)")

    # ==== Вердикт ====
    print("\n" + "=" * 78)
    print("ИТОГ: v46 (two-stage)")
    print("=" * 78)
    print(f"  v41 (champion):     cold=0.3372  warm=0.4839  public=0.3604")
    print(f"  Two-stage:          cold={m_cold:.4f}  warm={m_warm:.4f}")
    print(f"  Recall@{top_k} (val) = {recall_val:.4f}")
    print()

    if m_cold >= 0.337 and m_warm >= 0.487 and recall_val >= 0.95:
        print(f"  ✅ РЕКОМЕНДАЦИЯ: cold={m_cold:.4f}≥0.337, warm={m_warm:.4f}≥0.487, "
              f"recall≥0.95 — КАНДИДАТ")
        print(f"     файл: {test_path.name}")
    elif m_cold >= 0.337 and m_warm >= 0.483:
        print(f"  ⚠️  ОСТОРОЖНО: two-stage ≈ v41. Не тратим слот.")
    elif recall_val < 0.95:
        print(f"  ❌ Recall@{top_k}={recall_val:.4f} < 0.95 — классификатор "
              f"теряет target. Увеличить top_k или отбрасываем.")
    else:
        print(f"  ❌ НЕ САБМИТИМ: two-stage регресс")
    print("=" * 78)
    print(f"\n[OK] v46 завершён за {time.time() - t0:.1f} сек.")


if __name__ == "__main__":
    main()