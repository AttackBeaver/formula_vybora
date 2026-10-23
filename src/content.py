"""
Content-based features: TF-IDF косинус между описаниями дисциплин.

Профиль студента = усреднённый L2-норм вектор модулей истории,
косинус с кандидатом = скалярное произведение.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class ContentModel:
    """TF-IDF модель для косинусной близости между описаниями модулей."""

    def __init__(self) -> None:
        self.vectorizer: TfidfVectorizer | None = None
        self.module_ids: list[str] = []
        self.module_to_idx: dict[str, int] = {}
        self.vectors: np.ndarray | None = None  # [n_modules, V], L2-норм

    @classmethod
    def fit(cls, modules: list[dict[str, Any]]) -> ContentModel:
        model = cls()
        model.module_ids = [m["module_id"] for m in modules]
        model.module_to_idx = {mid: i for i, mid in enumerate(model.module_ids)}

        # Точка между title и description — чтобы TF-IDF не склеивал границы.
        texts = [
            f"{m.get('title') or ''}. {m.get('description') or ''}"
            for m in modules
        ]

        model.vectorizer = TfidfVectorizer(
            max_features=8000,
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 1),
            sublinear_tf=True,
        )
        matrix = model.vectorizer.fit_transform(texts)  # [n_modules, V], sparse
        matrix_norm = normalize(matrix, norm="l2", axis=1)
        # Плотная 322×8000×4 байта ≈ 10 МБ — влезает.
        model.vectors = matrix_norm.toarray().astype(np.float32)
        return model

    def student_profile_mean(self, history_modules: list[str]) -> np.ndarray | None:
        """L2-норм средний вектор истории; None, если истории нет."""
        idxs = [self.module_to_idx[mid] for mid in history_modules if mid in self.module_to_idx]
        if not idxs:
            return None
        mean_vec = self.vectors[idxs].mean(axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm < 1e-9:
            return None
        return (mean_vec / norm).astype(np.float32)

    def cosine_mean(
        self,
        profile: np.ndarray | None,
        candidate_ids: list[str],
    ) -> np.ndarray:
        """Косинус между усреднённым профилем и каждым кандидатом."""
        if profile is None:
            return np.zeros(len(candidate_ids), dtype=np.float32)
        idxs = [self.module_to_idx.get(mid, -1) for mid in candidate_ids]
        sims = np.zeros(len(candidate_ids), dtype=np.float32)
        valid = [(i, idx) for i, idx in enumerate(idxs) if idx >= 0]
        if not valid:
            return sims
        cand_idxs = np.array([idx for _, idx in valid], dtype=np.int64)
        row_idxs = np.array([i for i, _ in valid], dtype=np.int64)
        sims[row_idxs] = self.vectors[cand_idxs] @ profile
        return sims

    def cosine_max(
        self,
        history_modules: list[str],
        candidate_ids: list[str],
    ) -> np.ndarray:
        """
        Максимальный косинус между кандидатом и каждым модулем истории.
        Острее, чем cosine_mean: ловит «хотя бы одну» похожую тему.
        """
        idxs_h = [
            self.module_to_idx[mid]
            for mid in history_modules
            if mid in self.module_to_idx
        ]
        if not idxs_h:
            return np.zeros(len(candidate_ids), dtype=np.float32)
        hist_matrix = self.vectors[idxs_h]  # [H, V]

        idxs_c = [self.module_to_idx.get(mid, -1) for mid in candidate_ids]
        sims = np.zeros(len(candidate_ids), dtype=np.float32)
        valid = [(i, idx) for i, idx in enumerate(idxs_c) if idx >= 0]
        if not valid:
            return sims
        cand_idxs = np.array([idx for _, idx in valid], dtype=np.int64)
        row_idxs = np.array([i for i, _ in valid], dtype=np.int64)
        cand_matrix = self.vectors[cand_idxs]  # [C, V]

        # cos: [C, H] = [C, V] @ [V, H]. Затем max по истории.
        cos = cand_matrix @ hist_matrix.T
        sims[row_idxs] = cos.max(axis=1)
        return sims