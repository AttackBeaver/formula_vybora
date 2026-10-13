"""
Content-based модель на семантических эмбеддингах.

ЗАЧЕМ:
    Текущий src/content.py использует TF-IDF (bag-of-words) на 8000
    токенов. Для коротких текстов (title + description по 100-200
    символов) это даёт слабый сигнал: "психология" и "психика" для
    TF-IDF — разные слова, для эмбеддингов — рядом.

    Мы провели 12 экспериментов (v10-v28), все — с TF-IDF. Ни один
    не побил v7. Пора поменять сам content-сигнал.

ЧТО ДЕЛАЕТ:
    Загружает pretrained multilingual sentence-transformer, кодирует
    каждый модуль в 384-мерный вектор, L2-нормализует.
    Интерфейс совместим с ContentModel — те же методы:
        - student_profile_mean(history_modules) → вектор профиля
        - cosine_mean(profile, candidates) → np.ndarray [C]
        - cosine_max(history, candidates) → np.ndarray [C]

    Кэширует эмбеддинги в models/module_embeddings.npz — чтобы не
    пересчитывать каждый раз (первый запуск ~30 сек, потом мгновенно).

АРХИТЕКТУРА:
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    Это лучший по соотношению качество/размер для русского + английского.
    Размер ~470 MB (скачается один раз в ~/.cache/huggingface).

ПРОВЕРКА:
    python -m src.content_embed --check
    Выведет top-3 ближайших модуля к модулю m_001 и статистику эмбеддингов.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDINGS_PATH = MODELS_DIR / "module_embeddings.npz"

# Имя модели — multilingual MiniLM, лучшее соотношение качество/размер
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class ContentEmbedModel:
    """
    Семантическая модель на эмбеддингах sentence-transformer.
    Совместимый интерфейс с ContentModel из src.content.
    """

    def __init__(self) -> None:
        self.model: Any = None
        self.module_ids: list[str] = []
        self.module_to_idx: dict[str, int] = {}
        self.vectors: np.ndarray | None = None  # [n_modules, D], L2-норм

    # ------------------------------------------------------------------
    @classmethod
    def fit(cls, modules: list[dict[str, Any]]) -> "ContentEmbedModel":
        model = cls()
        model.module_ids = [m["module_id"] for m in modules]
        model.module_to_idx = {mid: i for i, mid in enumerate(model.module_ids)}

        MODELS_DIR.mkdir(exist_ok=True)

        # Проверяем кэш: сохранены ли эмбеддинги для этих же module_ids
        if EMBEDDINGS_PATH.exists():
            try:
                data = np.load(EMBEDDINGS_PATH, allow_pickle=True)
                cached_ids = list(data["module_ids"])
                cached_vecs = data["vectors"]
                if cached_ids == model.module_ids:
                    model.vectors = cached_vecs.astype(np.float32)
                    print(
                        f"[ContentEmbedModel] Загружены эмбеддинги из "
                        f"{EMBEDDINGS_PATH.name}: shape={model.vectors.shape}"
                    )
                    return model
                else:
                    print(
                        f"[ContentEmbedModel] Кэш устарел "
                        f"(module_ids не совпадают), пересчитываю..."
                    )
            except Exception as e:
                print(f"[ContentEmbedModel] Ошибка чтения кэша ({e}), "
                      f"пересчитываю...")

        # Загружаем sentence-transformer
        print(f"[ContentEmbedModel] Загружаю модель {MODEL_NAME}...")
        from sentence_transformers import SentenceTransformer
        model.model = SentenceTransformer(MODEL_NAME)

        # Кодируем тексты
        texts = [
            f"{m.get('title') or ''}. {m.get('description') or ''}"
            for m in modules
        ]
        print(f"[ContentEmbedModel] Кодирую {len(texts)} модулей...")
        vecs = model.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-нормализация
        ).astype(np.float32)
        model.vectors = vecs

        # Кэшируем
        np.savez(
            EMBEDDINGS_PATH,
            module_ids=np.array(model.module_ids, dtype=object),
            vectors=vecs,
        )
        print(f"[ContentEmbedModel] Сохранено в {EMBEDDINGS_PATH.name}")

        return model

    # ------------------------------------------------------------------
    def student_profile_mean(
        self,
        history_modules: list[str],
    ) -> np.ndarray | None:
        """Средний L2-нормализованный вектор истории."""
        idxs = [
            self.module_to_idx[mid]
            for mid in history_modules
            if mid in self.module_to_idx
        ]
        if not idxs:
            return None
        mean_vec = self.vectors[idxs].mean(axis=0)
        norm = float(np.linalg.norm(mean_vec))
        if norm < 1e-9:
            return None
        return (mean_vec / norm).astype(np.float32)

    # ------------------------------------------------------------------
    def cosine_mean(
        self,
        profile: np.ndarray | None,
        candidate_ids: list[str],
    ) -> np.ndarray:
        """Косинус между профилем и каждым кандидатом."""
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

    # ------------------------------------------------------------------
    def cosine_max(
        self,
        history_modules: list[str],
        candidate_ids: list[str],
    ) -> np.ndarray:
        """Максимальный косинус между кандидатом и модулями истории."""
        idxs_h = [
            self.module_to_idx[mid]
            for mid in history_modules
            if mid in self.module_to_idx
        ]
        if not idxs_h:
            return np.zeros(len(candidate_ids), dtype=np.float32)
        hist_matrix = self.vectors[idxs_h]  # [H, D]

        idxs_c = [self.module_to_idx.get(mid, -1) for mid in candidate_ids]
        sims = np.zeros(len(candidate_ids), dtype=np.float32)
        valid = [(i, idx) for i, idx in enumerate(idxs_c) if idx >= 0]
        if not valid:
            return sims
        cand_idxs = np.array([idx for _, idx in valid], dtype=np.int64)
        row_idxs = np.array([i for i, _ in valid], dtype=np.int64)
        cand_matrix = self.vectors[cand_idxs]  # [C, D]

        # cos: [C, H] = [C, D] @ [D, H]. Затем max по истории.
        cos = cand_matrix @ hist_matrix.T
        sims[row_idxs] = cos.max(axis=1)
        return sims

    # ------------------------------------------------------------------
    def similarity_matrix_topk(
        self,
        module_id: str,
        k: int = 3,
    ) -> list[tuple[str, float]]:
        """Top-k ближайших модулей к данному (для диагностики)."""
        if module_id not in self.module_to_idx:
            return []
        idx = self.module_to_idx[module_id]
        vec = self.vectors[idx]
        sims = self.vectors @ vec  # [N]
        # Исключаем сам модуль
        order = np.argsort(-sims)
        out: list[tuple[str, float]] = []
        for i in order:
            if i == idx:
                continue
            out.append((self.module_ids[i], float(sims[i])))
            if len(out) == k:
                break
        return out


# ---------------------------------------------------------------------------
# CLI: проверка
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Content embeddings (SentenceTransformer)")
    parser.add_argument("--check", action="store_true",
                        help="Проверить: top-3 соседа для m_001 и стат.")
    args = parser.parse_args()

    from src.data_loader import load_all
    data = load_all()
    modules = data["modules"]

    print("=" * 78)
    print("CONTENT EMBEDDINGS — проверка")
    print("=" * 78)
    print(f"Модулей: {len(modules)}")

    model = ContentEmbedModel.fit(modules)

    print(f"\nЭмбеддинги: shape={model.vectors.shape}")
    print(f"  mean norm = {np.linalg.norm(model.vectors, axis=1).mean():.4f}")
    print(f"  std norm  = {np.linalg.norm(model.vectors, axis=1).std():.4f}")
    print(f"  mean of dim-means = {model.vectors.mean(axis=1).mean():+.4f}")

    # Проверим top-3 соседа для нескольких модулей
    sample_ids = ["m_001", "m_050", "m_100"]
    for mid in sample_ids:
        if mid not in model.module_to_idx:
            continue
        neighbors = model.similarity_matrix_topk(mid, k=3)
        print(f"\n  Top-3 соседа для {mid}:")
        for nid, sim in neighbors:
            # Найдём title для наглядности
            title = next(
                (m.get("title", "")[:60] for m in modules if m["module_id"] == nid),
                "?"
            )
            print(f"    {nid}  cos={sim:.4f}  «{title}»")

    print("\n[OK] Content embeddings готовы.")


if __name__ == "__main__":
    main()