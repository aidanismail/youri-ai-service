from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import linear_kernel

BASE_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = BASE_DIR / "ml_engine" / "weights"

LOGGER = logging.getLogger(__name__)
SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s-]", flags=re.UNICODE)


def normalize_ingredient(value: object) -> str:
    text = PUNCT_RE.sub(" ", str(value or "").casefold())
    return SPACE_RE.sub(" ", text).strip(" -_")


def ingredient_to_token(ingredient: str) -> str:
    return ingredient.replace(" ", "_")


class LexicalMatcherModel:
    def __init__(
        self,
        weights_dir: str | Path = WEIGHTS_DIR,
        cosine_weight: float = 0.70,
        user_coverage_weight: float = 0.20,
        recipe_coverage_weight: float = 0.10,
        min_score: float = 0.01,
    ) -> None:
        self.weights_dir = Path(weights_dir)
        self.cosine_weight = cosine_weight
        self.user_coverage_weight = user_coverage_weight
        self.recipe_coverage_weight = recipe_coverage_weight
        self.min_score = min_score

        self.vectorizer = None
        self.recipe_matrix = None
        self.recipe_metadata: list[dict[str, Any]] = []
        self.recipe_ingredient_sets: list[frozenset[str]] = []
        self.recipe_ingredient_counts = np.array([], dtype=np.float32)
        self.ingredient_index: dict[str, np.ndarray] = {}
        self.load_error: str | None = None

        self._load_artifacts()

    @property
    def is_ready(self) -> bool:
        return self.vectorizer is not None and self.recipe_matrix is not None

    def _load_artifacts(self) -> None:
        try:
            self.vectorizer = joblib.load(self.weights_dir / "tfidf_vectorizer.pkl")
            self.recipe_matrix = joblib.load(self.weights_dir / "recipe_matrix.pkl")
            self.recipe_metadata = joblib.load(self.weights_dir / "recipe_metadata.pkl")
        except FileNotFoundError as exc:
            self.load_error = f"Artifacts belum tersedia: {exc.filename}"
            LOGGER.error("%s. Jalankan scripts/train_tfidf.py.", self.load_error)
            return
        except Exception as exc:
            self.load_error = f"Gagal memuat artifacts: {exc}"
            LOGGER.exception(self.load_error)
            return

        if sparse.issparse(self.recipe_matrix):
            self.recipe_matrix = self.recipe_matrix.tocsr().astype(np.float32)
            self.recipe_matrix.sort_indices()

        if len(self.recipe_metadata) != self.recipe_matrix.shape[0]:
            self.load_error = "Jumlah metadata tidak sama dengan jumlah baris matrix."
            LOGGER.error(self.load_error)
            self.vectorizer = None
            self.recipe_matrix = None
            return

        self._build_ingredient_index()
        LOGGER.info(
            "TF-IDF matcher siap: %s resep, %s fitur",
            self.recipe_matrix.shape[0],
            self.recipe_matrix.shape[1],
        )

    def _build_ingredient_index(self) -> None:
        ingredient_to_indices: dict[str, list[int]] = defaultdict(list)
        ingredient_sets: list[frozenset[str]] = []

        for index, metadata in enumerate(self.recipe_metadata):
            ingredients = metadata.get("ingredients") or []
            normalized = set()

            for ingredient in ingredients:
                ingredient = normalize_ingredient(ingredient)
                if ingredient:
                    normalized.add(ingredient)

            ingredient_sets.append(frozenset(normalized))
            for ingredient in normalized:
                ingredient_to_indices[ingredient].append(index)

        self.recipe_ingredient_sets = ingredient_sets
        self.recipe_ingredient_counts = np.array(
            [max(len(ingredients), 1) for ingredients in ingredient_sets],
            dtype=np.float32,
        )
        self.ingredient_index = {
            ingredient: np.asarray(indices, dtype=np.int32)
            for ingredient, indices in ingredient_to_indices.items()
        }

    def _extract_valid_ingredients(self, user_ingredients: list[Any]) -> list[str]:
        valid_ingredients: list[str] = []
        seen: set[str] = set()

        for item in user_ingredients or []:
            if isinstance(item, dict):
                if not item.get("is_valid", True):
                    continue
                raw_value = item.get("name") or item.get("cleaned_entity") or item.get("ingredient")
            else:
                raw_value = item

            ingredient = normalize_ingredient(raw_value)
            if ingredient and ingredient not in seen:
                seen.add(ingredient)
                valid_ingredients.append(ingredient)

        return valid_ingredients

    def _metadata_overlap_counts(self, ingredients: list[str]) -> np.ndarray:
        counts = np.zeros(len(self.recipe_metadata), dtype=np.float32)

        for ingredient in ingredients:
            indices = self.ingredient_index.get(ingredient)
            if indices is not None:
                counts[indices] += 1.0

        return counts

    def predict_match(
        self,
        user_ingredients: list[Any],
        top_k: int = 10,
        include_details: bool = False,
    ) -> list[dict[str, Any]]:
        if not self.is_ready:
            return [{"error": self.load_error or "Model belum siap di-load."}]

        ingredients = self._extract_valid_ingredients(user_ingredients)
        if not ingredients:
            return []

        top_k = max(1, min(int(top_k), len(self.recipe_metadata)))
        query = " ".join(ingredient_to_token(ingredient) for ingredient in ingredients)
        user_vector = self.vectorizer.transform([query])

        cosine_scores = linear_kernel(user_vector, self.recipe_matrix).ravel().astype(np.float32)
        overlap_counts = self._metadata_overlap_counts(ingredients)
        user_coverage = overlap_counts / max(len(ingredients), 1)
        recipe_coverage = overlap_counts / self.recipe_ingredient_counts

        scores = (
            self.cosine_weight * cosine_scores
            + self.user_coverage_weight * user_coverage
            + self.recipe_coverage_weight * recipe_coverage
        )

        candidate_indices = np.flatnonzero(scores > self.min_score)
        if candidate_indices.size == 0:
            return []

        if candidate_indices.size > top_k:
            top_positions = np.argpartition(scores[candidate_indices], -top_k)[-top_k:]
            candidate_indices = candidate_indices[top_positions]

        candidate_indices = sorted(
            candidate_indices,
            key=lambda idx: (scores[idx], cosine_scores[idx], user_coverage[idx]),
            reverse=True,
        )[:top_k]

        results: list[dict[str, Any]] = []
        for index in candidate_indices:
            metadata = self.recipe_metadata[index]
            result = {
                "recipe_id": str(metadata["id"]),
                "title": metadata["title"],
                "match_percentage": int(round(min(float(scores[index]), 1.0) * 100)),
            }

            if include_details:
                recipe_ingredients = self.recipe_ingredient_sets[index]
                matched = [ingredient for ingredient in ingredients if ingredient in recipe_ingredients]
                result.update(
                    {
                        "url": metadata.get("url", ""),
                        "matched_ingredients": matched,
                        "missing_input_ingredients": [
                            ingredient for ingredient in ingredients if ingredient not in recipe_ingredients
                        ],
                        "similarity_score": round(float(cosine_scores[index]), 4),
                    }
                )

            results.append(result)

        return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    model = LexicalMatcherModel()
    sample_input = [
        {"name": "tempe", "is_valid": True},
        {"name": "bawang merah", "is_valid": True},
        {"name": "bawang putih", "is_valid": True},
        {"name": "serai", "is_valid": True},
        {"name": "paku payung", "is_valid": False},
    ]

    predictions = model.predict_match(sample_input, top_k=5, include_details=True)
    for rank, prediction in enumerate(predictions, start=1):
        print(f"{rank}. {prediction}")