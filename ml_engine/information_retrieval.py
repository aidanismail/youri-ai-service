from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

BASE_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = BASE_DIR / "ml_engine" / "weights"
DATA_PATH = BASE_DIR / "data" / "final_recipes_mongodb_v_final.json"
FLAVOR_VECTOR_PATH = WEIGHTS_DIR / "flavor_vectors.pkl"

LOGGER = logging.getLogger(__name__)
SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s-]", flags=re.UNICODE)
UNKNOWN_GROUP = "Unknown"
DEFAULT_FLAVOR_VECTORS = {
    5759: np.array([0.8, 0.2, 0.1, 0.0, 0.5], dtype=np.float32),
    481: np.array([0.2, 0.8, 0.6, 0.1, 0.0], dtype=np.float32),
    2836: np.array([0.0, 0.1, 0.9, 0.1, 0.0], dtype=np.float32),
    6545: np.array([0.1, 0.6, 0.7, 0.0, 0.2], dtype=np.float32),
}


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

        self.vectorizer: Any | None = None
        self.recipe_matrix: sparse.csr_matrix | None = None
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
        paths = {
            "vectorizer": self.weights_dir / "tfidf_vectorizer.pkl",
            "matrix": self.weights_dir / "recipe_matrix.pkl",
            "metadata": self.weights_dir / "recipe_metadata.pkl",
        }
        missing = [path.name for path in paths.values() if not path.exists()]

        if missing:
            self.load_error = f"Artifacts belum tersedia: {', '.join(missing)}"
            LOGGER.error("%s. Jalankan scripts/train_tfidf.py.", self.load_error)
            return

        try:
            vectorizer = joblib.load(paths["vectorizer"])
            recipe_matrix = joblib.load(paths["matrix"])
            recipe_metadata = joblib.load(paths["metadata"])
        except Exception as exc:
            self.load_error = f"Gagal memuat artifacts: {exc}"
            LOGGER.exception(self.load_error)
            return

        if not hasattr(vectorizer, "transform"):
            self.load_error = "Artifact vectorizer tidak memiliki method transform."
            LOGGER.error(self.load_error)
            return

        if not sparse.issparse(recipe_matrix):
            recipe_matrix = sparse.csr_matrix(recipe_matrix, dtype=np.float32)
        else:
            recipe_matrix = recipe_matrix.tocsr().astype(np.float32)

        recipe_matrix.sort_indices()

        if not isinstance(recipe_metadata, list):
            self.load_error = "Artifact metadata harus berupa list."
            LOGGER.error(self.load_error)
            return

        if len(recipe_metadata) != recipe_matrix.shape[0]:
            self.load_error = "Jumlah metadata tidak sama dengan jumlah baris matrix."
            LOGGER.error(self.load_error)
            return

        if recipe_matrix.shape[0] == 0 or recipe_matrix.shape[1] == 0:
            self.load_error = "Matrix TF-IDF kosong."
            LOGGER.error(self.load_error)
            return

        self.vectorizer = vectorizer
        self.recipe_matrix = recipe_matrix
        self.recipe_metadata = recipe_metadata
        self._build_ingredient_index()

        LOGGER.info(
            "TF-IDF matcher siap: %s resep, %s fitur",
            recipe_matrix.shape[0],
            recipe_matrix.shape[1],
        )

    def _build_ingredient_index(self) -> None:
        ingredient_to_indices: dict[str, list[int]] = defaultdict(list)
        ingredient_sets: list[frozenset[str]] = []

        for index, metadata in enumerate(self.recipe_metadata):
            raw_ingredients = metadata.get("ingredients") or []
            normalized_ingredients = {
                ingredient
                for ingredient in (normalize_ingredient(value) for value in raw_ingredients)
                if ingredient
            }

            ingredient_sets.append(frozenset(normalized_ingredients))
            for ingredient in normalized_ingredients:
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

    @staticmethod
    def _extract_valid_ingredients(user_ingredients: list[Any]) -> list[str]:
        valid_ingredients: list[str] = []
        seen: set[str] = set()

        for item in user_ingredients or []:
            raw_value: Any
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

        assert self.vectorizer is not None
        assert self.recipe_matrix is not None

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
            metadata = self.recipe_metadata[int(index)]
            result = {
                "recipe_id": str(metadata.get("id", f"recipe-{int(index):05d}")),
                "title": str(metadata.get("title", "")),
                "match_percentage": int(round(min(float(scores[index]), 1.0) * 100)),
            }

            if include_details:
                recipe_ingredients = self.recipe_ingredient_sets[int(index)]
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


class SmartSubstitutionEngine:
    def __init__(
        self,
        data_path: str | Path = DATA_PATH,
        flavor_vectors_path: str | Path = FLAVOR_VECTOR_PATH,
        enforce_food_group: bool = True,
    ) -> None:
        self.data_path = Path(data_path)
        self.flavor_vectors_path = Path(flavor_vectors_path)
        self.enforce_food_group = enforce_food_group
        self.ingredient_kb: dict[str, dict[str, Any]] = {}
        self.flavor_vectors: dict[int, np.ndarray] = {}
        self.vector_dim = 5
        self.load_error: str | None = None

        self._build_knowledge_base()
        self._load_flavor_vectors()

    @property
    def is_ready(self) -> bool:
        return bool(self.ingredient_kb) and self.load_error is None

    def _build_knowledge_base(self) -> None:
        if not self.data_path.exists():
            self.load_error = f"Dataset tidak ditemukan: {self.data_path}"
            LOGGER.error(self.load_error)
            return

        try:
            with self.data_path.open("r", encoding="utf-8") as file:
                dataset = json.load(file)
        except json.JSONDecodeError as exc:
            self.load_error = f"Gagal parsing JSON dataset: {exc}"
            LOGGER.error(self.load_error)
            return
        except OSError as exc:
            self.load_error = f"Gagal membaca dataset: {exc}"
            LOGGER.error(self.load_error)
            return

        recipes = dataset.get("data") if isinstance(dataset, dict) else None
        if not isinstance(recipes, list):
            self.load_error = "Dataset harus memiliki key 'data' berisi list resep."
            LOGGER.error(self.load_error)
            return

        for recipe in recipes:
            if not isinstance(recipe, dict):
                continue

            mapped_ingredients = recipe.get("mapped_ingredients") or []
            if not isinstance(mapped_ingredients, list):
                continue

            for ingredient in mapped_ingredients:
                if not isinstance(ingredient, dict):
                    continue

                normalized_name = normalize_ingredient(ingredient.get("cleaned_entity"))
                if not normalized_name:
                    continue

                food_group = str(ingredient.get("food_group") or UNKNOWN_GROUP).strip() or UNKNOWN_GROUP
                flavordb_id = self._extract_flavordb_id(ingredient)
                existing = self.ingredient_kb.get(normalized_name)

                if existing is None:
                    self.ingredient_kb[normalized_name] = {
                        "food_group": food_group,
                        "flavordb_id": flavordb_id,
                    }
                    continue

                if existing["food_group"] == UNKNOWN_GROUP and food_group != UNKNOWN_GROUP:
                    existing["food_group"] = food_group

                if existing["flavordb_id"] is None and flavordb_id is not None:
                    existing["flavordb_id"] = flavordb_id

        if not self.ingredient_kb:
            self.load_error = "Knowledge base ingredient kosong."
            LOGGER.error(self.load_error)
            return

        LOGGER.info("Knowledge base substitution siap: %s bahan unik.", len(self.ingredient_kb))

    @staticmethod
    def _extract_flavordb_id(ingredient: dict[str, Any]) -> int | None:
        flavor_mapping = ingredient.get("flavordb_mapping")
        if not isinstance(flavor_mapping, dict):
            return None

        raw_id = flavor_mapping.get("flavordb_id")
        if raw_id is None:
            return None

        try:
            return int(raw_id)
        except (TypeError, ValueError):
            return None

    def _load_flavor_vectors(self) -> None:
        vectors: dict[int, np.ndarray] | None = None

        if self.flavor_vectors_path.exists():
            try:
                vectors = self._coerce_flavor_vectors(joblib.load(self.flavor_vectors_path))
            except Exception as exc:
                LOGGER.warning("Gagal memuat flavor vectors, memakai fallback: %s", exc)

        if not vectors:
            vectors = {key: value.copy() for key, value in DEFAULT_FLAVOR_VECTORS.items()}

        self.flavor_vectors = vectors
        self.vector_dim = len(next(iter(vectors.values()))) if vectors else self.vector_dim

    @staticmethod
    def _coerce_flavor_vectors(raw_vectors: Any) -> dict[int, np.ndarray]:
        if not isinstance(raw_vectors, dict):
            return {}

        vectors: dict[int, np.ndarray] = {}
        expected_dim: int | None = None

        for raw_key, raw_value in raw_vectors.items():
            try:
                key = int(raw_key)
                vector = np.asarray(raw_value, dtype=np.float32).reshape(-1)
            except (TypeError, ValueError):
                continue

            if vector.size == 0:
                continue

            expected_dim = expected_dim or int(vector.size)
            if vector.size != expected_dim:
                LOGGER.warning("Flavor vector %s dilewati karena dimensi tidak konsisten.", key)
                continue

            vectors[key] = vector

        return vectors

    def _get_flavor_vector(self, flavordb_id: int | None) -> np.ndarray:
        if flavordb_id is None:
            return np.zeros((1, self.vector_dim), dtype=np.float32)

        vector = self.flavor_vectors.get(flavordb_id)
        if vector is None:
            return np.zeros((1, self.vector_dim), dtype=np.float32)

        return np.asarray(vector, dtype=np.float32).reshape(1, -1)

    def find_best_substitutes(
        self,
        missing_ingredient: str,
        surplus_ingredients: list[str],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not self.is_ready:
            LOGGER.warning("Substitution engine belum siap: %s", self.load_error)
            return []

        norm_missing = normalize_ingredient(missing_ingredient)
        missing_data = self.ingredient_kb.get(norm_missing)

        if not missing_data:
            LOGGER.info("Bahan hilang '%s' tidak ditemukan di knowledge base.", norm_missing)
            return []

        missing_group = missing_data["food_group"]
        missing_vector = self._get_flavor_vector(missing_data["flavordb_id"])
        candidates: list[dict[str, Any]] = []
        seen: set[str] = set()

        for surplus in surplus_ingredients:
            norm_surplus = normalize_ingredient(surplus)
            if not norm_surplus or norm_surplus == norm_missing or norm_surplus in seen:
                continue

            seen.add(norm_surplus)
            surplus_data = self.ingredient_kb.get(norm_surplus)
            if not surplus_data:
                continue

            surplus_group = surplus_data["food_group"]
            same_known_group = (
                missing_group != UNKNOWN_GROUP
                and surplus_group != UNKNOWN_GROUP
                and missing_group == surplus_group
            )

            if self.enforce_food_group and missing_group != UNKNOWN_GROUP and surplus_group != UNKNOWN_GROUP:
                if not same_known_group:
                    continue

            surplus_vector = self._get_flavor_vector(surplus_data["flavordb_id"])
            flavor_score = 0.0
            if np.any(missing_vector) and np.any(surplus_vector):
                flavor_score = float(cosine_similarity(missing_vector, surplus_vector)[0][0])

            candidates.append(
                {
                    "substitute_name": surplus,
                    "normalized_name": norm_surplus,
                    "food_group": surplus_group,
                    "flavor_similarity_score": round(flavor_score, 4),
                    "same_food_group": same_known_group,
                }
            )

        candidates.sort(
            key=lambda item: (
                item["same_food_group"],
                item["flavor_similarity_score"],
                item["substitute_name"],
            ),
            reverse=True,
        )

        return candidates[: max(1, int(top_k))]
