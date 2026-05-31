from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from ml_engine.ingredient_normalization import (
    canonical_ingredient,
    ingredient_to_token,
    ingredient_variants,
    normalize_ingredient,
    substitution_group,
)

BASE_DIR = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = BASE_DIR / "ml_engine" / "weights"

LOGGER = logging.getLogger(__name__)
UNKNOWN_GROUP = "Unknown"
UNKNOWN_SUBSTITUTION_GROUP = "unknown"

SUBSTITUTION_COMPATIBILITY = {
    "allium": {"allium"},
    "aromatic_spice": {"aromatic_spice", "hot_spice"},
    "hot_spice": {"hot_spice", "aromatic_spice"},
    "aromatic_leaf": {"aromatic_leaf", "fresh_herb"},
    "fresh_herb": {"fresh_herb", "aromatic_leaf", "allium"},
    "egg": {"egg"},
    "meat": {"meat"},
    "offal": {"offal", "meat"},
    "processed_meat": {"processed_meat", "meat"},
    "seafood": {"seafood"},
    "plant_protein": {"plant_protein"},
    "legume": {"legume", "plant_protein"},
    "creamy_fat": {"creamy_fat", "cooking_fat"},
    "cooking_fat": {"cooking_fat", "creamy_fat"},
    "carb": {"carb"},
    "flour_starch": {"flour_starch", "carb"},
    "starchy_vegetable": {"starchy_vegetable", "vegetable", "carb"},
    "vegetable": {"vegetable", "starchy_vegetable", "strong_vegetable"},
    "strong_vegetable": {"strong_vegetable", "vegetable"},
    "mushroom": {"mushroom", "vegetable"},
    "acid": {"acid"},
    "fruit": {"fruit", "acid"},
    "condiment_sauce": {"condiment_sauce", "fermented_condiment"},
    "fermented_condiment": {"fermented_condiment", "condiment_sauce"},
    "basic_seasoning": {"basic_seasoning", "condiment_sauce"},
    "basic_liquid": {"basic_liquid"},
    "leavening": {"leavening"},
    "beverage": {"beverage", "basic_liquid"},
    "wrapper_leaf": {"wrapper_leaf"},
    "crunchy_topping": {"crunchy_topping"},
}
MIN_SUBSTITUTION_SCORE = 0.60
MIN_SUBSTITUTION_FLAVOR_SCORE = 0.35
FALLBACK_FOOD_GROUP_BLOCKLIST = {"Vegetables", "Other Ingredients"}

COMMON_INGREDIENTS = {
    # basic liquid
    "air",
    "beri air",
    "air matang",
    "air putih",
    "air secukup nya",
    "air es",
    "air hangat",
    "air merebus daging",
    "sesuai selera air secukupny",

    # garam / gula / campuran seasoning dasar
    "garam",
    "gula",
    "gula pasir",
    "gula merah",
    "gula merah sisir",
    "gula jawa",
    "gula putih",
    "blok gula jawa",
    "st gula putih",
    "secukupny gula merah",
    "sejimpit gula",
    "gulput",

    "garam gula",
    "gula garam",
    "garam gula merica bubuk",
    "gula garam lada",
    "gula garam penyedap rasa",
    "gula garam penyedap",
    "gula garam minyak",
    "garam gula penyedap",
    "garam gula secukup nya",
    "garam gula air",
    "garam gula merah air",
    "garam gula kaldu bubuk",
    "optional garam gula",
    "sesuai selera gulagaram merica",

    "lada garam",
    "garam lada",
    "garam merica bubuk",
    "merica garam",
    "garam royco",
    "royco garam air",
    "garam kaldu sapi bubuk",
    "garampenyedap rasa",
    "garam penyedap",
    "garam micin",
    "sdikit garam micin",
    "kecap garam",
    "sesuai selera garam",
    "st garam",
    "sejumput garam",
    "garam bila perlu",

    # minyak / lemak umum
    "minyak",
    "minyak goreng",
    "minyak menggoreng",
    "minyak menumis",
    "minyak goreng menumis",
    "minyak menumis bumbu",
    "minyak sayur",
    "olive oil",
    "minyak samin",
    "minyak wijen",
    "margarin",
    "mentega",
    "mentega blueband",
    "mentega butter",
    "mentega tawar",
    "munjung mentega",

    # lada / merica
    "lada",
    "lada bubuk",
    "lada hitam",
    "lada hitam kasar",
    "lada butiran",
    "lada putih",
    "lada putih bubuk",
    "ladaku",

    "merica",
    "merica bubuk",
    "merica hitam gerus",
    "merica butiran",
    "bubuk merica",
    "black pepper",
    "mrica",
    "sd teh mrica",
    "bumbu lada hitam",
    "saori saus lada hitam",

    # penyedap / kaldu / brand seasoning
    "penyedap rasa",
    "penyedap",
    "bumbu penyedap",
    "penyedap rasa ayam",
    "penyedap rasa sapi",
    "jika suka penyedap royco",

    "kaldu bubuk",
    "bubuk kaldu sapi",
    "kaldu jamur",
    "kaldu ayam bubuk",
    "kaldu jamur bubuk",
    "kaldu bubuk instan",
    "kaldu sapi bubuk",
    "kaldu sapi",
    "kaldu ayam",
    "kaldu sapi rebusan",
    "air kaldu ayam",
    "kaldu sapi saya royco",

    "royco",
    "royco ayam",
    "royco kaldu sapi ayam",
    "masako",
    "masako ayam",
    "magic lezat",
    "sasa",
    "blok knorr",
    "sejimpit vetsin",

    # generic / noise ingredient label
    "bumbu",
    "bumbu bubuk",
    "bumbu alusin",
    "bumbu kasar",
    "bumbu tumis",
    "bumbu celupan",
    "bumbu pelengkap",
    "bumbu marinade",
    "bumbu marinase",
    "bumbu kuah",
    "bumbu kuah tongseng",

    "bahan",
    "bahan pelengkap",
    "bahan utama",
    "pelengkap",
    "di haluskan",
    "tambahan",
    "lainnya",
    "satu",
}

MIN_MEANINGFUL_USER_INGREDIENTS = 3
MIN_MATCHED_MEANINGFUL_INGREDIENTS = 3
MIN_MATCH_PERCENTAGE = 55
MAX_RECIPE_INGREDIENTS_FOR_FULL_MATCH = 8

COMMON_INGREDIENTS = {canonical_ingredient(ingredient) for ingredient in COMMON_INGREDIENTS}

TITLE_CORE_ALIASES = {
    "ayam": "ayam",
    "sapi": "daging sapi",
    "kambing": "daging kambing",
    "tongkol": "tongkol",
    "tuna": "tuna",
    "ikan": "ikan",
    "udang": "udang",
    "mie": "mie",
    "telur": "telur",
    "telor": "telur",
    "tahu": "tahu",
    "tempe": "tempe",
}
ANCHOR_INGREDIENTS = set(TITLE_CORE_ALIASES.values())


def infer_title_core_ingredients(title: object) -> set[str]:
    normalized_title = normalize_ingredient(title)
    return {
        canonical
        for keyword, canonical in TITLE_CORE_ALIASES.items()
        if keyword in normalized_title
    }


def title_conflicts_with_user_ingredients(title: object, user_ingredients: set[str]) -> bool:
    normalized_title = normalize_ingredient(title)
    no_santan_title = (
        "no santen" in normalized_title
        or "no santan" in normalized_title
        or "tanpa santan" in normalized_title
        or "tanpa santen" in normalized_title
    )
    return no_santan_title and "santan" in user_ingredients


class LexicalMatcherModel:
    def __init__(self, weights_dir: str | Path = WEIGHTS_DIR):
        self.weights_dir = Path(weights_dir)
        self.vectorizer: Any | None = None
        self.recipe_matrix: sparse.csr_matrix | None = None
        self.recipe_metadata: list[dict[str, Any]] = []
        self.load_error: str | None = None
        self._load_artifacts()

    @property
    def is_ready(self) -> bool:
        return (
            self.vectorizer is not None
            and self.recipe_matrix is not None
            and bool(self.recipe_metadata)
            and hasattr(self, "known_ingredients")
            and hasattr(self, "known_canonical_ingredients")
        )

    def _load_artifacts(self) -> None:
        try:
            self.vectorizer = joblib.load(self.weights_dir / "tfidf_vectorizer.pkl")
            self.recipe_matrix = joblib.load(self.weights_dir / "recipe_matrix.pkl")
            self.recipe_metadata = joblib.load(self.weights_dir / "recipe_metadata.pkl")
            self.ingredient_master = joblib.load(self.weights_dir / "ingredient_master.pkl")
            self.known_ingredients = set(self.ingredient_master.keys())
            self.known_canonical_ingredients = {
                canonical_ingredient(ingredient)
                for ingredient in self.known_ingredients
                if canonical_ingredient(ingredient)
            }
            LOGGER.info("TF-IDF matcher siap: %s resep", self.recipe_matrix.shape[0])
        except Exception as exc:
            self.load_error = f"Gagal memuat artifacts: {exc}"
            LOGGER.exception(self.load_error)

    def predict_match(self, user_ingredients: list[Any], top_k: int = 10, include_details: bool = False) -> list[dict[str, Any]]:
        if not self.is_ready:
            return []

        valid_ingredients = []
        query_terms = []
        for item in user_ingredients:
            if not item.get("is_valid"):
                continue

            norm_name = normalize_ingredient(item.get("name"))
            if not norm_name:
                continue

            canonical_name = canonical_ingredient(norm_name)
            if (
                norm_name not in self.known_ingredients
                and canonical_name not in self.known_canonical_ingredients
            ):
                continue

            valid_ingredients.append(canonical_name)
            query_terms.extend(ingredient_variants(norm_name))
        valid_ingredients = list(dict.fromkeys(valid_ingredients))
        query_terms = list(dict.fromkeys(query_terms))

        if not valid_ingredients:
            return []

        meaningful_user_ingredients = {
            ing for ing in valid_ingredients
            if ing not in COMMON_INGREDIENTS
        }

        if len(meaningful_user_ingredients) < MIN_MEANINGFUL_USER_INGREDIENTS:
            return []
        requested_anchor_ingredients = meaningful_user_ingredients & ANCHOR_INGREDIENTS

        query = " ".join(ingredient_to_token(ing) for ing in query_terms)
        user_vector = self.vectorizer.transform([query])

        scores = linear_kernel(user_vector, self.recipe_matrix).ravel().astype(np.float32)
        
        # Ambil indeks yang punya skor > 0
        candidate_indices = np.flatnonzero(scores > 0.01)
        if candidate_indices.size == 0:
            return []

        candidate_pool_size = max(top_k * 8, 80)

        if candidate_indices.size > candidate_pool_size:
            top_positions = np.argpartition(
                scores[candidate_indices],
                -candidate_pool_size,
            )[-candidate_pool_size:]
            candidate_indices = candidate_indices[top_positions]

        candidate_indices = sorted(
            candidate_indices,
            key=lambda idx: scores[idx],
            reverse=True,
        )

        results = []
        for index in candidate_indices:
            metadata = self.recipe_metadata[int(index)]
            if title_conflicts_with_user_ingredients(
                metadata.get("title", ""),
                meaningful_user_ingredients,
            ):
                continue

            ingredient_source = metadata.get("canonical_ingredients") or metadata.get("ingredients", [])
            recipe_ingredients = {
                canonical_ingredient(ing)
                for ing in ingredient_source
            }

            meaningful_recipe_ingredients = {
                ing for ing in recipe_ingredients
                if ing not in COMMON_INGREDIENTS
            }

            if not meaningful_recipe_ingredients:
                continue

            matched_ingredients = meaningful_user_ingredients & meaningful_recipe_ingredients

            core_ingredients = {
                canonical_ingredient(ing)
                for ing in (
                    metadata.get("core_canonical_ingredients")
                    or metadata.get("core_ingredients", [])
                )
            }

            meaningful_core_ingredients = {
                ing for ing in core_ingredients
                if ing not in COMMON_INGREDIENTS
            }
            title_core_ingredients = infer_title_core_ingredients(metadata.get("title", ""))
            if title_core_ingredients and not (meaningful_user_ingredients & title_core_ingredients):
                continue

            meaningful_core_ingredients |= {
                ing for ing in title_core_ingredients
                if ing not in COMMON_INGREDIENTS
            }

            core_matched = bool(meaningful_user_ingredients & meaningful_core_ingredients)
            if meaningful_core_ingredients and not core_matched:
                continue
            
            min_matched = min(
                MIN_MATCHED_MEANINGFUL_INGREDIENTS,
                len(meaningful_user_ingredients),
            )
            if not meaningful_core_ingredients:
                min_matched = max(min_matched, min(5, len(meaningful_user_ingredients)))

            if len(matched_ingredients) < min_matched:
                continue

            recipe_anchor_ingredients = (
                meaningful_recipe_ingredients & ANCHOR_INGREDIENTS
            ) | meaningful_core_ingredients
            anchor_match_ratio = 1.0
            if requested_anchor_ingredients:
                anchor_match_ratio = len(
                    requested_anchor_ingredients & recipe_anchor_ingredients
                ) / len(requested_anchor_ingredients)
                if anchor_match_ratio <= 0:
                    continue
            
            full_match_target = min(
                len(meaningful_recipe_ingredients),
                MAX_RECIPE_INGREDIENTS_FOR_FULL_MATCH,
            )
            recipe_coverage = min(len(matched_ingredients) / full_match_target, 1.0)
            user_precision = len(matched_ingredients) / len(meaningful_user_ingredients)
            core_bonus = 0.05 if core_matched else 0.0
            required_percentage = MIN_MATCH_PERCENTAGE if meaningful_core_ingredients else 65

            score = min((recipe_coverage * 0.75) + (user_precision * 0.20) + core_bonus, 1.0)
            score *= 0.55 + (0.45 * anchor_match_ratio)
            match_percentage = int(round(score * 100))

            if match_percentage < required_percentage:
                continue
            
        

            results.append({
                "recipe_id": metadata["id"],
                "title": metadata["title"],
                "match_percentage": match_percentage,
                "_matched_count": len(matched_ingredients),
                "_tfidf_score": float(scores[index]),
            })

        results.sort(
            key=lambda item: (
                item["match_percentage"],
                item["_matched_count"],
                item["_tfidf_score"],
            ),
            reverse=True,
        )

        for item in results:
            item.pop("_matched_count", None)
            item.pop("_tfidf_score", None)

        return results[:top_k]



class SmartSubstitutionEngine:
    def __init__(self, weights_dir: str | Path = WEIGHTS_DIR):
        self.weights_dir = Path(weights_dir)
        self.ingredient_kb: dict[str, dict[str, Any]] = {}
        self.load_error: str | None = None
        self._load_master_data()

    @property
    def is_ready(self) -> bool:
        return bool(self.ingredient_kb)

    def _load_master_data(self) -> None:
        try:
            # Load dari artifact yang dibuat oleh train_tfidf.py
            self.ingredient_kb = joblib.load(self.weights_dir / "ingredient_master.pkl")
            LOGGER.info("Knowledge base substitusi siap: %s bahan", len(self.ingredient_kb))
        except Exception as exc:
            self.load_error = f"Gagal memuat Master DB. Jalankan train_tfidf.py dulu: {exc}"
            LOGGER.error(self.load_error)

    def find_best_substitutes(self, missing_ingredient: str, surplus_ingredients: list[str], top_k: int = 5) -> list[dict[str, Any]]:
        if not self.is_ready:
            return []

        norm_missing = normalize_ingredient(missing_ingredient)
        canonical_missing = canonical_ingredient(norm_missing)
        missing_data = self.ingredient_kb.get(norm_missing) or self.ingredient_kb.get(canonical_missing)

        if not missing_data:
            return [] # Jika bahan hilang tidak ada di DB, skip.

        missing_group = missing_data["food_group"]
        missing_vector = missing_data["flavor_vector"]
        missing_substitution_group = substitution_group(norm_missing)
        candidates = []

        for surplus in surplus_ingredients:
            norm_surplus = normalize_ingredient(surplus)
            canonical_surplus = canonical_ingredient(norm_surplus)
            surplus_data = self.ingredient_kb.get(norm_surplus) or self.ingredient_kb.get(canonical_surplus)
            
            if not surplus_data or canonical_surplus == canonical_missing:
                continue

            surplus_group = surplus_data["food_group"]
            surplus_substitution_group = substitution_group(norm_surplus)

            # LAYER 3: Cek Rasa
            surplus_vector = surplus_data["flavor_vector"]
            flavor_score = 0.0
            if np.any(missing_vector) and np.any(surplus_vector):
                flavor_score = float(cosine_similarity(missing_vector, surplus_vector)[0][0])

            if missing_substitution_group != UNKNOWN_SUBSTITUTION_GROUP:
                compatible_groups = SUBSTITUTION_COMPATIBILITY.get(
                    missing_substitution_group,
                    {missing_substitution_group},
                )
                if surplus_substitution_group not in compatible_groups:
                    continue
                functional_score = 1.0 if surplus_substitution_group == missing_substitution_group else 0.75
            else:
                if (
                    missing_group == UNKNOWN_GROUP
                    or surplus_group == UNKNOWN_GROUP
                    or missing_group != surplus_group
                    or missing_group in FALLBACK_FOOD_GROUP_BLOCKLIST
                ):
                    continue
                functional_score = 0.65

            substitution_score = (flavor_score * 0.65) + (functional_score * 0.35)
            if (
                substitution_score < MIN_SUBSTITUTION_SCORE
                or flavor_score < MIN_SUBSTITUTION_FLAVOR_SCORE
            ):
                continue

            candidates.append({
                "substitute_name": surplus,
                "food_group": surplus_group,
                "substitution_group": surplus_substitution_group,
                "flavor_similarity_score": round(flavor_score, 4),
                "substitution_score": round(substitution_score, 4),
            })

        candidates.sort(
            key=lambda x: (x["substitution_score"], x["flavor_similarity_score"]),
            reverse=True,
        )
        return candidates[:top_k]
