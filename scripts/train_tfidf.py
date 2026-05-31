from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml_engine.ingredient_normalization import (
    canonical_ingredient,
    ingredient_to_token,
    ingredient_variants,
    normalize_ingredient,
)

DATA_PATH = BASE_DIR / "data" / "dataset_opsi_1_final_with_images.json"
WEIGHTS_DIR = BASE_DIR / "ml_engine" / "weights"

LOGGER = logging.getLogger(__name__)
SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s-]", flags=re.UNICODE)


def slugify(text: str) -> str:
    """Mengubah judul menjadi URL slug (contoh: Pepes Tahu -> pepes-tahu)"""
    text = PUNCT_RE.sub("", str(text).lower())
    return SPACE_RE.sub("-", text.strip())

def build_vectorizer(min_df: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\w[\w-]*\b",
        min_df=min_df,
        norm="l2",
        smooth_idf=True,
        sublinear_tf=True,
        dtype=np.float32,
    )

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    LOGGER.info("Memuat dataset dari %s", DATA_PATH)
    if not DATA_PATH.exists():
        LOGGER.error("Dataset tidak ditemukan!")
        return

    with DATA_PATH.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    # 1. Ekstrak Master Ingredients
    LOGGER.info("Mengekstrak Master Ingredients...")
    master_ingredients = {}
    for item in dataset.get("data_ingredients_master", []):
        norm_name = normalize_ingredient(item.get("clean_name"))
        vector = item.get("flavor_vector")
        master_ingredients[norm_name] = {
            "ingredient_id": item.get("ingredient_id"),
            "food_group": item.get("food_group", "Unknown"),
            "flavor_vector": np.asarray(vector, dtype=np.float32).reshape(1, -1) if vector else np.zeros((1, 7), dtype=np.float32)
        }
    
    # 2. Ekstrak Resep & Boosting is_core
    LOGGER.info("Memproses Data Resep Mapped...")
    metadata = []
    corpus = []
    
    # Buat dictionary cepat untuk lookup ID ke Nama Clean
    id_to_clean_name = {v.get("ingredient_id"): k for k, v in master_ingredients.items()}

    for index, recipe in enumerate(dataset.get("data_recipes_mapped", []), start=1):
        recipe_id = slugify(recipe.get("title", f"recipe-{index}"))
        
        doc_tokens = []
        clean_ingredients_list = []
        canonical_ingredients_list = []
        core_ingredients_list = []
        core_canonical_ingredients_list = []
        
        for ing in recipe.get("ingredients", []):
            ing_id = ing.get("ingredient_id")
            clean_name = id_to_clean_name.get(ing_id)
            
            if clean_name:
                canonical_name = canonical_ingredient(clean_name)
                tokens = [ingredient_to_token(term) for term in ingredient_variants(clean_name)]

                # BOOSTING is_core: Jika true, masukkan token 3x lipat ke dokumen
                if ing.get("is_core") is True:
                    for token in tokens:
                        doc_tokens.extend([token, token, token])
                    core_ingredients_list.append(clean_name)
                    if canonical_name not in core_canonical_ingredients_list:
                        core_canonical_ingredients_list.append(canonical_name)
                else:
                    doc_tokens.extend(tokens)
                
                clean_ingredients_list.append(clean_name)
                if canonical_name not in canonical_ingredients_list:
                    canonical_ingredients_list.append(canonical_name)

        if doc_tokens:
            corpus.append(" ".join(doc_tokens))
            metadata.append({
                "id": recipe_id,
                "title": recipe.get("title"),
                "ingredients": clean_ingredients_list,
                "canonical_ingredients": canonical_ingredients_list,
                "core_ingredients": core_ingredients_list,
                "core_canonical_ingredients": core_canonical_ingredients_list,
            })

    # 3. Training TF-IDF
    LOGGER.info("Training TF-IDF pada %s resep...", len(corpus))
    vectorizer = build_vectorizer(min_df=1)
    matrix = vectorizer.fit_transform(corpus).tocsr().astype(np.float32)
    matrix.sort_indices()

    # 4. Export Artifacts
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, WEIGHTS_DIR / "tfidf_vectorizer.pkl", compress=3)
    joblib.dump(matrix, WEIGHTS_DIR / "recipe_matrix.pkl", compress=3)
    joblib.dump(metadata, WEIGHTS_DIR / "recipe_metadata.pkl", compress=3)
    # Artifact Baru: Master DB untuk Substitusi Engine Layer 2 & 3
    joblib.dump(master_ingredients, WEIGHTS_DIR / "ingredient_master.pkl", compress=3) 
    
    LOGGER.info("SUKSES! Artifacts tersimpan di %s", WEIGHTS_DIR)

if __name__ == "__main__":
    main()
