from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

matplotlib.use("Agg")

import seaborn as sns
from matplotlib import pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "final_recipes_mongodb_v_final.json"
WEIGHTS_DIR = BASE_DIR / "ml_engine" / "weights"
EDA_PLOT_PATH = BASE_DIR / "logs" / "eda_top_ingredients.png"

LOGGER = logging.getLogger(__name__)
SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s-]", flags=re.UNICODE)
RecipeMetadata = dict[str, object]


def normalize_ingredient(value: object) -> str:
    text = PUNCT_RE.sub(" ", str(value or "").casefold())
    return SPACE_RE.sub(" ", text).strip(" -_")


def ingredient_to_token(ingredient: str) -> str:
    return ingredient.replace(" ", "_")


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []

    for value in values:
        if value and value not in seen:
            seen.add(value)
            unique_values.append(value)

    return unique_values


def recipe_id_from_url(url: str, index: int) -> str:
    slug = urlparse(url or "").path.rstrip("/").split("/")[-1]
    return slug or f"recipe-{index:05d}"


def load_and_extract_data(filepath: Path) -> tuple[list[RecipeMetadata], list[str], list[str]]:
    LOGGER.info("Memuat dataset dari %s", filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {filepath}")

    with filepath.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, dict):
        raise ValueError("Dataset JSON harus berupa object.")

    recipes = dataset.get("data")
    if not isinstance(recipes, list):
        raise ValueError("Dataset JSON harus memiliki key 'data' berisi list resep.")

    metadata: list[RecipeMetadata] = []
    corpus: list[str] = []
    ingredient_occurrences: list[str] = []
    seen_recipe_ids: set[str] = set()
    skipped = 0

    for index, recipe in enumerate(recipes, start=1):
        if not isinstance(recipe, dict):
            skipped += 1
            continue

        mapped_ingredients = recipe.get("mapped_ingredients") or []
        if not isinstance(mapped_ingredients, list):
            skipped += 1
            continue

        ingredients = unique_preserve_order(
            [
                normalize_ingredient(item.get("cleaned_entity"))
                for item in mapped_ingredients
                if isinstance(item, dict)
            ]
        )

        if not ingredients:
            skipped += 1
            continue

        url = str(recipe.get("URL") or "").strip()
        recipe_id = recipe_id_from_url(url, index)
        if recipe_id in seen_recipe_ids:
            recipe_id = f"{recipe_id}-{index:05d}"
        seen_recipe_ids.add(recipe_id)

        tokens = [ingredient_to_token(ingredient) for ingredient in ingredients]
        title = str(recipe.get("Title") or f"Recipe {index}").strip() or f"Recipe {index}"

        metadata.append(
            {
                "id": recipe_id,
                "title": title,
                "url": url,
                "ingredients": ingredients,
                "ingredient_tokens": tokens,
                "ingredient_count": len(ingredients),
            }
        )
        corpus.append(" ".join(tokens))
        ingredient_occurrences.extend(ingredients)

    if not corpus:
        raise ValueError("Tidak ada resep valid dengan mapped_ingredients.")

    LOGGER.info("Resep valid: %s | Dilewati: %s", len(corpus), skipped)
    return metadata, corpus, ingredient_occurrences


def perform_eda(ingredients: list[str], output_path: Path, top_n: int = 20) -> None:
    if not ingredients:
        LOGGER.warning("EDA dilewati karena tidak ada ingredient valid.")
        return

    counts = Counter(ingredients)
    top_items = counts.most_common(top_n)
    top_df = pd.DataFrame(top_items, columns=["ingredient", "frequency"])

    LOGGER.info("Total ingredient unik: %s", len(counts))
    LOGGER.info("Top 5 ingredient:\n%s", top_df.head().to_string(index=False))

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_df, x="frequency", y="ingredient", color="#2f7f73")
        plt.title(f"Top {top_n} Ingredient Paling Sering Muncul")
        plt.xlabel("Frekuensi Resep")
        plt.ylabel("Ingredient")
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        LOGGER.info("Grafik EDA tersimpan di %s", output_path)
    except OSError as exc:
        LOGGER.warning("Grafik EDA tidak tersimpan: %s", exc)
    finally:
        plt.close()


def build_vectorizer(min_df: int, max_df: float, max_features: int | None) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\w[\w-]*\b",
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        norm="l2",
        smooth_idf=True,
        sublinear_tf=True,
        dtype=np.float32,
    )


def train_and_export_tfidf(
    corpus: list[str],
    metadata: list[RecipeMetadata],
    weights_dir: Path,
    min_df: int,
    max_df: float,
    max_features: int | None,
) -> None:
    LOGGER.info("Training TF-IDF pada %s resep", len(corpus))

    vectorizer = build_vectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
    )
    matrix = vectorizer.fit_transform(corpus).tocsr().astype(np.float32)
    matrix.sort_indices()

    model_info = {
        "n_recipes": matrix.shape[0],
        "n_features": matrix.shape[1],
        "min_df": min_df,
        "max_df": max_df,
        "max_features": max_features,
        "artifact_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_unique_ingredients": len(vectorizer.vocabulary_),
    }

    weights_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, weights_dir / "tfidf_vectorizer.pkl", compress=3)
    joblib.dump(matrix, weights_dir / "recipe_matrix.pkl", compress=3)
    joblib.dump(metadata, weights_dir / "recipe_metadata.pkl", compress=3)
    joblib.dump(model_info, weights_dir / "tfidf_model_info.pkl", compress=3)

    LOGGER.info("Matriks TF-IDF: %s", matrix.shape)
    LOGGER.info("Artifacts tersimpan di %s", weights_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TF-IDF recipe matcher.")
    parser.add_argument("--data-path", type=Path, default=DATA_PATH)
    parser.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--skip-eda", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.min_df < 1:
        raise ValueError("--min-df harus >= 1.")

    if not 0 < args.max_df <= 1:
        raise ValueError("--max-df harus di antara 0 dan 1.")

    if args.max_features is not None and args.max_features < 1:
        raise ValueError("--max-features harus >= 1 jika diisi.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()
    validate_args(args)

    metadata, corpus, ingredients = load_and_extract_data(args.data_path)

    if not args.skip_eda:
        perform_eda(ingredients, EDA_PLOT_PATH)

    train_and_export_tfidf(
        corpus=corpus,
        metadata=metadata,
        weights_dir=args.weights_dir,
        min_df=args.min_df,
        max_df=args.max_df,
        max_features=args.max_features,
    )


if __name__ == "__main__":
    main()
