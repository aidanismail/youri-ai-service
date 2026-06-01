from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import (
    get_lexical_matcher,
    get_substitution_engine,
    verify_internal_api_key,
)

from ml_engine.substitution_dl import SubstitutionRankerModel

from api.schemas import (
    CharacterDialog,
    ItemData,
    MatchRequest,
    MatchResponse,
    MatchResult,
    SubItem,
    SubstitutionData,
    SubstitutionMapping,
    SubstitutionRequest,
    SubstitutionResponse,
)
from ml_engine.information_retrieval import LexicalMatcherModel, SmartSubstitutionEngine

router = APIRouter(tags=["Internal AI System"])
AUTH_DEPENDENCY = [Depends(verify_internal_api_key)]
dl_ranker = None

def _valid_items(items: list[ItemData]) -> list[ItemData]:
    return [item for item in items if item.is_valid and item.name]


def _sub_item(item: ItemData | None, name: str) -> SubItem:
    return SubItem(id=item.id if item and item.id else "unknown", name=name)


@router.post("/match", response_model=MatchResponse, dependencies=AUTH_DEPENDENCY)
def match_recipes(
    payload: MatchRequest,
    lexical_matcher: Annotated[LexicalMatcherModel, Depends(get_lexical_matcher)],
) -> MatchResponse:
    valid_ingredients = _valid_items(payload.ingredients)
    if not valid_ingredients:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Ingredients valid tidak boleh kosong.",
        )

    predictions = lexical_matcher.predict_match(
        user_ingredients=[ingredient.model_dump() for ingredient in valid_ingredients],
        top_k=10,
        include_details=False,
    )

    if not predictions:
        return MatchResponse(
            message="Belum ada resep yang cukup cocok. Coba tambahkan bahan utama atau bumbu yang lebih spesifik.",
            data=[],
        )

    results = [
        MatchResult(
            recipe_id=prediction["recipe_id"],
            title=prediction.get("title", "Resep Tanpa Nama"),
            match_percentage=prediction["match_percentage"],
            steps=prediction.get("steps", []),
        )
        for prediction in predictions
    ]

    return MatchResponse(
        message="Matching successful",
        data=results,
    )


@router.post("/ai-substitution", response_model=SubstitutionResponse, dependencies=AUTH_DEPENDENCY)
def ai_substitution(
    payload: SubstitutionRequest,
    substitution_engine: Annotated[SmartSubstitutionEngine, Depends(get_substitution_engine)],
) -> SubstitutionResponse:
    missing_items = _valid_items(payload.missing_ingredients)
    surplus_items = _valid_items(payload.surplus_ingredients)

    if not missing_items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing ingredients valid tidak boleh kosong.",
        )

    if not surplus_items:
        return SubstitutionResponse(
            message="AI substitution completed",
            data=SubstitutionData(
                character=CharacterDialog(
                    status="fail",
                    dialog="Aku belum bisa mencari pengganti karena tidak ada bahan surplus yang valid.",
                ),
                substitutions_mapping=None,
            ),
        )

    surplus_names = [item.name for item in surplus_items]
    surplus_by_name = {item.name.casefold(): item for item in surplus_items}
    mappings: list[SubstitutionMapping] = []
    unresolved: list[str] = []

    for missing_item in missing_items:
        candidates = substitution_engine.find_best_substitutes(
            missing_ingredient=missing_item.name,
            surplus_ingredients=surplus_names,
            top_k=1,
        )

        if not candidates:
            unresolved.append(missing_item.name)
            continue

        best_candidate = candidates[0]
        replacement_name = best_candidate["substitute_name"]
        
        # 1. TANGKAP SKORNYA DARI KANDIDAT
        flavor_score = best_candidate.get("flavor_similarity_score", 0.0)
        
        functional_score = 1.0 if best_candidate.get("same_food_group") else 0.0
            
        global dl_ranker
        if dl_ranker is None:
            dl_ranker = SubstitutionRankerModel(engine_instance=substitution_engine)
                
        dl_score = dl_ranker.predict_rank_score_context(
            candidate_item=replacement_name, 
            all_recipe_ingredients=surplus_names
        )
            
        # Gabungkan Bobot Penilaian (50% DL + 30% Flavor Cosine + 20% Functional Filter)
        combined_score = (dl_score * 0.50) + (flavor_score * 0.30) + (functional_score * 0.20)
        flavor_score = round(combined_score, 4)  # Menimpa nilai agar masuk ke respons JSON
            
        replacement_item = surplus_by_name.get(replacement_name.casefold())
        
        mappings.append(
            SubstitutionMapping(
                missing_item=_sub_item(missing_item, missing_item.name),
                replaced_with=_sub_item(replacement_item, replacement_name),
                flavor_score=flavor_score # 2. MASUKKAN SKORNYA KE SINI
            )
        )

    if not mappings:
        return SubstitutionResponse(
            message="AI substitution completed",
            data=SubstitutionData(
                character=CharacterDialog(
                    status="fail",
                    dialog="Aku belum menemukan pengganti yang realistis dari bahan surplusmu.",
                ),
                substitutions_mapping=None,
            ),
        )

    mapping_text = ", ".join(
        f"{mapping.missing_item.name} -> {mapping.replaced_with.name}"
        for mapping in mappings
    )
    status_value = "mid_success" if unresolved else "fully_success"
    dialog = f"Aku menemukan pengganti yang paling cocok: {mapping_text}."

    if unresolved:
        dialog += f" Belum ada pengganti realistis untuk: {', '.join(unresolved)}."

    return SubstitutionResponse(
        message="AI substitution successful",
        data=SubstitutionData(
            character=CharacterDialog(
                status=status_value,
                dialog=dialog,
            ),
            substitutions_mapping=mappings,
        ),
    )
