from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks

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
    SubstitutionRequest,
    TaskStatusResponse,
    TaskStatusData,
    SubstitutionData,
    SubstitutionMapping,
    SubItem,
)
from ml_engine.information_retrieval import LexicalMatcherModel, SmartSubstitutionEngine

router = APIRouter(tags=["Internal AI System"])
AUTH_DEPENDENCY = [Depends(verify_internal_api_key)]
dl_ranker = None

# In-Memory Cache bertindak sebagai simulasi Redis lokal RAM
AI_TASK_CACHE: dict[str, dict[str, Any]] = {}

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
            message="Belum ada resep yang cukup cocok.",
            data=[],
        )

    # FIXED: Sesuai dengan spesifikasi YAML terbaru dari Miika (Hanya mengembalikan ID dan Persentase)
    results = [
        MatchResult(
            recipe_id=prediction["recipe_id"],
            match_percentage=prediction["match_percentage"],
        )
        for prediction in predictions
    ]

    return MatchResponse(
        message="Matching successful",
        data=results,
    )


def _run_async_dl_substitution(
    task_id: str,
    missing_items: list[ItemData],
    surplus_items: list[ItemData],
    substitution_engine: SmartSubstitutionEngine
):
    """
    Fungsi Latar Belakang (Worker Thread) murni untuk memproses kalkulasi model Deep Learning.
    Fungsi ini tidak mengembalikan HTTP Response, melainkan menulis status hasil langsung ke cache.
    """
    global dl_ranker
    try:
        # FIXED: Menghapus pengecekan 'payload' gaib karena data sudah dikirim matang dari parameter fungsi[cite: 1]
        if not surplus_items:
            AI_TASK_CACHE[task_id] = {
                "status": "completed",
                "result": SubstitutionData(
                    character=CharacterDialog(
                        status="fail",
                        dialog="Aku belum bisa mencari pengganti karena tidak ada bahan surplus yang valid.",
                    ),
                    substitutions_mapping=None,
                ),
            }
            return

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
            
            flavor_score = best_candidate.get("flavor_similarity_score", 0.0)
            functional_score = 1.0 if best_candidate.get("same_food_group") else 0.0
                
            if dl_ranker is None:
                dl_ranker = SubstitutionRankerModel(engine_instance=substitution_engine)
                    
            dl_score = dl_ranker.predict_rank_score_context(
                candidate_item=replacement_name, 
                all_recipe_ingredients=surplus_names
            )
                
            combined_score = (dl_score * 0.50) + (flavor_score * 0.30) + (functional_score * 0.20)
            flavor_score = round(combined_score, 4)
                
            replacement_item = surplus_by_name.get(replacement_name.casefold())
            
            mappings.append(
            SubstitutionMapping(
                missing_item=_sub_item(missing_item, missing_item.name),
                replaced_with=_sub_item(replacement_item, replacement_name),
                flavor_score=flavor_score # <--- Masukin ke sini biar nilainya tumpah ke JSON!
            )
)

        if not mappings:
            AI_TASK_CACHE[task_id] = {
                "status": "completed",
                "result": SubstitutionData(
                    character=CharacterDialog(
                        status="fail",
                        dialog="Aku belum menemukan pengganti yang realistis dari bahan surplusmu.",
                    ),
                    substitutions_mapping=None,
                ),
            }
            return

        mapping_text = ", ".join(
            f"{mapping.missing_item.name} -> {mapping.replaced_with.name}"
            for mapping in mappings
        )
        status_value = "mid_success" if unresolved else "fully_success"
        dialog = f"Aku menemukan pengganti yang paling cocok: {mapping_text}."

        if unresolved:
            dialog += f" Belum ada pengganti realistis untuk: {', '.join(unresolved)}."

        # FIXED: Status cache di-update dengan aman di dalam scope try block sebelum ditutup[cite: 1]
        AI_TASK_CACHE[task_id] = {
            "status": "completed",
            "result": SubstitutionData(
                character=CharacterDialog(
                    status=status_value,
                    dialog=dialog,
                ),
                substitutions_mapping=mappings,
            ),
        }

    except Exception:
        # Jika algoritma tensor TensorFlow crash, tandai status sebagai FAILED[cite: 1]
        AI_TASK_CACHE[task_id] = {"status": "failed", "result": None}


@router.post("/ai-substitution", dependencies=AUTH_DEPENDENCY)
def ai_substitution(
    payload: SubstitutionRequest,
    background_tasks: BackgroundTasks,
    substitution_engine: Annotated[SmartSubstitutionEngine, Depends(get_substitution_engine)],
):
    missing_items = _valid_items(payload.missing_ingredients)
    surplus_items = _valid_items(payload.surplus_ingredients)

    if not missing_items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Missing ingredients kosong."
        )

    # FIXED: Menandai status awal pengerjaan tugas di cache RAM sebagai PROCESSING[cite: 1]
    AI_TASK_CACHE[payload.task_id] = {"status": "processing", "result": None}

    # Melempar fungsi kalkulasi berat ke background thread agar endpoint langsung merespons instant 200 (Anti-RTO)[cite: 1]
    background_tasks.add_task(
        _run_async_dl_substitution,
        payload.task_id, 
        missing_items, 
        surplus_items, 
        substitution_engine
    )

    return {"message": "AI substitution process initiated successfully"}


@router.get("/ai-result/{task_id}", response_model=TaskStatusResponse, dependencies=AUTH_DEPENDENCY)
def get_ai_result(task_id: str) -> TaskStatusResponse:
    """
    Endpoint baru penunjang polling asinkronus bagi backend utama (Express.js)[cite: 1].
    """
    task = AI_TASK_CACHE.get(task_id)
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Tugas AI tidak ditemukan atau sudah kedaluwarsa."
        )
        
    return TaskStatusResponse(
        message="Status tugas AI berhasil dimuat",
        data=TaskStatusData(
            status=task["status"], 
            result=task["result"]
        )
    )