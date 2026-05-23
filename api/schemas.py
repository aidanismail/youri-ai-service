from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)


class ItemData(APIModel):
    id: str | None = None
    name: str = Field(..., min_length=1)
    is_valid: bool = True


class TimeScope(APIModel):
    time: int = Field(..., ge=1)
    scope: str = Field(..., min_length=1)


class MatchRequest(APIModel):
    recipe_name: str | None = None
    recipe_id: str | None = None
    ingredients: list[ItemData] = Field(..., min_length=1)
    tools: list[ItemData] = Field(default_factory=list)
    time: TimeScope | None = None
    exclude_ingredients: list[dict[str, Any]] | None = None


class MatchResult(APIModel):
    recipe_id: str
    match_percentage: int = Field(..., ge=0, le=100)


class MatchResponse(APIModel):
    message: str
    data: list[MatchResult]


class SubstitutionRequest(APIModel):
    recipe_id: str = Field(..., min_length=1)
    missing_ingredients: list[ItemData] = Field(..., min_length=1)
    surplus_ingredients: list[ItemData] = Field(default_factory=list)


class SubItem(APIModel):
    id: str
    name: str


class SubstitutionMapping(APIModel):
    missing_item: SubItem
    replaced_with: SubItem
    flavor_score: float | None = None


class CharacterDialog(APIModel):
    status: Literal["fully_success", "mid_success", "fail"]
    dialog: str = Field(..., min_length=1)


class SubstitutionData(APIModel):
    character: CharacterDialog
    substitutions_mapping: list[SubstitutionMapping] | None = None


class SubstitutionResponse(APIModel):
    message: str
    data: SubstitutionData
