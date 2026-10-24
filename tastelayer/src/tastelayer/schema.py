"""Pydantic v2 models for the TasteLayer data contracts (BRIEF.md section 4).

Taste dimensions, all 0-10, anchors low / mid / high:
  spice       capsaicin heat         black pepper / jalapeno / habanero
  sweet       sweetness              roasted red pepper / caramelised onion / maple syrup
  sour        acidity/tang           natural yoghurt / lemon juice / tamarind
  bitter      bitterness             rocket / 70% dark chocolate / radicchio
  umami       savouriness            tomato / parmesan / anchovy or miso
  richness    fat/heaviness          olive-oil dressing / cream sauce / ghee-heavy dal makhani
  crunch      crispy texture         soft bread / toasted nuts / fried papadum
  creaminess  smooth/creamy texture  clear broth / hummus / mascarpone

Scale convention: 0 = absent, 2-3 = low anchor, 5 = mid anchor, 8-9 = high anchor, 10 = extreme.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

TASTE_DIMS: list[str] = ["spice", "sweet", "sour", "bitter", "umami", "richness", "crunch", "creaminess"]

TASTE_DIMENSIONS = {
    "spice": {"meaning": "capsaicin heat", "anchors": ["black pepper", "jalapeño", "habanero"]},
    "sweet": {"meaning": "sweetness", "anchors": ["roasted red pepper", "caramelised onion", "maple syrup"]},
    "sour": {"meaning": "acidity/tang", "anchors": ["natural yoghurt", "lemon juice", "tamarind"]},
    "bitter": {"meaning": "bitterness", "anchors": ["rocket", "70% dark chocolate", "radicchio"]},
    "umami": {"meaning": "savouriness", "anchors": ["tomato", "parmesan", "anchovy or miso"]},
    "richness": {"meaning": "fat/heaviness", "anchors": ["olive-oil dressing", "cream sauce", "ghee-heavy dal makhani"]},
    "crunch": {"meaning": "crispy/crunchy texture", "anchors": ["soft bread", "toasted nuts", "fried papadum"]},
    "creaminess": {"meaning": "smooth/creamy texture", "anchors": ["clear broth", "hummus", "mascarpone"]},
}
ANCHOR_SCORES = {"low": 2.5, "mid": 5.0, "high": 8.5}

Flag = Literal["yes", "no", "unknown"]
Cuisine = Literal["indian", "italian"]


class TasteVector(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spice: float = Field(ge=0, le=10)
    sweet: float = Field(ge=0, le=10)
    sour: float = Field(ge=0, le=10)
    bitter: float = Field(ge=0, le=10)
    umami: float = Field(ge=0, le=10)
    richness: float = Field(ge=0, le=10)
    crunch: float = Field(ge=0, le=10)
    creaminess: float = Field(ge=0, le=10)

    def to_list(self) -> list[float]:
        return [getattr(self, d) for d in TASTE_DIMS]

    @classmethod
    def from_list(cls, xs) -> "TasteVector":
        return cls(**{d: float(min(10, max(0, x))) for d, x in zip(TASTE_DIMS, xs)})


class Dietary(BaseModel):
    vegan: Flag = "unknown"
    vegetarian: Flag = "unknown"
    halal: Flag = "unknown"
    kosher: Flag = "unknown"


class Dish(BaseModel):
    # Core contract fields
    id: str
    restaurant_id: str
    name: str
    description: str = ""
    cuisine: Cuisine
    category: Literal["starter", "main", "side", "dessert", "drink"] = "main"
    dish_family: str = "other"
    price_gbp: Optional[float] = None
    ingredients: list[str] = []
    allergens: list[str] = []
    dietary: Dietary = Dietary()
    taste: Optional[TasteVector] = None
    serving_temp: Literal["hot", "warm", "cold"] = "hot"
    adventurousness: float = Field(default=0, ge=0, le=10)
    tagged_by: str = "human"  # claude | nebius:<model> | nebius:consensus | trained-regressor | human
    verified: bool = False  # human checked allergens + dietary flags (filters rely on this); models never change it
    # Proposed optional additions (CONTRACT_CHANGES.md)
    taste_confidence: Optional[dict[str, float]] = None
    taste_std: Optional[dict[str, float]] = None
    ingredients_canonical: list[str] = []
    allergens_suggested: list[str] = []  # human review only; filters never read this
    taste_model_version: Optional[str] = None
    taste_verified: bool = False  # human checked the taste vector (gold set)
    is_mock: bool = False

    @field_validator("tagged_by")
    @classmethod
    def _check_tagged_by(cls, v: str) -> str:
        ok = v in {"claude", "trained-regressor", "human"} or v.startswith("nebius:")
        if not ok:
            raise ValueError(f"bad tagged_by: {v}")
        return v


class Restaurant(BaseModel):
    id: str
    name: str
    cuisine: Cuisine
    price_tier: int = Field(ge=1, le=4)
    vibe_tags: list[str] = []
    area: str = ""
    lat: float = 0.0
    lng: float = 0.0
    halal_status: Literal["certified", "some", "none", "unknown"] = "unknown"
    menu_source_url: str = ""
    dish_ids: list[str] = []


class DishRating(BaseModel):
    dish_id: str
    stars: int = Field(ge=1, le=5)


class VenueChoice(BaseModel):
    round_id: str
    options: list[str]
    chosen: str


class UserProfile(BaseModel):
    id: str
    first_name: str = ""
    last_name: str = ""
    dietary: list[str] = []
    allergies: list[str] = []
    disliked_ingredients: list[str] = []
    taste_vector: dict[str, float] = {}
    taste_confidence: dict[str, float] = {}
    dish_ratings: list[DishRating] = []
    venue_choices: list[VenueChoice] = []
    price_pref: float = 2.5
    vibe_pref: dict[str, float] = {}
    adventurousness: float = 5.0
    created_at: str = ""


# ---- LLM tagging I/O ----
class IntTaste(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spice: int
    sweet: int
    sour: int
    bitter: int
    umami: int
    richness: int
    crunch: int
    creaminess: int


class TagItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    uid: str
    taste: IntTaste
    confidence: float


class TagBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[TagItem]


class TagRecord(BaseModel):
    """One model's label for one dish, as stored in labels.parquet."""
    uid: str
    model: str
    taste: TasteVector
    confidence: float = Field(ge=0, le=1)
    rubric_version: str
