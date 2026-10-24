"""
Shared data contracts (BRIEF.md section 4). All three roles code against these.

Do not change a field here without telling the whole team first.

    python -m schema.validate data/mock
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

TASTE_KEYS = ("spice", "sweet", "sour", "bitter", "umami", "richness", "crunch", "creaminess")

# UK FSA's 14 regulated allergens. "nuts" means tree nuts; peanuts are separate.
ALLERGENS = (
    "celery", "gluten", "crustaceans", "egg", "fish", "lupin", "milk",
    "molluscs", "mustard", "nuts", "peanuts", "sesame", "soya", "sulphites",
)

DIETARY_KEYS = ("vegan", "vegetarian", "halal", "kosher")

Score = Annotated[float, Field(ge=0, le=10)]
YesNoUnknown = Literal["yes", "no", "unknown"]
Cuisine = Literal["indian", "italian"]
VibeTag = Literal["casual", "date-night", "trendy", "traditional", "quick", "family"]
Allergen = Literal[ALLERGENS]
DietaryRestriction = Literal[DIETARY_KEYS]


class TasteVector(BaseModel):
    spice: Score = 5
    sweet: Score = 5
    sour: Score = 5
    bitter: Score = 5
    umami: Score = 5
    richness: Score = 5
    crunch: Score = 5
    creaminess: Score = 5


class TasteConfidence(BaseModel):
    spice: float = Field(0, ge=0, le=1)
    sweet: float = Field(0, ge=0, le=1)
    sour: float = Field(0, ge=0, le=1)
    bitter: float = Field(0, ge=0, le=1)
    umami: float = Field(0, ge=0, le=1)
    richness: float = Field(0, ge=0, le=1)
    crunch: float = Field(0, ge=0, le=1)
    creaminess: float = Field(0, ge=0, le=1)


class Dietary(BaseModel):
    """"unknown" is never treated as safe by the recommender."""
    vegan: YesNoUnknown = "unknown"
    vegetarian: YesNoUnknown = "unknown"
    halal: YesNoUnknown = "unknown"
    kosher: YesNoUnknown = "unknown"


# ----- 4.2 / 4.3 catalogue (data role owns) -----

class Dish(BaseModel):
    id: str = Field(pattern=r"^d_\d{4}$")
    restaurant_id: str = Field(pattern=r"^r_\d{2}$")
    name: str
    description: str = ""
    cuisine: Cuisine
    category: Literal["starter", "main", "side", "dessert", "drink"]
    dish_family: str
    price_gbp: float = Field(ge=0)
    ingredients: list[str]
    allergens: list[Allergen]
    dietary: Dietary
    taste: TasteVector
    serving_temp: Literal["hot", "warm", "cold"]
    adventurousness: Score
    tagged_by: str  # "claude" | "nebula:<model>" | "human"
    verified: bool = False


class Restaurant(BaseModel):
    id: str = Field(pattern=r"^r_\d{2}$")
    name: str
    cuisine: Cuisine
    price_tier: int = Field(ge=1, le=4)  # 1 cheap/casual .. 4 premium
    vibe_tags: list[VibeTag]
    area: str
    lat: float = 0.0
    lng: float = 0.0
    halal_status: Literal["certified", "some", "none", "unknown"]
    menu_source_url: str
    dish_ids: list[str] = []


# ----- 4.4 user profile (recommender role owns) -----

class DishRating(BaseModel):
    dish_id: str
    stars: int = Field(ge=1, le=5)


class VenueChoice(BaseModel):
    round_id: str
    options: list[str]  # restaurant ids
    chosen: str


class UserProfile(BaseModel):
    id: str
    first_name: str = ""
    last_name: str = ""
    dietary: list[DietaryRestriction] = []
    allergies: list[Allergen] = []
    disliked_ingredients: list[str] = []
    taste_vector: TasteVector = TasteVector()
    taste_confidence: TasteConfidence = TasteConfidence()
    dish_ratings: list[DishRating] = []
    venue_choices: list[VenueChoice] = []
    price_pref: float = Field(2.5, ge=1, le=4)
    vibe_pref: dict[VibeTag, float] = {}
    adventurousness: Score = 5
    created_at: str = ""  # ISO8601


# ----- 4.5 API bodies and responses -----

class ProfileCreate(BaseModel):
    first_name: str
    last_name: str = ""
    dietary: list[DietaryRestriction] = []
    allergies: list[Allergen] = []
    disliked_ingredients: list[str] = []


class RatingsIn(BaseModel):
    ratings: list[DishRating]


class VenueOption(BaseModel):
    restaurant: Restaurant
    dish: Dish


class VenueRound(BaseModel):
    round_id: str
    dish_family: str
    options: list[VenueOption]


class TasteMap(BaseModel):
    taste_vector: TasteVector
    taste_confidence: TasteConfidence
    adventurousness: float
    price_pref: float
    vibe_pref: dict[str, float]
    top_dimensions: list[str]  # strongest dimensions first, for the profile headline


class RecommendRequest(BaseModel):
    profile_id: str
    occasion: str
    party_size: int | None = Field(None, ge=1)
    budget: int | None = Field(None, ge=1, le=4)  # max price_tier


class Recommendation(BaseModel):
    restaurant: Restaurant
    dishes: list[Dish]
    score: float
    explanation: str
    warnings: list[str]
