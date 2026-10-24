"""
Exclusion-only safety rules over dish text (BRIEF.md 3 and 8.4: hard filters are pure code).

Menus rarely state dietary flags or allergens, and the TasteLayer catalogue has every flag
"unknown". These keyword rules read a dish's name, description and ingredients and can only
make a dish LESS eligible: they turn a dietary flag into "no" or add an implied allergen.
They never mark anything "yes" or safe, so a missed keyword falls back to the data's own
flag ("unknown" -> warning, or exclusion for strict profiles). No LLM output is read here.
"""

from __future__ import annotations

import re
from functools import lru_cache

from schema.models import Dish

MEAT = ("chicken", "lamb", "beef", "pork", "mutton", "goat", "veal", "duck", "turkey", "venison",
        "rabbit", "bacon", "ham", "sausage", "pepperoni", "salami", "chorizo", "prosciutto", "pancetta",
        "guanciale", "nduja", "mortadella", "bresaola", "meatball", "mince", "keema", "steak", "burger",
        "boti", "seekh", "gelatin")
SEAFOOD_FISH = ("fish", "salmon", "tuna", "cod", "haddock", "anchovy", "anchovies", "sardine", "mackerel",
                "sea bass", "seabass", "trout", "swordfish", "squid ink", "bottarga", "pomfret", "tilapia")
SHELLFISH = ("prawn", "shrimp", "crab", "lobster", "langoustine", "scampi", "crayfish")
MOLLUSCS = ("squid", "calamari", "octopus", "mussel", "clam", "scallop", "oyster", "vongole", "cuttlefish")
DAIRY = ("milk", "cream", "butter", "cheese", "paneer", "ghee", "yoghurt", "yogurt", "dahi", "raita", "lassi",
         "mozzarella", "parmesan", "parmigiano", "pecorino", "ricotta", "mascarpone", "burrata", "gorgonzola",
         "provolone", "fontina", "taleggio", "stracciatella", "bechamel", "malai", "makhani", "kulfi", "kheer",
         "gelato", "ice cream", "khoya", "rasmalai", "tiramisu")
EGG = ("egg", "eggs", "mayonnaise", "mayo", "aioli", "carbonara", "tiramisu", "meringue")
HONEY = ("honey",)
PORK = ("pork", "bacon", "ham", "pancetta", "guanciale", "prosciutto", "chorizo", "salami", "pepperoni",
        "nduja", "mortadella", "lard", "speck", "porchetta", "cotechino")
ALCOHOL = ("wine", "beer", "rum", "brandy", "vodka", "marsala", "amaretto", "limoncello", "prosecco",
           "grappa", "sake", "whisky", "whiskey", "liqueur", "vin santo")
NUTS = ("nut", "walnut", "almond", "cashew", "pistachio", "hazelnut", "pecan", "macadamia", "pine nut",
        "praline", "marzipan", "frangipane", "amaretti", "badam", "kaju", "pesto")
PEANUTS = ("peanut", "groundnut", "satay")
GLUTEN = ("wheat", "flour", "bread", "naan", "roti", "chapati", "paratha", "puri", "bhatura", "kulcha",
          "pasta", "spaghetti", "penne", "linguine", "tagliatelle", "fettuccine", "lasagne", "lasagna",
          "ravioli", "tortellini", "gnocchi", "rigatoni", "pappardelle", "orecchiette", "macaroni",
          "pizza", "calzone", "focaccia", "ciabatta", "bruschetta", "crouton", "breadcrumb", "semolina",
          "couscous", "barley", "rye", "samosa", "pakora", "bhaji", "batter", "seitan", "biscuit", "cake",
          "savoiardi", "panko", "dough", "cannoli", "arancini")
SESAME = ("sesame", "tahini")
MUSTARD = ("mustard",)
CELERY = ("celery", "celeriac")
SOYA = ("soy", "soya", "tofu", "edamame", "miso", "tempeh")
SULPHITES = ("wine", "vinegar", "balsamic")

ALLERGEN_TERMS = {
    "fish": SEAFOOD_FISH, "crustaceans": SHELLFISH, "molluscs": MOLLUSCS, "milk": DAIRY, "egg": EGG,
    "nuts": NUTS, "peanuts": PEANUTS, "gluten": GLUTEN, "sesame": SESAME, "mustard": MUSTARD,
    "celery": CELERY, "soya": SOYA, "sulphites": SULPHITES,
}


# Plant "milks" and nut butters are not dairy; removed from the text before the DAIRY check only
# (an "almond milk" still counts for the nuts allergen).
PLANT_DAIRY = re.compile(r"\b(coconut|almond|oat|soy|soya|rice|cashew)\s+(milk|cream|butter|yogh?urt)\b"
                         r"|\b(peanut|cocoa|nut)\s+butter\b", re.I)


@lru_cache(maxsize=None)
def _pattern(terms: tuple[str, ...]) -> re.Pattern:
    return re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")(e?s)?\b", re.I)


def dish_text(dish: Dish) -> str:
    return " | ".join([dish.name, dish.description or "", *dish.ingredients, *dish.ingredients_canonical])


def mentions(dish: Dish, terms: tuple[str, ...]) -> str | None:
    text = dish_text(dish)
    if terms is DAIRY:
        text = PLANT_DAIRY.sub(" ", text)
    m = _pattern(terms).search(text)
    return m.group(1).lower() if m else None


def implied_allergens(dish: Dish) -> set[str]:
    """Allergens the dish text implies, on top of the listed ones."""
    return {a for a, terms in ALLERGEN_TERMS.items() if mentions(dish, terms)}


def rule_out(dish: Dish, restriction: str) -> str | None:
    """Why the dish text rules out a dietary restriction, or None. Never says a dish IS compliant."""
    meat, seafood = mentions(dish, MEAT), mentions(dish, SEAFOOD_FISH + SHELLFISH + MOLLUSCS)
    if restriction == "vegetarian":
        return meat or seafood
    if restriction == "vegan":
        return meat or seafood or mentions(dish, DAIRY) or mentions(dish, EGG + HONEY)
    if restriction == "halal":
        return mentions(dish, PORK + ALCOHOL)
    if restriction == "kosher":
        hit = mentions(dish, PORK + SHELLFISH + MOLLUSCS)
        dairy = mentions(dish, DAIRY)
        return hit or (f"{meat} with {dairy}" if meat and dairy else None)
    return None


def effective_dietary(dish: Dish, restriction: str) -> tuple[str, str | None]:
    """(flag, reason): the data's flag, downgraded to "no" when the text rules it out."""
    reason = rule_out(dish, restriction)
    if reason:
        return "no", reason
    flag = getattr(dish.dietary, restriction)
    if restriction == "vegetarian" and flag != "yes" and dish.dietary.vegan == "yes":
        flag = "yes"  # vegan implies vegetarian (data says vegan, so this is not a guess)
    return flag, None
