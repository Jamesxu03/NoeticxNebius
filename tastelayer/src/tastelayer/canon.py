"""Ingredient canonicalisation: lowercase, strip quantities/units/prep words, singularise, map synonyms."""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

SYNONYMS_PATH = Path(__file__).parent / "data" / "synonyms.yaml"

UNITS = {
    "cup", "cups", "tablespoon", "tablespoons", "tbsp", "tbs", "teaspoon", "teaspoons", "tsp", "grams", "g", "gm", "gms",
    "kg", "kilogram", "ml", "l", "litre", "liter", "litres", "liters", "oz", "ounce", "ounces", "lb", "lbs", "pound",
    "pounds", "pinch", "dash", "handful", "sprig", "sprigs", "bunch", "clove", "cloves", "inch", "inches", "piece",
    "pieces", "can", "cans", "tin", "tins", "packet", "pack", "package", "jar", "bottle", "slice", "slices", "stick",
    "sticks", "small", "medium", "large", "big", "whole", "few", "some", "nos", "no", "number", "pod", "pods",
}
PREP_WORDS = {
    "chopped", "finely", "roughly", "coarsely", "fresh", "freshly", "sliced", "thinly", "thickly", "diced", "minced",
    "grated", "crushed", "peeled", "deseeded", "seeded", "cubed", "halved", "quartered", "julienned", "shredded",
    "boiled", "cooked", "uncooked", "raw", "soaked", "drained", "rinsed", "washed", "trimmed", "softened", "melted",
    "beaten", "whisked", "sifted", "toasted", "roasted", "optional", "taste", "required", "needed", "as", "per",
    "to", "for", "garnish", "garnishing", "serving", "of", "and", "or", "a", "an", "the", "about", "approx",
    "approximately", "plus", "more", "extra", "divided", "room", "temperature", "cold", "hot", "warm", "lukewarm",
    "frozen", "thawed", "packed", "heaped", "level", "tender", "ripe", "organic", "pitted", "lightly", "well",
    "into", "cut", "cuts", "strips", "rings", "wedges", "pieces", "chunks", "dry", "dried", "store", "bought",
    "homemade", "good", "quality", "low", "fat", "reduced", "sodium", "unsalted", "salted", "boneless", "skinless",
    "mashed", "pureed", "blanched", "steamed", "fried", "deep",
}
# Keep these prep words when they form the ingredient identity.
KEEP_PHRASES = {
    "dried fenugreek leaves", "dry mango powder", "sun dried tomato", "sun-dried tomato", "roasted red pepper",
    "fried onion", "dried oregano", "dried basil", "dried red chilli", "cold water", "hot water",
}
# Strip "ground" except where it names a distinct product.
GROUND_KEEP = {"ground beef", "ground lamb", "ground pork", "ground turkey", "ground chicken"}

_num_re = re.compile(r"[\d½¼¾⅓⅔/.\-–]+")
_paren_re = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_nonword_re = re.compile(r"[^a-z\s\-']")


@lru_cache(maxsize=1)
def synonyms() -> dict[str, str]:
    with open(SYNONYMS_PATH) as f:
        return {str(k).lower(): str(v).lower() for k, v in yaml.safe_load(f).items()}


def singularize(word: str) -> str:
    if len(word) <= 3 or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("oes", "ches", "shes", "xes")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


@lru_cache(maxsize=200_000)
def canonicalize(s: str) -> str | None:
    if not s or not isinstance(s, str):
        return None
    x = s.lower().strip()
    x = _paren_re.sub(" ", x)
    x = x.split(",")[0]  # "onion, finely chopped" -> "onion"
    x = re.sub(r"\s+-\s+.*$", "", x)  # "salt - to taste" -> "salt"
    x = _num_re.sub(" ", x)
    x = _nonword_re.sub(" ", x)
    x = re.sub(r"\s+", " ", x).strip()
    if not x:
        return None
    syn = synonyms()
    if x in syn:
        return syn[x]
    if x in KEEP_PHRASES:
        return syn.get(x, x)
    if x not in GROUND_KEEP:
        x = re.sub(r"\bground\b", " ", x)
    toks = [t for t in x.split() if t not in UNITS and t not in PREP_WORDS]
    if not toks:
        return None
    toks = toks[:-1] + [singularize(toks[-1])]
    x = " ".join(toks).strip(" -'")
    if len(x) < 2:
        return None
    return syn.get(x, x)


def canonicalize_list(items) -> list[str]:
    out, seen = [], set()
    for it in items or []:
        c = canonicalize(it)
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out
