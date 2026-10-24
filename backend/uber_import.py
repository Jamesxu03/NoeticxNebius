"""
Warm-start a taste profile from a user's Uber Eats order history (their "Download your data" export).

Privacy rules, enforced here:
- From an export zip only the Eats order CSVs are opened (EATS_ORDER_FILES). Profile, payment,
  location and trip files are never read.
- Only item names, quantities, prices, dates and customisations are parsed. Special instructions,
  addresses and order ids are ignored.
- Nothing raw is stored. The profile keeps derived signals only (taste nudges, a price preference
  and `ordered_ingredients`), and the summary returned to the user gives counts, cuisines and ingredients.

Signals, all deterministic (no LLM):
- taste: TasteLayer's lexicon (tastelayer/src/tastelayer/lexicon.yaml) on the dims it matched, plus the
  tagged taste of catalogue dishes whose names share a distinctive word ("Margherita" -> our margheritas)
- flavour: Epicure ingredients found in item names (salmon, chorizo, wasabi...) -> profile.ordered_ingredients
- price: mean unit price -> price tier
"""

from __future__ import annotations

import csv
import io
import math
import re
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.flavour import resolve
from schema.models import TASTE_KEYS, TasteConfidence, TasteVector, UberImportSummary, UserProfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tastelayer/src"))
from tastelayer.lexicon import lookup_features  # noqa: E402  (pure python + yaml; no model weights)

MAX_BYTES = 20 * 1024 * 1024
EATS_ORDER_FILES = re.compile(r"(^|/)Eats/(eats_order_details|user_orders)[^/]*\.csv$", re.I)
LEXICON_DEFAULT = 2.0  # lexicon.yaml `defaults`: a dim left at this value had no match
MAX_TASTE_BLEND = 0.4  # orders move the slider vector at most 40% toward their estimate
TASTE_BLEND_PER_ITEM = 0.04
PRICE_TIERS = (8.0, 14.0, 22.0)  # GBP unit price upper bounds for tiers 1-3; above is tier 4
MAX_ORDERED_INGREDIENTS = 30
NAME_MATCH_MIN_IDF = 1.5  # a shared word must be this rare in catalogue names to count as a match
STOPWORDS = {
    "the", "and", "with", "for", "of", "a", "an", "in", "on", "no", "pcs", "pieces", "piece", "box", "feast",
    "kids", "kid", "meal", "deluxe", "platter", "persons", "person", "ways", "magic", "double", "regular",
    "large", "small", "medium", "upgrade", "bottle", "ml", "can", "classic", "special", "house", "combo",
    "side", "extra", "dip", "sandwich", "roll", "rolls", "set", "bundle", "portion", "half", "full", "one",
    "two", "three", "omega",
}
CUISINE_WORDS = {
    "japanese": ("sushi", "sashimi", "maki", "nigiri", "uramaki", "temaki", "wasabi", "teriyaki", "ramen",
                 "gyoza", "katsu", "tempura", "edamame", "miso", "poke"),
    "italian": ("pizza", "margherita", "pasta", "spaghetti", "penne", "lasagne", "risotto", "burrata",
                "mozzarella", "carbonara", "tiramisu", "calzone", "focaccia", "gnocchi", "sourdough"),
    "indian": ("curry", "tikka", "masala", "naan", "biryani", "korma", "dal", "paneer", "samosa", "tandoori",
               "bhaji", "vindaloo", "dosa"),
    "fast food": ("burger", "mcmuffin", "fries", "nuggets", "wrap", "hash brown", "milkshake", "mcflurry"),
    "chinese": ("dim sum", "chow mein", "kung pao", "bao", "dumpling", "wonton", "sweet and sour"),
}


@dataclass
class OrderItem:
    restaurant: str
    item: str
    quantity: int
    unit_price: float | None
    currency: str
    date: str
    customizations: str

    @property
    def text(self) -> str:
        return f"{self.item} {self.customizations}"


# ----- reading the export -----

def _clean(s: str) -> str:
    return re.sub(r"[®™©]", "", s or "").strip()


def _float(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _rows_to_items(rows: list[dict]) -> list[OrderItem]:
    items = []
    for r in rows:
        status = (r.get("Order_Status") or "completed").strip().lower()
        name = _clean(r.get("Item_Name", ""))
        if status != "completed" or not name:
            continue
        qty = max(int(_float(r.get("Item_quantity")) or 1), 1)
        price = _float(r.get("Item_Price"))
        items.append(OrderItem(
            restaurant=_clean(r.get("Restaurant_Name", "")),
            item=name,
            quantity=qty,
            unit_price=price / qty if price is not None else None,
            currency=(r.get("Currency") or "").strip(),
            date=(r.get("Request_Time_Local") or "")[:10],
            customizations=_clean(r.get("Customizations", "")),
        ))
    return items


def _read_csv(raw: bytes) -> list[dict]:
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig", errors="replace"))))
    if rows and "Item_Name" not in rows[0]:
        raise ValueError("CSV has no Item_Name column; expected an Uber Eats order file")
    return rows


def read_orders(data: bytes, filename: str = "") -> list[OrderItem]:
    """Order items from an Uber export zip or one of its Eats order CSVs."""
    if len(data) > MAX_BYTES:
        raise ValueError("file too large")
    if zipfile.is_zipfile(io.BytesIO(data)):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = [n for n in z.namelist() if EATS_ORDER_FILES.search(n)]
            # eats_order_details has the most columns; user_orders is the fallback
            names.sort(key=lambda n: "eats_order_details" not in n.lower())
            if not names:
                raise ValueError("no Eats order file (Eats/eats_order_details*.csv) in this zip")
            info = z.getinfo(names[0])
            if info.file_size > MAX_BYTES:
                raise ValueError("order file too large")
            return _rows_to_items(_read_csv(z.read(info)))
    if filename and not filename.lower().endswith(".csv"):
        raise ValueError("upload the Uber data zip or its Eats order CSV")
    return _rows_to_items(_read_csv(data))


# ----- signals -----

def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def detect_cuisines(items: list[OrderItem]) -> dict[str, int]:
    counts: Counter = Counter()
    for it in items:
        text = f"{it.restaurant} {it.item}".lower()
        for cuisine, words in CUISINE_WORDS.items():
            if any(re.search(rf"\b{re.escape(w)}", text) for w in words):
                counts[cuisine] += it.quantity
                break
    return dict(counts.most_common())


def ordered_ingredients(items: list[OrderItem], vocab: dict, alias: dict,
                        matcher: "NameMatcher | None" = None) -> dict[str, float]:
    """Epicure vocab ingredients for the ordered items, weighted by quantity.

    Two sources: ingredients named in the item text ("Salmon Avocado Roll"), and the ingredients
    of catalogue dishes the item name matches ("Margherita Pizza" -> dough, tomato, mozzarella...).
    """
    weights: Counter = Counter()
    for it in items:
        ids = matcher.match(it.item) if matcher else []
        for dish_id in ids:
            dish = matcher.dishes[dish_id]
            names = {n for ing in [*dish.ingredients, *dish.ingredients_canonical] for n in resolve(ing, vocab, alias)}
            for name in names:
                weights[name] += it.quantity / len(ids)
        words = [w for w in _words(it.text) if w not in STOPWORDS and len(w) > 2]
        found = set()
        for i, w in enumerate(words):
            if i + 1 < len(words):
                pair = resolve(f"{w} {words[i + 1]}", vocab, alias)
                if pair:
                    found.update(pair)
                    continue
            found.update(resolve(w, vocab, alias))
        for name in found:
            weights[name] += it.quantity
    return dict(weights.most_common(MAX_ORDERED_INGREDIENTS))


class NameMatcher:
    """Catalogue dishes sharing a distinctive word with an ordered item's name."""

    def __init__(self, dishes: dict):
        self.dishes = dishes
        self.index: dict[str, list[str]] = defaultdict(list)
        for d in dishes.values():
            for w in set(_words(d.name)) - STOPWORDS:
                self.index[w].append(d.id)
        n = max(len(dishes), 1)
        self.idf = {w: math.log(n / len(ids)) for w, ids in self.index.items()}

    def match(self, item_name: str, k: int = 5) -> list[str]:
        """Up to k catalogue dish ids sharing the item's rarest distinctive word."""
        words = [w for w in set(_words(item_name)) - STOPWORDS if self.idf.get(w, 0) >= NAME_MATCH_MIN_IDF]
        if not words:
            return []
        return self.index[max(words, key=lambda w: self.idf[w])][:k]

    def taste(self, item_name: str) -> np.ndarray | None:
        ids = self.match(item_name)
        if not ids:
            return None
        return np.mean([[getattr(self.dishes[i].taste, k) for k in TASTE_KEYS] for i in ids], axis=0)


def item_taste(it: OrderItem, matcher: NameMatcher) -> tuple[np.ndarray, np.ndarray]:
    """(scores, mask): per-dim estimate and which dims have evidence."""
    lex, _ = lookup_features({"name": it.item, "description": it.customizations, "ingredients": []})
    lex = np.asarray(lex, dtype=float)
    lex_mask = np.abs(lex - LEXICON_DEFAULT) > 1e-6
    cat = matcher.taste(it.item)
    if cat is None:
        return lex, lex_mask
    scores = np.where(lex_mask, (lex + cat) / 2, cat)
    return scores, np.ones(len(TASTE_KEYS), dtype=bool)


def price_tier(unit_price: float) -> int:
    return 1 + sum(unit_price > t for t in PRICE_TIERS)


# ----- apply to a profile -----

def apply_orders(profile: UserProfile, items: list[OrderItem], cat) -> UberImportSummary:
    """Nudge `profile` in place from order items and return a summary for the user."""
    matcher = NameMatcher(cat.dishes)
    sums, counts = np.zeros(len(TASTE_KEYS)), np.zeros(len(TASTE_KEYS))
    for it in items:
        scores, mask = item_taste(it, matcher)
        sums += scores * mask * it.quantity
        counts += mask * it.quantity

    user = np.array([getattr(profile.taste_vector, k) for k in TASTE_KEYS], dtype=float)
    conf = np.array([getattr(profile.taste_confidence, k) for k in TASTE_KEYS], dtype=float)
    alpha = np.minimum(MAX_TASTE_BLEND, TASTE_BLEND_PER_ITEM * counts)
    estimate = np.divide(sums, counts, out=user.copy(), where=counts > 0)
    new = np.clip(user + alpha * (estimate - user), 0, 10)
    profile.taste_vector = TasteVector(**dict(zip(TASTE_KEYS, np.round(new, 2))))
    profile.taste_confidence = TasteConfidence(**dict(zip(TASTE_KEYS, np.round(np.minimum(conf + alpha / 2, 1), 2))))

    gbp = [it.unit_price for it in items for _ in range(it.quantity)
           if it.unit_price is not None and it.currency.upper() in ("GBP", "")]
    if gbp:
        tier = float(np.mean([price_tier(p) for p in gbp]))
        weight = 0.3 if profile.venue_choices else 0.6  # venue picks are the stronger signal
        profile.price_pref = round(float(np.clip((1 - weight) * profile.price_pref + weight * tier, 1, 4)), 2)

    if cat.flavour.available:
        merged = Counter(profile.ordered_ingredients)
        merged.update(ordered_ingredients(items, cat.flavour.vocab, cat.flavour.alias, matcher))
        profile.ordered_ingredients = {k: round(v, 2) for k, v in merged.most_common(MAX_ORDERED_INGREDIENTS)}

    dates = sorted(it.date for it in items if it.date)
    return UberImportSummary(
        items=sum(it.quantity for it in items),
        restaurants=len({it.restaurant for it in items}),
        date_from=dates[0] if dates else "",
        date_to=dates[-1] if dates else "",
        cuisines=detect_cuisines(items),
        top_ingredients=list(profile.ordered_ingredients)[:8],
        taste_shift={k: round(float(d), 2) for k, d in zip(TASTE_KEYS, new - user) if abs(d) >= 0.05},
        price_pref=profile.price_pref,
    )
