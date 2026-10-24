"""
Validate a data folder against the shared contracts and check cross-references.

    python -m schema.validate data/mock
    python -m schema.validate data

Exits non-zero if anything fails, so the data role can run it before every push.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from schema.models import TASTE_KEYS, Dish, Restaurant

MIN_VENUE_FAMILIES = 3  # brief 4.5: "same dish at 3 venues" needs at least 3 dish families


def main(folder: str) -> int:
    base = Path(folder)
    errors: list[str] = []

    try:
        restaurants = TypeAdapter(list[Restaurant]).validate_json((base / "restaurants.json").read_text())
        dishes = TypeAdapter(list[Dish]).validate_json((base / "dishes.json").read_text())
    except ValidationError as exc:
        print(exc)
        return 1

    dims = json.loads(Path("data/taste_dimensions.json").read_text())
    if [d["key"] for d in dims["dimensions"]] != list(TASTE_KEYS):
        errors.append("taste_dimensions.json keys do not match schema.models.TASTE_KEYS")

    by_rest = {r.id: r for r in restaurants}
    dish_ids = {d.id for d in dishes}
    if len(dish_ids) != len(dishes):
        errors.append("duplicate dish ids")
    for d in dishes:
        if d.restaurant_id not in by_rest:
            errors.append(f"{d.id}: unknown restaurant {d.restaurant_id}")
        elif d.id not in by_rest[d.restaurant_id].dish_ids:
            errors.append(f"{d.id}: missing from {d.restaurant_id}.dish_ids")
    for r in restaurants:
        for i in r.dish_ids:
            if i not in dish_ids:
                errors.append(f"{r.id}: dish_ids lists unknown dish {i}")

    tiers = defaultdict(set)
    for d in dishes:
        if d.restaurant_id in by_rest:
            tiers[d.dish_family].add(by_rest[d.restaurant_id].price_tier)
    venue_families = sorted(f for f, t in tiers.items() if len(t) >= 3)
    if len(venue_families) < MIN_VENUE_FAMILIES:
        errors.append(f"only {len(venue_families)} dish families are served at 3+ price tiers "
                      f"({venue_families}); need {MIN_VENUE_FAMILIES} for venue rounds")

    unverified = sum(not d.verified for d in dishes)
    print(f"{len(restaurants)} restaurants, {len(dishes)} dishes, {unverified} not yet verified")
    print(f"venue-round families (3+ price tiers): {', '.join(venue_families) or 'none'}")
    for e in errors:
        print(f"ERROR {e}")
    print("OK" if not errors else f"{len(errors)} error(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "data"))
