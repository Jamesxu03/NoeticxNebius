"""
Regenerate backend/mocks/*.json by calling the real endpoints once.

The frontend codes against these files until the API is deployed, so rerun this
whenever a response shape changes:

    python -m backend.make_mocks
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app

OUT = Path(__file__).resolve().parent / "mocks"


def main() -> None:
    c = TestClient(app)

    def save(name: str, resp) -> dict | list:
        resp.raise_for_status()
        data = resp.json()
        (OUT / f"{name}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        return data

    save("taste_dimensions", c.get("/taste-dimensions"))
    profile = save("profile", c.post("/profile", json={
        "first_name": "Alex", "last_name": "Demo", "dietary": ["vegetarian"],
        "allergies": ["nuts"], "disliked_ingredients": ["coriander"],
    }))
    pid = profile["id"]
    save("profile_after_tolerances", c.put(f"/profile/{pid}/tolerances", json={
        "spice": 7, "sweet": 4, "sour": 5, "bitter": 3, "umami": 7, "richness": 6, "crunch": 5, "creaminess": 6,
    }))
    dishes = save("survey_dishes", c.get("/survey/dishes", params={"profile_id": pid}))
    save("profile_after_ratings", c.post(f"/profile/{pid}/ratings", json={
        "ratings": [{"dish_id": d["id"], "stars": s} for d, s in zip(dishes, [5, 4, 2, 3, 5, 1, 4])],
    }))
    rounds = save("venue_rounds", c.get("/survey/venue-rounds", params={"profile_id": pid}))
    for r in rounds:
        ids = [o["restaurant"]["id"] for o in r["options"]]
        c.post(f"/profile/{pid}/venue-choice", json={"round_id": r["round_id"], "options": ids, "chosen": ids[1]})
    sample = Path(__file__).resolve().parent / "fixtures/uber_eats_orders_sample.csv"  # synthetic, not real data
    save("uber_import", c.post(f"/profile/{pid}/import/uber",
                               files={"file": ("orders.csv", sample.read_bytes(), "text/csv")}))
    save("taste_map", c.get(f"/profile/{pid}/taste-map"))
    save("recommend", c.post("/recommend", json={"profile_id": pid, "occasion": "date-night"}))
    print(f"wrote {len(list(OUT.glob('*.json')))} mock files to {OUT}")


if __name__ == "__main__":
    main()
