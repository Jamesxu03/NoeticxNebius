"""Backend tests: hard-filter safety per persona, Epicure behaviour, LLM fallback, API flow.

    python -m pytest
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import llm
from backend import recommender as rec
from backend.flavour import FlavourIndex
from backend.main import PROFILES, app, cat
from schema.models import Dish, UserProfile

client = TestClient(app)
OCCASIONS = [*rec.OCCASIONS, "unlisted-occasion"]


# ----- safety: the rules BRIEF 3 and 8.4 make non-negotiable -----

@pytest.mark.parametrize("pid", sorted(PROFILES))
@pytest.mark.parametrize("occasion", OCCASIONS)
@pytest.mark.parametrize("budget", [None, 1, 4])
def test_recommendations_respect_hard_filters(pid, occasion, budget):
    profile = PROFILES[pid]
    results = rec.recommend(profile, cat, occasion, budget=budget)
    for r in results:
        if budget is not None:
            assert r.restaurant.price_tier <= budget
        for d in r.dishes:
            assert not set(d.allergens) & set(profile.allergies), f"{d.name} has an allergen of {pid}"
            for restriction in profile.dietary:
                status = getattr(d.dietary, restriction)
                assert status != "no", f"{d.name} is not {restriction}"
                if status == "unknown":
                    assert any(d.name in w and restriction in w for w in r.warnings)
            if profile.allergies and not d.verified:
                assert any(d.name in w and "allergen" in w for w in r.warnings)
        assert len(r.warnings) == len(set(r.warnings))


@pytest.mark.parametrize("pid", sorted(PROFILES))
def test_survey_and_venue_rounds_are_safe(pid):
    profile = PROFILES[pid]
    dishes = rec.survey_dishes(profile, cat)
    assert len({d.id for d in dishes}) == len(dishes)
    for d in dishes + [o.dish for r in rec.venue_rounds(profile, cat) for o in r.options]:
        assert rec.hard_filter(d, profile)[0]


def test_disliked_matches_whole_words():
    dish = cat.dishes["d_0001"].model_copy(update={"ingredients": ["eggplant", "chickpeas", "coriander leaves"]})
    assert rec.disliked_in(dish, ["egg"]) == []
    assert rec.disliked_in(dish, ["chickpea", "Coriander"]) == ["chickpea", "Coriander"]


# ----- Epicure -----

def test_epicure_loads_and_is_centred():
    assert cat.flavour.available
    assert len(cat.flavour.dish_vecs) == len(cat.dishes)
    X = np.stack(list(cat.flavour.dish_vecs.values()))
    assert np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-5)
    # centring is what makes cosines informative: raw dish vectors average ~0.5+ cosine
    assert abs((X @ X.T)[np.triu_indices(len(X), 1)].mean()) < 0.1


def test_same_family_dishes_are_closer():
    biryanis = [d for d in cat.dishes.values() if d.dish_family == "biryani"]
    others = [d for d in cat.dishes.values() if d.cuisine == "italian"]
    same = np.mean([cat.flavour.dish_similarity(biryanis[0].id, d.id) for d in biryanis[1:]])
    diff = np.mean([cat.flavour.dish_similarity(biryanis[0].id, d.id) for d in others])
    assert same > diff + 0.3


def test_five_star_rating_lifts_similar_dishes():
    base = UserProfile(id="t")
    fan = UserProfile(id="t", dish_ratings=[{"dish_id": "d_0022", "stars": 5}])  # Chicken Biryani
    vec = rec.epicure_user_vector(fan, cat)
    assert rec.epicure_user_vector(base, cat) is None
    lamb_biryani, tiramisu = cat.dishes["d_0033"], cat.dishes["d_0004"]
    rest = cat.restaurants[lamb_biryani.restaurant_id]
    lift = (rec.score_dish(lamb_biryani, rest, fan, "family", vec, cat)
            - rec.score_dish(lamb_biryani, rest, base, "family", None, cat))
    assert lift > 0
    assert cat.flavour.similarity(lamb_biryani.id, vec) > cat.flavour.similarity(tiramisu.id, vec)


def test_disliked_ingredient_pulls_flavour_away():
    p = UserProfile(id="t", disliked_ingredients=["anchovies"])
    vec = rec.epicure_user_vector(p, cat)
    assert cat.flavour.similarity("d_0016", vec) < 0  # Grilled Radicchio & Anchovy


@pytest.mark.parametrize("pid", sorted(PROFILES))
def test_stretch_pick_is_new_and_safe(pid):
    profile = PROFILES[pid]
    results = rec.recommend(profile, cat, "date-night")
    stretch = [r for r in results if r.kind == "stretch"]
    assert len(stretch) <= 1 and all(r.kind == "match" for r in results[:-1])
    if stretch:
        dish = stretch[0].dishes[0]
        rated = {cat.dishes[r.dish_id].dish_family for r in profile.dish_ratings}
        shown = {d.id for r in results[:-1] for d in r.dishes}
        assert dish.dish_family not in rated and dish.id not in shown
        assert rec.hard_filter(dish, profile)[0] and not rec.disliked_in(dish, profile.disliked_ingredients)


def test_because_you_liked_names_a_highly_rated_dish():
    profile = PROFILES["u_002"]
    loved = {cat.dishes[r.dish_id].name for r in profile.dish_ratings if r.stars >= 4}
    reasons = [r.because_you_liked for r in rec.recommend(profile, cat, "comfort") if r.because_you_liked]
    assert reasons and set(reasons) <= loved


def test_runs_without_epicure(monkeypatch):
    missing = FlavourIndex(cat.dishes, asset="does/not/exist.npz")
    assert not missing.available
    monkeypatch.setattr(cat, "flavour", missing)
    results = rec.recommend(PROFILES["u_001"], cat, "date-night")
    assert results and all(r.kind == "match" and r.because_you_liked is None for r in results)
    assert rec.survey_dishes(PROFILES["u_001"], cat)


# ----- LLM wrapper -----

@pytest.fixture
def fresh_llm(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "CACHE_PATH", tmp_path / "llm.json")
    monkeypatch.setattr(llm, "_cache", None)
    return monkeypatch


def test_llm_disabled_returns_template(fresh_llm):
    fresh_llm.delenv("NEBIUS_API_KEY", raising=False)
    assert llm.rewrite_explanation({"a": 1}, "template") == "template"


def test_llm_rewrites_and_caches(fresh_llm):
    calls = []

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "A lovely pick for you."}}]}

    fresh_llm.setenv("NEBIUS_API_KEY", "test-key")
    fresh_llm.setattr(llm.httpx, "post", lambda *a, **k: calls.append(k) or Resp())
    assert llm.rewrite_explanation({"a": 1}, "template") == "A lovely pick for you."
    assert llm.rewrite_explanation({"a": 1}, "template") == "A lovely pick for you."
    assert len(calls) == 1
    assert calls[0]["headers"]["Authorization"] == "Bearer test-key"


def test_llm_failure_falls_back(fresh_llm):
    def boom(*a, **k):
        raise RuntimeError("network down")

    fresh_llm.setenv("NEBIUS_API_KEY", "test-key")
    fresh_llm.setattr(llm.httpx, "post", boom)
    assert llm.rewrite_explanation({"a": 1}, "template") == "template"


# ----- API flow, as the frontend calls it -----

def test_full_survey_flow():
    p = client.post("/profile", json={"first_name": "Test", "dietary": ["vegetarian"], "allergies": ["nuts"]}).json()
    pid = p["id"]
    assert client.put(f"/profile/{pid}/tolerances", json={k: 6 for k in rec.TASTE_KEYS}).status_code == 200
    dishes = client.get("/survey/dishes", params={"profile_id": pid}).json()
    assert dishes
    ratings = [{"dish_id": d["id"], "stars": s} for d, s in zip(dishes, [5, 4, 1, 3, 5, 2])]
    assert client.post(f"/profile/{pid}/ratings", json={"ratings": ratings}).status_code == 200
    for r in client.get("/survey/venue-rounds", params={"profile_id": pid}).json():
        ids = [o["restaurant"]["id"] for o in r["options"]]
        assert client.post(f"/profile/{pid}/venue-choice",
                           json={"round_id": r["round_id"], "options": ids, "chosen": ids[0]}).status_code == 200
    assert client.get(f"/profile/{pid}/taste-map").status_code == 200
    recs = client.post("/recommend", json={"profile_id": pid, "occasion": "date-night"}).json()
    assert recs and {"kind", "because_you_liked"} <= set(recs[0])
    for r in recs:
        for d in r["dishes"]:
            assert "nuts" not in d["allergens"]
            assert d["dietary"]["vegetarian"] != "no"
    del PROFILES[pid]
