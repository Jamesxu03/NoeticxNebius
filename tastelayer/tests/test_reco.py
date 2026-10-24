"""Recommender tests. Run: .venv/bin/python -m pytest tests/test_reco.py -q --cov=tastelayer.reco.filters --cov-branch"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from tastelayer.reco import filters, recommend as rec_mod, score as score_mod
from tastelayer.reco.filters import apply_filters, check_dish, norm_allergen
from tastelayer.reco.profile import RecoProfile, as_reco
from tastelayer.reco.recommend import Catalog, recommend
from tastelayer.reco.score import IngredientSpace, aggregate, explain, price_fit, rank, score_dish, taste_fit, vibe_fit
from tastelayer.reco.update import adventurousness, init_from_sliders, new_profile, rate, venue_choice
from tastelayer.schema import TASTE_DIMS, Dietary, Dish, Restaurant, TasteVector, UserProfile

YES = Dietary(vegan="yes", vegetarian="yes", halal="yes", kosher="yes")


def dish(i="d_1", rid="r_1", taste=None, **kw) -> Dish:
    base = dict(id=i, restaurant_id=rid, name=kw.pop("name", f"Dish {i}"), cuisine="indian",
                taste=TasteVector(**(taste or {d: 5 for d in TASTE_DIMS})))
    base.update(kw)
    return Dish(**base)


def prof(**kw) -> RecoProfile:
    return RecoProfile(id="u", **kw)


# ---------------- filters: every branch ----------------

def test_norm_allergen_synonyms():
    assert norm_allergen(" Peanuts ") == "peanut"
    assert norm_allergen("tree nuts") == norm_allergen("cashew") == "nuts"
    assert norm_allergen("dairy") == "milk"
    assert norm_allergen("celery") == "celery"


def test_no_restrictions_keeps_unverified_without_warning():
    assert check_dish(dish(verified=False), prof()) == (True, [], None)


def test_allergen_hit_excludes_even_when_non_strict():
    d = dish(allergens=["Peanut", "milk"], verified=True, dietary=YES)
    keep, w, reason = check_dish(d, prof(allergies=["peanuts"]), strict=False)
    assert not keep and reason == "contains allergen: peanut"


def test_allergen_family_match():
    d = dish(allergens=["cashew"], verified=True)
    assert check_dish(d, prof(allergies=["tree nuts"]))[2] == "contains allergen: nuts"


def test_blank_allergy_is_ignored():
    assert check_dish(dish(verified=False), prof(allergies=["  "])) == (True, [], None)


def test_allergy_unverified_strict_excludes():
    assert check_dish(dish(verified=False), prof(allergies=["peanut"]), strict=True)[2] == "allergens not verified"


def test_allergy_unverified_non_strict_warns():
    assert check_dish(dish(verified=False), prof(allergies=["peanut"]), strict=False) == (True, ["allergens not verified"], None)


def test_allergy_verified_clean_keeps():
    assert check_dish(dish(verified=True, allergens=["milk"]), prof(allergies=["peanut"])) == (True, [], None)


def test_dietary_no_excludes_even_non_strict():
    d = dish(verified=True, dietary=Dietary(vegan="no", vegetarian="yes", halal="yes", kosher="yes"))
    assert check_dish(d, prof(dietary=["Vegan"]), strict=False)[2] == "not vegan"


def test_dietary_unknown_strict_excludes_and_non_strict_warns():
    d = dish(verified=True, dietary=Dietary(halal="unknown"))
    assert check_dish(d, prof(dietary=["halal"]), strict=True)[2] == "unverified: halal"
    assert check_dish(d, prof(dietary=["halal"]), strict=False) == (True, ["unverified: halal"], None)


def test_dietary_yes_but_dish_unverified_is_not_safe():
    d = dish(verified=False, dietary=YES)
    assert check_dish(d, prof(dietary=["kosher"]), strict=True)[2] == "unverified: kosher"
    assert check_dish(d, prof(dietary=["kosher"]), strict=False)[1] == ["unverified: kosher"]


def test_dietary_verified_yes_keeps():
    assert check_dish(dish(verified=True, dietary=YES), prof(dietary=["vegan", "halal"])) == (True, [], None)


def test_vegan_implies_vegetarian():
    d = dish(verified=True, dietary=Dietary(vegan="yes", vegetarian="unknown"))
    assert check_dish(d, prof(dietary=["vegetarian"])) == (True, [], None)


def test_vegetarian_no_with_non_vegan():
    d = dish(verified=True, dietary=Dietary(vegan="no", vegetarian="no"))
    assert check_dish(d, prof(dietary=["vegetarian"]))[2] == "not vegetarian"


def test_unsupported_restriction():
    d = dish(verified=True, dietary=YES)
    assert check_dish(d, prof(dietary=["pescatarian"]), strict=True)[2] == "cannot check restriction: pescatarian"
    assert check_dish(d, prof(dietary=["pescatarian"]), strict=False) == (True, ["cannot check restriction: pescatarian"], None)


def test_disliked_ingredient_sources():
    p = prof(disliked_ingredients=["coriander"])
    assert check_dish(dish(ingredients=["Coriander"]), p)[2] == "disliked ingredient: coriander"
    assert check_dish(dish(ingredients_canonical=["coriander"]), p)[0] is False
    assert check_dish(dish(name="Coriander chicken"), p)[0] is False
    assert check_dish(dish(description="garnished with corianders"), p)[0] is False  # plural
    assert check_dish(dish(ingredients=["cumin"]), p) == (True, [], None)


def test_disliked_is_whole_word_and_blank_ignored():
    p = prof(disliked_ingredients=["ham", " "])
    assert check_dish(dish(name="Graham cracker", description="with champagne"), p)[0] is True
    assert check_dish(dish(name="Ham & pineapple"), p)[0] is False


def test_warnings_accumulate_then_disliked_still_excludes():
    p = prof(allergies=["peanut"], dietary=["halal"], disliked_ingredients=["onion"])
    keep, w, _ = check_dish(dish(verified=False), p, strict=False)
    assert keep and w == ["allergens not verified", "unverified: halal"]
    assert check_dish(dish(verified=False, ingredients=["onion"]), p, strict=False)[2] == "disliked ingredient: onion"


def test_apply_filters_splits_and_preserves_order():
    ds = [dish("d_2", allergens=["peanut"], verified=True), dish("d_1", verified=True), dish("d_3", verified=True)]
    kept, excl = apply_filters(prof(allergies=["peanut"]), ds)
    assert [d.id for d, _ in kept] == ["d_1", "d_3"] and [(d.id, r) for d, r in excl] == [("d_2", "contains allergen: peanut")]


def test_filters_ignore_model_outputs():
    p = prof(allergies=["milk"], dietary=["vegan"], disliked_ingredients=["chilli"])
    a = dish(verified=True, dietary=YES)
    b = a.model_copy(update={"taste": TasteVector(**{d: 10 for d in TASTE_DIMS}), "allergens_suggested": ["milk"],
                             "taste_confidence": {d: 0.0 for d in TASTE_DIMS}, "tagged_by": "trained-regressor"})
    assert check_dish(a, p) == check_dish(b, p) == (True, [], None)
    src = Path(filters.__file__).read_text()
    code = re.sub(r'""".*?"""', "", src, flags=re.S)
    for forbidden in (".taste", "allergens_suggested", "taste_confidence", "tagged_by", "lexicon", "model"):
        assert forbidden not in code, forbidden


# ---------------- scoring ----------------

def _catalog(n=6):
    rests = {"r_1": Restaurant(id="r_1", name="A", cuisine="indian", price_tier=1, vibe_tags=["quick", "casual"]),
             "r_2": Restaurant(id="r_2", name="B", cuisine="indian", price_tier=4, vibe_tags=["date-night", "trendy"])}
    rng = np.random.default_rng(0)
    vocab = ["chilli", "cream", "lemon", "rice", "onion", "basil"]
    space = IngredientSpace(rng.normal(size=(len(vocab), 8)), {w: i for i, w in enumerate(vocab)})
    dishes = {}
    for i in range(n):
        t = {d: float(rng.integers(0, 11)) for d in TASTE_DIMS}
        dishes[f"d_{i}"] = dish(f"d_{i}", rid="r_1" if i % 2 else "r_2", taste=t, verified=True, dietary=YES,
                                ingredients_canonical=[vocab[i % 6], vocab[(i + 2) % 6]])
    return Catalog(dishes, rests, space, {k: "test" for k in dishes})


def test_taste_fit_bounds_and_weights():
    u = [5.0] * 8
    assert taste_fit(u, u, [0.3] * 8) == pytest.approx(1.0)
    assert taste_fit([0] * 8, [10] * 8, [0.3] * 8) == pytest.approx(0.0)
    x = [5, 5, 5, 5, 5, 5, 5, 0]
    assert taste_fit(u, x, [0.3] * 8, {"creaminess": 3.0}) < taste_fit(u, x, [0.3] * 8)


def test_price_and_vibe_fit():
    assert price_fit(2, 2.0) == 1.0 and price_fit(4, 1.0) == 0.0 and price_fit(4, 2.0, shift=2.0) == 1.0
    assert vibe_fit(["quick"], {}) == 0.5
    assert vibe_fit(["quick", "casual"], {"quick": 0.5, "casual": 0.5}) == 1.0
    assert vibe_fit(["trendy"], {"quick": 1.0}) == 0.0


def test_budget_zeroes_price_fit():
    cat = _catalog()
    d = cat.dishes["d_0"].model_copy(update={"price_gbp": 30.0})
    s = score_dish(prof(), d, cat.restaurants[d.restaurant_id], cat.space, budget=20)
    assert s["raw"]["price_fit"] == 0.0


def test_deterministic_ranking_and_tiebreak():
    cat = _catalog()
    p = prof(taste_vector={d: 6 for d in TASTE_DIMS}, liked_ingredients=["chilli"])
    a = recommend(p, catalog=cat, k=10)
    b = recommend(p, catalog=cat, k=10)
    assert a == b
    twin = {"d_a": dish("d_a", verified=True, dietary=YES), "d_b": dish("d_b", verified=True, dietary=YES)}
    cat2 = Catalog(twin, cat.restaurants, None, {})
    assert [d["dish_id"] for d in recommend(prof(), catalog=cat2)["top_dishes"]] == ["d_a", "d_b"]


def test_filters_run_before_scoring(monkeypatch):
    cat = _catalog()
    cat.dishes["d_0"] = cat.dishes["d_0"].model_copy(update={"allergens": ["peanut"]})
    seen = []
    real = score_mod.score_dish

    def spy(p, d, *a, **kw):
        seen.append(d.id)
        return real(p, d, *a, **kw)

    monkeypatch.setattr(rec_mod, "score_dish", spy)
    out = recommend(prof(allergies=["peanut"]), catalog=cat)
    assert "d_0" not in seen and len(seen) == len(cat.dishes) - 1 and out["excluded"] == 1


def test_strict_unknown_gives_empty_with_message_and_non_strict_warns():
    cat = _catalog()
    cat.dishes = {k: v.model_copy(update={"verified": False}) for k, v in cat.dishes.items()}
    p = prof(dietary=["halal"])
    s = recommend(p, catalog=cat)
    assert s["results"] == [] and "halal" in s["message"]
    ns = recommend(p, catalog=cat, strict=False)
    assert ns["results"] and all("unverified: halal" in d["warnings"] for d in ns["top_dishes"])


def test_theta_zero_equals_base_and_theta_changes_targets():
    cat = _catalog()
    dirs = {d: np.eye(8)[i] for i, d in enumerate(TASTE_DIMS)}
    rec_mod.tmf.load_directions.cache_clear()
    import tastelayer.reco.slerp as sl
    orig = sl.load_directions
    sl.load_directions = lambda *a, **k: dirs
    try:
        p = prof(liked_ingredients=["cream"], taste_vector={d: 4 for d in TASTE_DIMS})
        base = recommend(p, catalog=cat, k=10)
        zero = recommend(p, catalog=cat, k=10, theta=0, direction="spice")
        assert base == zero
        far = recommend(p, catalog=cat, k=10, theta=60, direction="spice")
        assert any("further" in r["dishes"][0]["breakdown"] for r in far["results"])
        with pytest.raises(ValueError):
            recommend(p, catalog=cat, theta=30, direction="nope")
    finally:
        sl.load_directions = orig


def test_aggregate_shape_and_restaurant_score():
    cat = _catalog()
    p = prof()
    scored = []
    for d in cat.dishes.values():
        s = score_dish(p, d, cat.restaurants[d.restaurant_id], cat.space)
        s["explanation"] = explain(s, p)
        scored.append(s)
    out = aggregate(rank(scored), cat.dishes, cat.restaurants, {}, k=5)
    assert {"restaurant", "dishes", "score", "explanation", "warnings"} <= set(out[0])
    for r in out:
        assert r["score"] == pytest.approx(np.mean([d["score"] for d in r["dishes"]]), abs=1e-3)
        assert len(r["dishes"]) <= 3
    assert out == sorted(out, key=lambda o: (-o["score"], o["restaurant"]["id"]))


def test_explanation_mentions_shared_loves():
    p = prof(taste_vector={**{d: 3 for d in TASTE_DIMS}, "spice": 9, "creaminess": 8})
    d = dish(taste={**{k: 2 for k in TASTE_DIMS}, "spice": 8, "creaminess": 9})
    r = Restaurant(id="r_1", name="A", cuisine="indian", price_tier=2)
    text = explain(score_dish(p, d, r), p)
    assert text.startswith("Matches your love of") and "heat" in text and "creaminess" in text


# ---------------- update ----------------

def test_init_from_sliders_and_new_profile():
    u, c = init_from_sliders({"spice": 12, "sweet": 3})
    assert u["spice"] == 10 and u["sour"] == 5 and set(c.values()) == {0.3}
    p = new_profile("u_x", {"spice": 9})
    assert 0 <= p.adventurousness <= 10 and p.taste_confidence["spice"] == 0.3


def test_rating_moves_toward_and_away():
    p = prof(taste_vector={d: 5 for d in TASTE_DIMS})
    x = dish(taste={**{d: 5 for d in TASTE_DIMS}, "spice": 9})
    up = rate(p, x, 5)
    assert up.taste_vector["spice"] == pytest.approx(5 + 0.3 * 1.0 * 4)
    down = rate(p, x, 1)
    assert down.taste_vector["spice"] < 5
    same = rate(p, x, 3)
    assert same.taste_vector == p.taste_vector and same.conf() == p.conf()
    assert up.taste_confidence["spice"] == pytest.approx(0.35) and len(up.dish_ratings) == 1
    assert p.dish_ratings == []  # input untouched
    with pytest.raises(ValueError):
        rate(p, x, 6)


def test_alpha_decays_with_more_ratings():
    x = dish(taste={**{d: 5 for d in TASTE_DIMS}, "spice": 10})
    p = prof(taste_vector={d: 5 for d in TASTE_DIMS})
    first = rate(p, x, 5).taste_vector["spice"] - 5
    p10 = p.model_copy(update={"dish_ratings": rate(p, x, 3).dish_ratings * 10})
    later = rate(p10, x, 5).taste_vector["spice"] - 5
    assert later < first


def test_centroid_ema_and_venue_choice():
    cat = _catalog()
    p = prof()
    d = cat.dishes["d_0"]
    p1 = rate(p, d, 5, cat.space)
    assert p1.ingredient_centroid is not None and np.isclose(np.linalg.norm(p1.ingredient_centroid), 1)
    p2 = rate(p1, cat.dishes["d_1"], 4, cat.space)
    assert not np.allclose(p1.ingredient_centroid, p2.ingredient_centroid)
    v = venue_choice(prof(price_pref=1.0), "v1", ["r_1", "r_2"], cat.restaurants["r_2"], d, cat.space)
    assert v.price_pref == pytest.approx(0.7 + 1.2)
    assert sum(v.vibe_pref.values()) == pytest.approx(1.0) and v.vibe_pref["trendy"] > 0
    assert v.venue_choices[0].chosen == "r_2" and v.dish_ratings[0].stars == 4


def test_adventurousness_formula():
    flat = prof(taste_vector={d: 5 for d in TASTE_DIMS})
    bold = prof(taste_vector={**{d: 5 for d in TASTE_DIMS}, "spice": 10, "sour": 10, "bitter": 10})
    assert adventurousness(bold) > adventurousness(flat)
    assert adventurousness(flat) == pytest.approx(10 * (0.4 * 0.5 + 0.0 + 0.4 * 0.5))
    cat = _catalog()
    rated = flat.model_copy(update={"liked_ingredients": ["chilli"],
                                    "dish_ratings": rate(flat, cat.dishes["d_3"], 5).dish_ratings})
    assert 0 <= adventurousness(rated, cat.dishes, cat.space) <= 10


def test_as_reco_accepts_contract_profile():
    assert as_reco(UserProfile(id="u")).strict is True
    assert as_reco({"id": "u", "strict": False}).strict is False
