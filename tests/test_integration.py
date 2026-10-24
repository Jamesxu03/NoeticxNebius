"""Tests for the merged pieces: safety rules, strict mode, TasteLayer catalogue, Uber Eats import.

    python -m pytest
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import recommender as rec
from backend import safety, uber_import
from backend.main import PROFILES, app, cat
from schema.models import Dish, UserProfile

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "backend/fixtures/uber_eats_orders_sample.csv"
client = TestClient(app)


@pytest.fixture(scope="module")
def tl_cat():
    return rec.Catalogue(str(ROOT / "data/tastelayer"))


def dish(**kw) -> Dish:
    base = dict(id="d_9999", restaurant_id="r_99", name="Test", cuisine="indian", category="main",
                dish_family="curry", price_gbp=10, ingredients=[], allergens=[], dietary={},
                taste={}, serving_temp="hot", adventurousness=5, tagged_by="human")
    return Dish(**{**base, **kw})


# ----- safety rules: exclusion only -----

@pytest.mark.parametrize("name,ingredients,restriction", [
    ("Chicken Korma", [], "vegetarian"),
    ("Calamari", [], "vegetarian"),
    ("Paneer Tikka", ["paneer"], "vegan"),
    ("Pepperoni Pizza", [], "halal"),
    ("Penne alla Vodka", [], "halal"),
    ("Prawn Curry", [], "kosher"),
    ("Chicken Alfredo", ["chicken", "cream"], "kosher"),
])
def test_text_rules_out_restriction(name, ingredients, restriction):
    d = dish(name=name, ingredients=ingredients, dietary={restriction: "yes"})
    assert safety.effective_dietary(d, restriction)[0] == "no"  # overrides even a "yes" in the data


def test_rules_never_upgrade_unknown():
    d = dish(name="Chana Masala", ingredients=["chickpeas", "coconut milk"])
    assert safety.effective_dietary(d, "vegan") == ("unknown", None)  # coconut milk is not dairy
    assert safety.effective_dietary(d, "vegetarian") == ("unknown", None)


@pytest.mark.parametrize("name,ingredients,allergen", [
    ("Pistachio Kulfi", [], "nuts"),
    ("Garlic Naan", [], "gluten"),
    ("Tiramisu", [], "egg"),
    ("Pesto Linguine", [], "nuts"),
    ("Chicken Satay", [], "peanuts"),
    ("Butter Chicken", [], "milk"),
])
def test_implied_allergens(name, ingredients, allergen):
    assert allergen in safety.implied_allergens(dish(name=name, ingredients=ingredients))


def test_no_false_allergens_from_substrings():
    d = dish(name="Eggplant Parmigiana with Butternut Squash and Nutmeg", ingredients=["coconut"])
    found = safety.implied_allergens(d)
    assert "egg" not in found and "nuts" not in found


def test_strict_mode_excludes_unknown_and_unverified():
    d = dish(name="Dal", ingredients=["lentils"])
    loose = UserProfile(id="t", dietary=["vegan"], allergies=["sesame"])
    strict = loose.model_copy(update={"strict": True})
    ok, warnings = rec.hard_filter(d, loose)
    assert ok and any("vegan" in w for w in warnings) and any("verified" in w for w in warnings)
    assert rec.hard_filter(d, strict) == (False, [])
    verified = d.model_copy(update={"verified": True, "dietary": d.dietary.model_copy(update={"vegan": "yes"})})
    assert rec.hard_filter(verified, strict)[0]


# ----- TasteLayer catalogue (data/tastelayer, synced from tastelayer/) -----

def test_tastelayer_catalogue_loads(tl_cat):
    assert len(tl_cat.restaurants) == 20 and len(tl_cat.dishes) >= 250
    assert tl_cat.flavour.available and len(tl_cat.flavour.dish_vecs) > 200


@pytest.mark.parametrize("dietary,allergies", [
    (["vegan"], ["nuts"]), (["vegetarian"], []), (["halal"], ["gluten"]), ([], ["milk", "egg"]),
])
def test_tastelayer_recommendations_are_safe(tl_cat, dietary, allergies):
    p = UserProfile(id="t", dietary=dietary, allergies=allergies)
    results = rec.recommend(p, tl_cat, "family")
    assert results
    for r in results:
        for d in r.dishes:
            assert not (set(d.allergens) | safety.implied_allergens(d)) & set(allergies)
            for restriction in dietary:
                assert safety.effective_dietary(d, restriction)[0] != "no", (d.name, restriction)


def test_tastelayer_venue_rounds(tl_cat):
    rounds = rec.venue_rounds(UserProfile(id="t", dietary=["vegetarian"]), tl_cat)
    assert len(rounds) >= 3
    for r in rounds:
        assert len({o.restaurant.price_tier for o in r.options}) == 3


# ----- Uber Eats import -----

def zipped(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_reads_only_eats_order_files(monkeypatch):
    data = zipped({
        "Uber Data/Eats/eats_order_details-0.csv": SAMPLE.read_bytes(),
        "Uber Data/Account and Profile/payment_methods-0.csv": b"card,1234",
        "Uber Data/Rider/rider_lifetime_trips-0.csv": b"from,to",
    })
    opened = []
    real = zipfile.ZipFile.read
    monkeypatch.setattr(zipfile.ZipFile, "read", lambda self, name, *a: opened.append(getattr(name, "filename", name)) or real(self, name, *a))
    items = uber_import.read_orders(data, "export.zip")
    assert opened == ["Uber Data/Eats/eats_order_details-0.csv"]
    assert {it.item for it in items} >= {"Margherita Pizza", "Tuna Sashimi"}
    assert "Lamb Vindaloo" not in {it.item for it in items}  # cancelled order skipped
    assert not any("blue door" in f"{it.item} {it.customizations}" for it in items)  # instructions ignored


def test_rejects_zip_without_eats_orders():
    with pytest.raises(ValueError):
        uber_import.read_orders(zipped({"Uber Data/Rider/rider_lifetime_trips-0.csv": b"a,b"}), "x.zip")


def test_import_endpoint_updates_profile():
    pid = client.post("/profile", json={"first_name": "Uber", "dietary": ["vegetarian"]}).json()["id"]
    before = client.get(f"/profile/{pid}").json()
    r = client.post(f"/profile/{pid}/import/uber",
                    files={"file": ("orders.csv", SAMPLE.read_bytes(), "text/csv")})
    assert r.status_code == 200, r.text
    out = r.json()
    s = out["summary"]
    assert s["items"] == 10 and s["restaurants"] == 4  # the cancelled vindaloo is skipped
    assert {"italian", "indian", "japanese"} <= set(s["cuisines"])
    assert "salmon" in s["top_ingredients"] or "salmon" in out["profile"]["ordered_ingredients"]
    assert out["profile"]["taste_vector"] != before["taste_vector"]
    assert "Demo" not in str(s)  # no restaurant names echoed back
    recs = client.post("/recommend", json={"profile_id": pid, "occasion": "comfort"}).json()
    for card in recs:  # imported meat orders never loosen the vegetarian filter
        for d in card["dishes"]:
            assert safety.effective_dietary(Dish(**d), "vegetarian")[0] != "no"
    bad = client.post(f"/profile/{pid}/import/uber", files={"file": ("x.pdf", b"%PDF", "application/pdf")})
    assert bad.status_code == 400
    del PROFILES[pid]


def test_import_moves_flavour_toward_orders():
    p = UserProfile(id="t")
    uber_import.apply_orders(p, uber_import.read_orders(SAMPLE.read_bytes(), "s.csv"), cat)
    vec = rec.epicure_user_vector(p, cat)
    assert vec is not None
    pizza = [d for d in cat.dishes.values() if d.dish_family == "pizza"]
    desserts = [d for d in cat.dishes.values() if d.category == "dessert"]
    mean = lambda ds: sum(cat.flavour.similarity(d.id, vec) for d in ds) / len(ds)
    assert mean(pizza) > mean(desserts)


# ----- frontend-facing additions: menu ranking, profile restore -----

MENU = [
    {"id": "1", "name": "Burrata", "ingredients": "tomato, basil, olive oil", "price": "12"},
    {"id": "2", "name": "Pistachio cannoli", "ingredients": "ricotta, pistachio", "price": "7"},
    {"id": "3", "name": "Chicken tikka", "ingredients": "chicken, yoghurt", "price": "14"},
    {"id": "4", "name": "Paneer butter masala", "ingredients": "paneer, tomato, cream, coriander", "price": "13"},
    {"id": "5", "name": "Mystery special", "ingredients": "", "price": "15"},
]


def test_menu_rank_filters_and_explains():
    pid = client.post("/profile", json={"first_name": "Menu", "dietary": ["vegetarian"], "allergies": ["nuts"],
                                        "disliked_ingredients": ["coriander"]}).json()["id"]
    client.post(f"/profile/{pid}/ratings", json={"ratings": [{"dish_id": "d_0001", "stars": 5}]})  # Margherita Pizza
    r = client.post("/menu/rank", json={"profile_id": pid, "items": MENU})
    assert r.status_code == 200, r.text
    out = {x["name"]: x for x in r.json()}
    assert [x["excluded"] for x in r.json()] == sorted(x["excluded"] for x in r.json())  # excluded items last
    assert out["Pistachio cannoli"]["excluded"] and "nuts" in out["Pistachio cannoli"]["exclusion_reason"]
    assert out["Chicken tikka"]["excluded"] and "vegetarian" in out["Chicken tikka"]["exclusion_reason"]
    assert out["Chicken tikka"]["score"] is None and out["Chicken tikka"]["matches"] == []
    assert out["Burrata"]["because_you_liked"] == "Margherita Pizza"
    assert any("coriander" in w for w in out["Paneer butter masala"]["warnings"])
    assert out["Burrata"]["score"] > out["Paneer butter masala"]["score"]
    assert all(any("vegetarian" in w for w in x["warnings"]) for x in r.json() if not x["excluded"])
    del PROFILES[pid]


def test_menu_rank_uses_tastenet_when_available(monkeypatch):
    from backend import menu_rank

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"taste": {k: 9.0 for k in rec.TASTE_KEYS}}]

    monkeypatch.setattr(menu_rank, "TASTELAYER_URL", "http://tastelayer.test")
    monkeypatch.setattr(menu_rank.httpx, "post", lambda *a, **k: Resp())
    res = menu_rank.rank_menu(UserProfile(id="t"), [menu_rank.MenuItemIn(name="Chicken Tikka Masala")], cat)
    assert res[0].taste_source == "tastenet" and res[0].taste["spice"] == 9.0
    monkeypatch.setattr(menu_rank.httpx, "post", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    res = menu_rank.rank_menu(UserProfile(id="t"), [menu_rank.MenuItemIn(name="Chicken Tikka Masala")], cat)
    assert res[0].taste_source == "estimate"  # service down -> local estimate, no error


def test_restore_profile_after_restart():
    p = client.post("/profile", json={"first_name": "Restore"}).json()
    client.post(f"/profile/{p['id']}/ratings", json={"ratings": [{"dish_id": "d_0001", "stars": 5}]})
    saved = client.get(f"/profile/{p['id']}").json()
    del PROFILES[p["id"]]  # backend restart
    assert client.get(f"/profile/{p['id']}").status_code == 404
    assert client.put(f"/profile/{p['id']}", json=saved).json()["dish_ratings"] == saved["dish_ratings"]
    assert client.put("/profile/u_someone_else", json=saved).status_code == 400
    # resending sliders keeps the confidence ratings already earned
    before = client.get(f"/profile/{p['id']}").json()["taste_confidence"]
    after = client.put(f"/profile/{p['id']}/tolerances", json={k: 6 for k in rec.TASTE_KEYS}).json()["taste_confidence"]
    assert all(after[k] >= before[k] for k in before)
    del PROFILES[p["id"]]


def test_blank_disliked_entry_matches_nothing():
    assert rec.disliked_in(cat.dishes["d_0001"], ["", "  "]) == []
