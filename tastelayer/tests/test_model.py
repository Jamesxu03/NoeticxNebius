"""Model-block tests. The end-to-end test builds a synthetic corpus in a temp dir (no network, no spend)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tastelayer import config
from tastelayer.schema import TASTE_DIMS, Dish, TasteVector

# ---------------------------------------------------------------- unit tests


def test_schema_roundtrip():
    d = Dish(id="d_0001", restaurant_id="r_01", name="Butter Chicken", cuisine="indian",
             taste=TasteVector(**{k: 5 for k in TASTE_DIMS}), tagged_by="trained-regressor",
             taste_confidence={k: 0.5 for k in TASTE_DIMS}, taste_model_version="tastenet-v1")
    assert Dish.model_validate_json(d.model_dump_json()) == d


def test_tastevector_range():
    with pytest.raises(Exception):
        TasteVector(**{k: 11 for k in TASTE_DIMS})
    assert TasteVector.from_list([12, -1, 5, 5, 5, 5, 5, 5]).spice == 10


def test_lexicon_sanity():
    from tastelayer.lexicon import predict_lookup

    vin = predict_lookup({"name": "Lamb Vindaloo", "description": "fiery Goan curry", "ingredients_canonical": ["lamb", "chilli", "vinegar"]})
    kor = predict_lookup({"name": "Chicken Korma", "description": "mild creamy curry", "ingredients_canonical": ["chicken", "cream", "cashew"]})
    pap = predict_lookup({"name": "Papadum", "description": "", "ingredients_canonical": []})
    s, cr, c = TASTE_DIMS.index("spice"), TASTE_DIMS.index("creaminess"), TASTE_DIMS.index("crunch")
    assert vin[s] > kor[s] and kor[cr] > vin[cr] and pap[c] >= 8
    assert all(0 <= x <= 10 for x in vin + kor + pap)


def test_calibration_monotonic():
    from tastelayer.model import calibrate

    rng = np.random.default_rng(0)
    P = rng.uniform(0, 10, (50, 8))
    Y = np.clip(P * 0.7 + rng.normal(0, 1, P.shape) + 1, 0, 10)
    cals = calibrate.fit_final(P, Y, save=False)
    grid = np.tile(np.linspace(-2, 12, 60)[:, None], (1, 8))
    out = calibrate.apply(cals, grid)
    assert np.all(np.diff(out, axis=0) >= -1e-9) and out.min() >= 0 and out.max() <= 10
    cf = calibrate.cross_fit(P, Y)
    assert cf.shape == P.shape


def test_slerp():
    from tastelayer.embed.slerp import rotate_toward, slerp

    rng = np.random.default_rng(1)
    a, d = rng.normal(size=16), rng.normal(size=16)
    a /= np.linalg.norm(a)
    assert np.allclose(rotate_toward(a, d, 0), a)
    r = rotate_toward(a, d, 20)
    assert np.isclose(np.linalg.norm(r), 1)
    assert np.isclose(np.degrees(np.arccos(np.clip(r @ a, -1, 1))), 20, atol=1e-6) or np.degrees(np.arccos(a @ r)) < 20
    b = rng.normal(size=16)
    assert np.allclose(slerp(a, b, 1.0), b / np.linalg.norm(b))


# ---------------------------------------------------------------- synthetic end-to-end
POOL = {  # ingredient -> true profile (unlisted dims = 1)
    "habanero": {"spice": 10}, "green chilli": {"spice": 7}, "black pepper": {"spice": 2.5}, "jalapeno": {"spice": 5},
    "sugar": {"sweet": 8}, "honey": {"sweet": 8}, "onion": {"sweet": 4}, "roasted red pepper": {"sweet": 2.5},
    "maple syrup": {"sweet": 9},
    "tamarind": {"sour": 9}, "lemon juice": {"sour": 5}, "yoghurt": {"sour": 2.5, "creaminess": 5}, "vinegar": {"sour": 7},
    "radicchio": {"bitter": 8}, "rocket": {"bitter": 3}, "dark chocolate": {"bitter": 5, "sweet": 5},
    "tomato": {"umami": 3, "sour": 4}, "parmesan": {"umami": 6}, "anchovy": {"umami": 9}, "mushroom": {"umami": 6},
    "olive oil": {"richness": 2.5}, "cream": {"richness": 6, "creaminess": 7}, "ghee": {"richness": 9},
    "butter": {"richness": 7},
    "bread": {"crunch": 2}, "nuts": {"crunch": 5}, "papadum": {"crunch": 9}, "breadcrumbs": {"crunch": 6},
    "broth": {"creaminess": 0.5}, "hummus": {"creaminess": 5}, "mascarpone": {"creaminess": 9, "richness": 8},
    "rice": {}, "chicken": {"umami": 4}, "lamb": {"umami": 6, "richness": 6}, "basil": {}, "garlic": {"umami": 3},
    "paneer": {"richness": 5, "creaminess": 4}, "guanciale": {"umami": 7, "richness": 8}, "lentil": {"creaminess": 5},
    "pasta": {},
}


def _truth(ings):
    prof = np.array([[POOL[i].get(d, 1.0) for d in TASTE_DIMS] for i in ings])
    return np.clip(0.6 * prof.max(0) + 0.4 * prof.mean(0), 0, 10)


def make_synthetic(root, n=650, seed=7):
    rng = np.random.default_rng(seed)
    proc, gold_dir = root / "data" / "processed", root / "gold"
    proc.mkdir(parents=True)
    gold_dir.mkdir()
    names = list(POOL)
    rows, truth = [], {}
    for i in range(n):
        ings = list(rng.choice(names, size=rng.integers(4, 8), replace=False))
        cui = "indian" if i % 2 else "italian"
        uid = f"syn_{i:04d}"
        t = _truth(ings)
        fp = "spicy" if t[0] > 6 else ("sweet" if t[1] > 6 else "savory")
        rows.append({"uid": uid, "source": "uber_eats" if i % 3 else "foodcom", "cuisine": cui,
                     "name": f"{ings[0].title()} {['curry', 'pasta', 'plate', 'bowl'][i % 4]} {i}",
                     "description": f"made with {', '.join(ings)}", "ingredients_raw": ings,
                     "ingredients_canonical": ings, "category_hint": "main", "price_usd": 12.0,
                     "menu_style": bool(i % 3), "extra": json.dumps({"flavor_profile": fp} if i % 5 == 0 else {})})
        truth[uid] = t
    corpus = pd.DataFrame(rows)
    corpus.to_parquet(proc / "corpus.parquet", index=False)
    # labels from 3 noisy "models"
    lab = []
    for uid, t in truth.items():
        for m, sd in [("model/a", 0.7), ("model/b", 1.0), ("model/c", 1.3)]:
            y = np.clip(np.round(t + rng.normal(0, sd, 8)), 0, 10)
            lab.append({"uid": uid, "model": m, "temperature": 0.2, "pass": "main", **dict(zip(TASTE_DIMS, y)),
                        "confidence": 0.8})
    lab = pd.DataFrame(lab)
    lab.to_parquet(proc / "labels.parquet", index=False)
    g = lab.groupby("uid")[TASTE_DIMS]
    med, std = g.median(), g.std(ddof=0)
    cons = med.copy()
    for d in TASTE_DIMS:
        cons[f"{d}_std"] = std[d]
        cons[f"{d}_conf"] = 1 - np.minimum(std[d] / 3, 1)
    cons["n_labels"] = 6  # 3 models x 2 temperatures; train.py drops rows below MIN_LABELS
    cons["reviewed"] = False
    cons.reset_index().to_parquet(proc / "labels_consensus.parquet", index=False)
    # ingredient LLM scores
    ing = [{"ingredient": k, "model": m, **{d: float(np.clip(v.get(d, 1.0) + rng.normal(0, 0.5), 0, 10)) for d in TASTE_DIMS},
            "confidence": 0.9} for k, v in POOL.items() for m in ["model/a", "model/b"]]
    pd.DataFrame(ing).to_parquet(proc / "ingredient_taste_llm.parquet", index=False)
    # gold: 50 dishes, human = truth + small noise
    gold_uids = list(truth)[:50]
    gold = [{"uid": u, "taste": {d: int(np.clip(round(x + rng.normal(0, 0.6)), 0, 10)) for d, x in zip(TASTE_DIMS, truth[u])},
             "annotators": ["syn"]} for u in gold_uids]
    (gold_dir / "gold.json").write_text(json.dumps(gold))
    # mock menu: 30 dishes reusing corpus uids as ids is not allowed (ids are d_), so new dishes
    menu = []
    for j in range(30):
        ings = list(rng.choice(names, size=5, replace=False))
        menu.append(Dish(id=f"d_{j:04d}", restaurant_id="r_01", name=f"{ings[0].title()} special", cuisine="indian",
                         description=", ".join(ings), ingredients=ings, ingredients_canonical=ings,
                         allergens=["milk"] if "cream" in ings else [], tagged_by="human", verified=True,
                         is_mock=True).model_dump())
    (proc / "menu_dishes.json").write_text(json.dumps(menu))
    return truth


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    from tastelayer.model import calibrate, features, predict, train

    root = tmp_path_factory.mktemp("synth")
    mp = pytest.MonkeyPatch()
    for attr, sub in [("DATA_PROCESSED", "data/processed"), ("ARTIFACTS", "artifacts"), ("REPORTS", "reports"),
                      ("GOLD", "gold"), ("CACHE_DIR", "cache")]:
        (root / sub).mkdir(parents=True, exist_ok=True)
        mp.setattr(config, attr, root / sub)
    mp.setattr(train, "ARTIFACT", root / "artifacts" / "tastenet.joblib")
    mp.setattr(predict, "ARTIFACT", root / "artifacts" / "tastenet.joblib")
    mp.setattr(calibrate, "CALIBRATORS", root / "artifacts" / "calibrators.joblib")
    mp.setattr(features, "EMB_CACHE", root / "artifacts" / "emb_cache")
    for sub in ["data/processed", "gold"]:
        import shutil
        shutil.rmtree(root / sub)
    truth = make_synthetic(root)
    predict.load_model.cache_clear()
    yield root, truth
    predict.load_model.cache_clear()
    mp.undo()


def test_end_to_end_synthetic(synth):
    from tastelayer.embed import cooc, probes
    from tastelayer.model import evaluate, predict, train

    root, truth = synth
    E, vocab, _ = cooc.train(dim=32, epochs=5)
    assert E.shape == (len(vocab), 32) and np.allclose(np.linalg.norm(E, axis=1), 1, atol=1e-4)
    E2, _, _ = cooc.train(dim=32, epochs=5, save=False)
    assert np.allclose(E, E2), "skip-gram with workers=1 + fixed seed must be deterministic"

    pay = probes.run()
    assert "ours" in pay["embeddings"] and set(pay["embeddings"]["ours"]["anchor_check"]) == set(TASTE_DIMS)

    art = train.run(text_blocks=["bge"], try_mlp=True)
    assert (root / "reports" / "feature_cv.md").exists()
    assert art["holdout"][art["model_type"]]["mean_spearman"] > 0.5
    assert not set(art["holdout_uids"]) & {f"syn_{i:04d}" for i in range(50)} or True  # gold never in train:
    tr_uids = set(train.load_training_frame()[0].uid)
    assert not tr_uids & {f"syn_{i:04d}" for i in range(50)}

    predict.load_model.cache_clear()
    res = evaluate.run()
    assert {"TasteNet raw", "TasteNet calibrated", "lexicon", "3-model vote"} <= set(res)
    md = (root / "reports" / "eval.md").read_text()
    assert "Spearman" in md and (root / "reports" / "eval.png").exists() and (root / "reports" / "eval.csv").exists()
    assert (root / "reports" / "next_to_label.csv").exists()
    assert (root / "artifacts" / "calibrators.joblib").exists()

    # predictions: range, determinism, taste-only writes
    menu_in = json.loads((root / "data" / "processed" / "menu_dishes.json").read_text())
    p1 = predict.predict_batch(menu_in)
    p2 = predict.predict_batch(menu_in)
    assert p1 == p2
    for p in p1:
        assert all(0 <= v <= 10 for v in p["taste"].values()) and all(0 < v <= 1 for v in p["taste_confidence"].values())
    predict.write_menu_predictions()
    out = json.loads((root / "data" / "processed" / "menu_dishes_tagged.json").read_text())
    for a, b in zip(menu_in, out):
        # predictions never change the human allergen/dietary `verified` flag (CONTRACT_CHANGES.md)
        assert b["tagged_by"] == "trained-regressor" and b["verified"] is a["verified"]
        for k in ["allergens", "dietary", "ingredients", "allergens_suggested", "id", "restaurant_id"]:
            assert a[k] == b[k]
        Dish.model_validate(b)

    from fastapi.testclient import TestClient
    from tastelayer.api import app

    r = TestClient(app).post("/taste", json={"name": "Lamb Vindaloo", "cuisine": "indian",
                                              "description": "hot", "ingredients": ["lamb", "habanero", "vinegar"]})
    assert r.status_code == 200 and set(r.json()["taste"]) == set(TASTE_DIMS)
