"""The served TasteNet must predict a never-seen dish with no network access at all."""
import socket
import uuid

import pytest

from tastelayer import config


@pytest.fixture
def no_network(monkeypatch):
    attempts = []

    def connect(self, addr, *a, **k):
        attempts.append(addr)
        raise OSError(f"network blocked: {addr}")

    real_gai = socket.getaddrinfo

    def gai(host, *a, **k):
        if host not in ("localhost", "127.0.0.1", "::1", None):
            attempts.append(("dns", host))
            raise OSError(f"dns blocked: {host}")
        return real_gai(host, *a, **k)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", gai)
    return attempts


@pytest.mark.skipif(not (config.ARTIFACTS / "tastenet.joblib").exists(), reason="no trained TasteNet")
def test_served_tastenet_predicts_new_dish_offline(no_network):
    import joblib

    from tastelayer.model import features, predict

    art = joblib.load(config.ARTIFACTS / "tastenet.joblib")
    assert "qwen" not in art["feature_set"], "served TasteNet must not need the Qwen embedding API"
    try:
        features.load_bge()
    except RuntimeError:
        pytest.skip("bge not cached locally (run make fetch-models)")
    features._BGE = None  # force a fresh load under the network block
    dish = {"id": "d_offline_test", "restaurant_id": "r_99", "cuisine": "italian",
            "name": "Smoked Aubergine & Nduja Arancini (offline test)",
            # unique text every run: guarantees an embedding-cache miss, so bge really loads and encodes
            "description": f"crisp fried risotto balls with smoky aubergine, spicy nduja and lemon aioli {uuid.uuid4().hex}",
            "ingredients": ["arborio rice", "aubergine", "nduja", "lemon", "mayonnaise", "parmesan"]}
    p = predict.predict_taste(dish)
    assert no_network == [], f"network attempted: {no_network}"
    assert set(p["taste"]) == {"spice", "sweet", "sour", "bitter", "umami", "richness", "crunch", "creaminess"}
    assert all(0 <= v <= 10 for v in p["taste"].values())
