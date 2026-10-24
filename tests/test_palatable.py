"""The uploaded UI uses one origin and passes only flavour inputs to TasteNet."""
import sys
import types

from fastapi.testclient import TestClient

from backend.palatable import app

client = TestClient(app)
DISH = {'name': 'Chilli paneer', 'cuisine': 'indian', 'ingredients': ['paneer', 'chilli']}


def test_prediction_routes(monkeypatch):
    seen = []
    def predict_batch(rows):
        seen.extend(rows)
        return [{'taste': {'spice': 8}, 'taste_confidence': {'spice': .8},
                 'taste_model_version': 'test-model', 'calibrated': False} for _ in rows]
    monkeypatch.setitem(sys.modules, 'tastelayer.model.predict', types.SimpleNamespace(predict_batch=predict_batch))
    assert client.post('/api/taste', json=DISH).json()['taste']['spice'] == 8
    assert len(client.post('/api/taste/batch', json={'dishes': [DISH, DISH]}).json()) == 2
    assert seen[0] == {**DISH, 'description': ''}


def test_model_unavailable_is_explicit(monkeypatch):
    def unavailable(rows):
        raise RuntimeError('BGE not cached')
    monkeypatch.setitem(sys.modules, 'tastelayer.model.predict', types.SimpleNamespace(predict_batch=unavailable))
    r = client.post('/api/taste', json=DISH)
    assert r.status_code == 503
    assert 'fetch_tastenet.py' in r.json()['detail']


def test_input_limits_and_ui():
    assert client.post('/api/taste', json={**DISH, 'cuisine': 'unknown'}).status_code == 422
    assert client.post('/api/taste/batch', json={'dishes': []}).status_code == 422
    assert client.post('/api/taste/batch', json={'dishes': [DISH] * 101}).status_code == 422
    assert 'tastenet.js' in client.get('/').text
    assert client.get('/tastenet.js').status_code == 200
