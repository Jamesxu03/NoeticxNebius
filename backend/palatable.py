"""Serve the uploaded Palatable UI and the real, local TasteNet model on one origin."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tastelayer' / 'src'))
app = FastAPI(title='Palatable + TasteNet')
_model_lock = Lock()  # The model's on-disk embedding cache is append-only, not concurrent.
log = logging.getLogger(__name__)


class DishInput(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    cuisine: Literal['indian', 'italian']
    description: str = Field(default='', max_length=2000)
    ingredients: list[str] = Field(default_factory=list, max_length=100)


class BatchInput(BaseModel):
    dishes: list[DishInput] = Field(min_length=1, max_length=100)


def predict(dishes: list[DishInput]):
    try:
        from tastelayer.model.predict import predict_batch
        with _model_lock:
            return predict_batch([dish.model_dump() for dish in dishes])
    except (FileNotFoundError, RuntimeError, OSError, ImportError) as exc:
        log.warning('TasteNet unavailable: %s', exc)
        raise HTTPException(503, 'TasteNet is unavailable. Install the inference dependencies and run scripts/fetch_tastenet.py to cache the BGE model; then retry.') from exc


@app.post('/api/taste')
def taste(dish: DishInput):
    return predict([dish])[0]


@app.post('/api/taste/batch')
def taste_batch(batch: BatchInput):
    return predict(batch.dishes)


app.mount('/', StaticFiles(directory=ROOT / 'uploaded-ui', html=True), name='palatable')
