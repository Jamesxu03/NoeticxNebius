"""Central config: paths, seeds, model list, prices. Secrets come from .env only."""
from __future__ import annotations

import os
import random
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
CACHE_DIR = ROOT / "cache"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
GOLD = ROOT / "gold"

SEED = int(os.getenv("SEED", "42"))
MAX_USD = float(os.getenv("MAX_USD", "60"))
ASK_USD = 40.0  # pause and ask the human before total spend passes this

NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")


def nebius_api_key() -> str:
    key = os.getenv("NEBIUS_API_KEY", "")
    if not key:
        raise RuntimeError("NEBIUS_API_KEY missing from .env")
    return key


def anthropic_enabled() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY", "").strip())


# Candidate tagging models (skip any not available on the account).
CANDIDATE_MODELS = [
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "openai/gpt-oss-120b",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "nvidia/Nemotron-3_5-Lightning",
    "google/gemma-3-27b-it",
]
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

# Models actually used for tagging (Nemotron dropped: truncated JSON; gemma: 503s).
TAGGING_MODELS = [
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
    "openai/gpt-oss-120b",
    "deepseek-ai/DeepSeek-V4-Flash-0731",
]
# Isotonic calibration on gold is evaluated (cross-fitted column) but NEVER served unless this is
# flipped deliberately. Decision 2026-10-24: gold is 28 dishes from one labeller, so keep raw TasteNet.
SERVE_CALIBRATION = False

SECOND_PASS_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
TAG_TEMPERATURES = [0.2, 0.7]  # 2 samples per dish per model
DISHES_PER_REQUEST = 10
INGREDIENTS_PER_REQUEST = 20
N_INGREDIENTS_TO_TAG = 3000
MODEL_SHORT = {
    "Qwen/Qwen3-235B-A22B-Instruct-2507": "qwen3-235b",
    "openai/gpt-oss-120b": "gpt-oss-120b",
    "deepseek-ai/DeepSeek-V4-Flash-0731": "deepseek-v4-flash",
}

# $ per 1M tokens (input, output). Confirm against Nebius pricing page.
PRICES = {
    "Qwen/Qwen3-235B-A22B-Instruct-2507": (0.20, 0.60),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "deepseek-ai/DeepSeek-V4-Flash-0731": (0.14, 0.28),
    "nvidia/Nemotron-3_5-Lightning": (0.06, 0.24),
    "google/gemma-3-27b-it": (0.10, 0.30),
    "Qwen/Qwen3-Embedding-8B": (0.01, 0.0),
}


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
