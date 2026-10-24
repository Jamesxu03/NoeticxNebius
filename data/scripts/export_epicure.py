"""
Export a slim, pinned copy of an Epicure model for the backend (run once, commit the output).

Writes data/epicure/<name>.npz (raw embeddings + vocab) and <name>.json (provenance), so the
backend needs no network, Hugging Face client or safetensors at runtime.

Usage
-----
    python data/scripts/export_epicure.py                 # chem, the model the backend uses
    python data/scripts/export_epicure.py --model core
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from compare_epicure import MODELS
from epicure import Epicure

OUT_DIR = Path(__file__).resolve().parent.parent / "epicure"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="chem", choices=sorted(MODELS))
    args = p.parse_args()
    repo, rev = MODELS[args.model]

    m = Epicure.from_pretrained(repo, revision=rev)
    vocab = np.array([m.itos[i] for i in range(len(m.itos))])
    OUT_DIR.mkdir(exist_ok=True)
    name = f"epicure-{args.model}"
    np.savez_compressed(OUT_DIR / f"{name}.npz", embeddings=m.E_raw, vocab=vocab)
    meta = {
        "repo": repo,
        "revision": rev,
        "shape": list(m.E_raw.shape),
        "license": "CC BY 4.0",
        "attribution": "Epicure by KAIKAKU.AI (Radzikowski and Chen, arXiv:2605.22391)",
    }
    (OUT_DIR / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {OUT_DIR / name}.npz {m.E_raw.shape}")


if __name__ == "__main__":
    main()
