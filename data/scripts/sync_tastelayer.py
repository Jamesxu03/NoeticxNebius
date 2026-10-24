"""
Copy the TasteLayer catalogue (tastelayer/data/processed) into a backend data folder.

TasteLayer (Role B's taste model) writes `menu_dishes_tagged.json` and `restaurants_mock.json`.
The backend reads `<DATA_DIR>/dishes.json` and `<DATA_DIR>/restaurants.json`. This script
validates both files against schema/models.py and writes them in that layout:

    python data/scripts/sync_tastelayer.py                  # -> data/tastelayer/
    DATA_DIR=data/tastelayer uvicorn backend.main:app       # serve it

Dietary flags, allergens and `verified` are copied through untouched: the taste model never
changes them, and neither does this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from schema.models import Dish, Restaurant  # noqa: E402

SRC = ROOT / "tastelayer/data/processed"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dishes", default=str(SRC / "menu_dishes_tagged.json"))
    p.add_argument("--restaurants", default=str(SRC / "restaurants_mock.json"))
    p.add_argument("--out", default=str(ROOT / "data/tastelayer"))
    args = p.parse_args()

    dishes = [Dish(**d).model_dump() for d in json.loads(Path(args.dishes).read_text())]
    restaurants = [Restaurant(**r).model_dump() for r in json.loads(Path(args.restaurants).read_text())]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "dishes.json").write_text(json.dumps(dishes, indent=1, ensure_ascii=False) + "\n")
    (out / "restaurants.json").write_text(json.dumps(restaurants, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {len(restaurants)} restaurants, {len(dishes)} dishes -> {out}")
    print(f"check it: python -m schema.validate {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
