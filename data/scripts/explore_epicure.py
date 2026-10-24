"""
Starter script for the Epicure ingredient embeddings.

Examples
--------
    python data/scripts/explore_epicure.py neighbors chicken
    python data/scripts/explore_epicure.py neighbors miso -k 10 --model Kaikaku/epicure-chem
    python data/scripts/explore_epicure.py slerp rice cuisine:South_Asian --theta 30
    python data/scripts/explore_epicure.py mode chocolate --kind factor
    python data/scripts/explore_epicure.py poles --prefix cuisine
    python data/scripts/explore_epicure.py search choc
"""

from __future__ import annotations

import argparse
import difflib
import sys

from epicure import Epicure

MODELS = ("Kaikaku/epicure-core", "Kaikaku/epicure-cooc", "Kaikaku/epicure-chem")


def check_ingredient(m: Epicure, name: str) -> None:
    if name in m.vocab:
        return
    close = difflib.get_close_matches(name, m.vocab.keys(), n=5)
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    sys.exit(f"Unknown ingredient {name!r}.{hint}")


def print_scores(rows: list[tuple[str, float]]) -> None:
    for name, score in rows:
        print(f"  {score:6.3f}  {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Epicure ingredient embeddings.")
    parser.add_argument("--model", default=MODELS[0],
                        help=f"Hugging Face repo id or local folder (default {MODELS[0]})")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("neighbors", help="ingredients closest in flavour space")
    p.add_argument("ingredient")
    p.add_argument("-k", type=int, default=5)

    p = sub.add_parser("slerp", help="rotate an ingredient toward a pole, e.g. a cuisine")
    p.add_argument("ingredient")
    p.add_argument("pole", help="supervised pole key, see `poles`")
    p.add_argument("--theta", type=float, default=30.0, help="angle in degrees")
    p.add_argument("-k", type=int, default=5)

    p = sub.add_parser("mode", help="closest flavour modes for an ingredient")
    p.add_argument("ingredient")
    p.add_argument("--kind", choices=["factor", "cuisine", "food_group", "nova_level",
                                      "cf_sensory", "usda_nutrient"])
    p.add_argument("-k", type=int, default=3)

    p = sub.add_parser("poles", help="list supervised poles usable with `slerp`")
    p.add_argument("--prefix")

    p = sub.add_parser("search", help="find ingredient names containing text")
    p.add_argument("text")

    args = parser.parse_args()
    m = Epicure.from_pretrained(args.model)

    if args.cmd == "neighbors":
        check_ingredient(m, args.ingredient)
        print_scores(m.neighbors(args.ingredient, k=args.k))
    elif args.cmd == "slerp":
        check_ingredient(m, args.ingredient)
        if args.pole not in m.supervised_poles:
            sys.exit(f"Unknown pole {args.pole!r}. Run `python data/scripts/explore_epicure.py poles` to list them.")
        print_scores(m.slerp(args.ingredient, args.pole, theta_deg=args.theta, k=args.k))
    elif args.cmd == "mode":
        check_ingredient(m, args.ingredient)
        for mode_id, label, score in m.closest_mode(args.ingredient, kind=args.kind, k=args.k):
            print(f"  {score:6.3f}  {mode_id:<12} {label}")
    elif args.cmd == "poles":
        for key in m.list_supervised_poles(args.prefix):
            print(f"  {key}")
    elif args.cmd == "search":
        for name in sorted(n for n in m.vocab if args.text.lower() in n.lower()):
            print(f"  {name}")


if __name__ == "__main__":
    main()
