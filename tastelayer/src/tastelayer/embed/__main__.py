"""`python -m tastelayer.embed [--core] [--dim 128] [--workers 1]`: mini-Epicure embeddings, FlavorGraph mapping, probes."""
from __future__ import annotations

import typer

from .. import config

app = typer.Typer(add_completion=False)


@app.command()
def main(dim: int = 128, core: bool = typer.Option(False, help="add NPMI-graph random walks"),
         workers: int = typer.Option(1, help=">1 is faster but not bit-reproducible"),
         flavorgraph: bool = True, probes: bool = True):
    from . import cooc, flavorgraph as fg, probes as pr

    config.seed_everything()
    E, vocab, _ = cooc.train(dim=dim, core=core, workers=workers)
    for w in pr.SANITY_WORDS:
        print(f"  {w}: {cooc.neighbours(E, vocab, w, 6) or '(not in vocab)'}")
    if flavorgraph:
        fg.build(vocab)
    if probes:
        try:
            pr.run()
        except RuntimeError as e:
            print(f"[probes] skipped: {e}")


if __name__ == "__main__":
    app()
