"""CLI: python -m tastelayer.tagging <command>"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from tastelayer import config

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("tag-dishes")
def tag_dishes_cmd(
    limit: Optional[int] = typer.Option(None, help="only the first N dishes (menu dishes first)"),
    uids_file: Optional[Path] = typer.Option(None, help="whitespace-separated uids to tag"),
    menu_only: bool = typer.Option(False, help="only data/processed/menu_dishes.json"),
    corpus: Path = typer.Option(config.DATA_PROCESSED / "corpus.parquet"),
    concurrency: int = typer.Option(64, help="max in-flight requests per model"),
):
    from tastelayer.tagging.tag import tag_dishes

    raise typer.Exit(tag_dishes(limit, uids_file, menu_only, concurrency=concurrency, corpus=corpus))


@app.command("tag-gaps")
def tag_gaps_cmd(
    models: str = typer.Option("Qwen/Qwen3-235B-A22B-Instruct-2507,openai/gpt-oss-120b", help="comma list"),
    minutes: Optional[float] = typer.Option(None, help="wall-clock cutoff; stops new requests cleanly"),
    concurrency: int = typer.Option(64, help="max in-flight requests per model"),
):
    """Fill missing (dish, model, temperature) labels: menu, then high-disagreement, then near-complete dishes."""
    from tastelayer.tagging.tag import tag_gaps

    raise typer.Exit(tag_gaps([x.strip() for x in models.split(",") if x.strip()], minutes, concurrency))


@app.command("tag-ingredients")
def tag_ingredients_cmd(
    limit: Optional[int] = None,
    concurrency: int = 64,
    vocab: Path = typer.Option(config.ARTIFACTS / "ingredient_vocab.json"),
):
    from tastelayer.tagging.tag import tag_ingredients

    raise typer.Exit(tag_ingredients(limit, concurrency=concurrency, vocab_path=vocab))


@app.command("second-pass")
def second_pass_cmd(corpus: Path = typer.Option(config.DATA_PROCESSED / "corpus.parquet"), concurrency: int = 64):
    from tastelayer.tagging.tag import second_pass

    raise typer.Exit(second_pass(concurrency, corpus=corpus))


@app.command()
def consensus():
    from tastelayer.tagging.consensus import run

    run()


@app.command()
def report(watch: int = typer.Option(0, help="rewrite every N seconds (e.g. 600)")):
    from tastelayer.tagging import report as r

    if watch:
        r.watch(watch)
    else:
        print(r.write_summary())


if __name__ == "__main__":
    app()
