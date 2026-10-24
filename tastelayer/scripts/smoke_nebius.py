"""Step 1 smoke test: list Nebius models, test strict json_schema vs json_object, show rate-limit headers."""
from __future__ import annotations

import json
import sys
import time

from openai import OpenAI
from pydantic import ValidationError

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))
from tastelayer import config  # noqa: E402
from tastelayer.schema import TagBatch  # noqa: E402

SAMPLE = [
    ("s01", "indian", "Chicken Tikka Masala", "Chargrilled chicken in a creamy spiced tomato sauce"),
    ("s02", "indian", "Lamb Vindaloo", "Fiery Goan curry with vinegar, chilli and potato"),
    ("s03", "indian", "Vegetable Samosa", "Crisp fried pastry filled with spiced potato and peas"),
    ("s04", "indian", "Mango Lassi", "Chilled yoghurt drink blended with alphonso mango"),
    ("s05", "indian", "Tarka Dal", "Yellow lentils tempered with garlic, cumin and ghee"),
    ("s06", "italian", "Margherita Pizza", "San Marzano tomato, fior di latte, basil"),
    ("s07", "italian", "Spaghetti Carbonara", "Guanciale, egg yolk, pecorino, black pepper"),
    ("s08", "italian", "Tiramisu", "Savoiardi soaked in espresso with mascarpone and cocoa"),
    ("s09", "italian", "Arancini", "Fried risotto balls with mozzarella centre"),
    ("s10", "italian", "Penne Arrabbiata", "Tomato sauce with garlic and dried chilli"),
]
SYSTEM = (
    "You are a professional chef and sensory panel lead. Rate each dish as served on 8 independent 0-10 "
    "integer scales: spice, sweet, sour, bitter, umami, richness, crunch, creaminess. Return only JSON."
)
USER = "Dishes:\n" + json.dumps([dict(uid=u, cuisine=c, name=n, description=d) for u, c, n, d in SAMPLE])
UIDS = {s[0] for s in SAMPLE}


def strict_schema() -> dict:
    """Pydantic schema inlined (no $refs) and made strict-friendly."""
    s = TagBatch.model_json_schema()
    defs = s.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].split("/")[-1]])
            out = {k: resolve(v) for k, v in node.items() if k != "title"}
            if out.get("type") == "object":
                out["additionalProperties"] = False
                out["required"] = list(out.get("properties", {}))
            return out
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    return resolve(s)


def check(content: str) -> str:
    try:
        b = TagBatch.model_validate_json(content)
    except ValidationError as e:
        return f"INVALID ({e.error_count()} errors)"
    except Exception as e:  # noqa: BLE001
        return f"INVALID ({type(e).__name__})"
    got = {i.uid for i in b.items}
    return "valid, uids match" if got == UIDS else f"valid but uid mismatch ({len(got)}/10)"


def call(client: OpenAI, model: str, mode: str) -> dict:
    if mode == "json_schema":
        rf = {"type": "json_schema", "json_schema": {"name": "taste_batch", "schema": strict_schema(), "strict": True}}
        user = USER
    else:
        rf = {"type": "json_object"}
        user = USER + "\n\nReturn JSON matching this schema:\n" + json.dumps(strict_schema())
    t0 = time.time()
    try:
        raw = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            response_format=rf,
            temperature=0.2,
            max_tokens=2500,
            seed=config.SEED,
        )
    except Exception as e:  # noqa: BLE001
        return {"mode": mode, "ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}"}
    resp = raw.parse()
    rl = {k: v for k, v in raw.headers.items() if "ratelimit" in k.lower() or k.lower() == "retry-after"}
    content = resp.choices[0].message.content or ""
    u = resp.usage
    pin, pout = config.PRICES.get(model, (0, 0))
    cost = (u.prompt_tokens * pin + u.completion_tokens * pout) / 1e6 if u else 0
    return {
        "mode": mode,
        "ok": True,
        "result": check(content),
        "latency_s": round(time.time() - t0, 1),
        "tokens": f"{u.prompt_tokens}/{u.completion_tokens}" if u else "?",
        "cost_usd": round(cost, 5),
        "ratelimit_headers": rl,
    }


def main() -> None:
    client = OpenAI(base_url=config.NEBIUS_BASE_URL, api_key=config.nebius_api_key(), timeout=120)
    available = sorted(m.id for m in client.models.list().data)
    print(f"Account has {len(available)} models.\n")
    wanted = config.CANDIDATE_MODELS + [config.EMBEDDING_MODEL]
    for w in wanted:
        print(f"  {'YES' if w in available else 'no '}  {w}")
    missing = [w for w in wanted if w not in available]
    if missing:
        print("\nClosest available alternatives:")
        for w in missing:
            stem = w.split("/")[-1].split("-")[0].lower()
            alts = [a for a in available if stem in a.lower()]
            print(f"  {w} -> {alts[:6] or '(none)'}")
    (config.REPORTS / "nebius_models.json").write_text(json.dumps(available, indent=1))

    results = {}
    for m in [w for w in config.CANDIDATE_MODELS if w in available]:
        results[m] = [call(client, m, "json_schema"), call(client, m, "json_object")]
        for r in results[m]:
            print(f"\n{m} [{r['mode']}]: {json.dumps(r)}")
    (config.REPORTS / "smoke_test.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
