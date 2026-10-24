"""Top-5 dishes per synthetic persona (strict and non-strict) -> reports/persona_report.md."""
from tastelayer import config
from tastelayer.reco.recommend import load_catalog, load_personas, recommend

cat = load_catalog()
src = {}
for s in cat.taste_source.values():
    src[s] = src.get(s, 0) + 1
lines = ["# Persona report", "", f"Catalog: {len(cat.dishes)} dishes, {len(cat.restaurants)} restaurants. "
         f"Taste source: {src}. Mock dishes are unverified, so strict restriction personas get no results.", ""]
for pid, p in load_personas().items():
    lines += [f"## {p.persona} (`{pid}`)", ""]
    for strict in (True, False):
        r = recommend(p, k=5, strict=strict, catalog=cat)
        lines.append(f"**{'strict' if strict else 'non-strict'}**: excluded {r['excluded']} {r['excluded_reasons'] or ''}")
        if r["message"]:
            lines += ["", f"> {r['message']}", ""]
            continue
        lines += ["", "| # | dish | restaurant | score | why | warnings |", "|---|---|---|---|---|---|"]
        for i, d in enumerate(r["top_dishes"], 1):
            lines.append(f"| {i} | {d['name']} | {cat.restaurants[d['restaurant_id']].name} | {d['score']:.3f} | "
                         f"{d['explanation']} | {', '.join(d['warnings']) or '-'} |")
        lines.append("")
        if not (p.dietary or p.allergies):
            break  # strict == non-strict when there are no restrictions
    lines.append("")
(config.REPORTS / "persona_report.md").write_text("\n".join(lines))
print("\n".join(lines[:60]))
