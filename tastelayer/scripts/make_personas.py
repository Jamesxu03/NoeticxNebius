"""Write data/processed/personas.json: 9 synthetic personas (sliders -> profile via reco.update)."""
import json
from tastelayer import config
from tastelayer.reco.update import new_profile

P = [
    ("u_p01", "vegan spice-lover", dict(spice=9, sweet=3, sour=6, bitter=4, umami=6, richness=4, crunch=5, creaminess=3),
     dict(dietary=["vegan"], price_pref=2.0, vibe_pref={"casual": 0.6, "trendy": 0.4}, liked_ingredients=["green chilli", "chickpea", "tamarind", "coriander"])),
    ("u_p02", "halal adventurous eater", dict(spice=7, sweet=4, sour=6, bitter=5, umami=8, richness=6, crunch=6, creaminess=5),
     dict(dietary=["halal"], price_pref=3.0, vibe_pref={"trendy": 0.6, "traditional": 0.4}, liked_ingredients=["lamb", "cumin", "yoghurt", "fenugreek"])),
    ("u_p03", "nut-allergy cautious eater", dict(spice=2, sweet=5, sour=3, bitter=1, umami=5, richness=5, crunch=3, creaminess=6),
     dict(allergies=["peanuts", "tree nuts"], price_pref=2.0, vibe_pref={"family": 0.7, "casual": 0.3}, liked_ingredients=["chicken", "tomato", "mozzarella"])),
    ("u_p04", "creamy-comfort lover", dict(spice=3, sweet=4, sour=2, bitter=1, umami=6, richness=9, crunch=2, creaminess=9),
     dict(price_pref=2.5, vibe_pref={"casual": 0.5, "traditional": 0.5}, liked_ingredients=["cream", "butter", "parmesan", "paneer"])),
    ("u_p05", "bitter & sour fan", dict(spice=4, sweet=2, sour=8, bitter=8, umami=6, richness=3, crunch=6, creaminess=2),
     dict(price_pref=3.0, vibe_pref={"trendy": 0.7, "date-night": 0.3}, disliked_ingredients=["cream"], liked_ingredients=["rocket", "lemon", "radicchio", "caper"])),
    ("u_p06", "vegetarian date-night", dict(spice=4, sweet=5, sour=4, bitter=3, umami=6, richness=7, crunch=4, creaminess=7),
     dict(dietary=["vegetarian"], price_pref=3.5, vibe_pref={"date-night": 0.8, "trendy": 0.2}, liked_ingredients=["mushroom", "truffle", "ricotta", "spinach"])),
    ("u_p07", "low-spice family", dict(spice=1, sweet=5, sour=3, bitter=1, umami=5, richness=5, crunch=5, creaminess=5),
     dict(price_pref=2.0, vibe_pref={"family": 0.8, "casual": 0.2}, disliked_ingredients=["coriander"], liked_ingredients=["chicken", "potato", "cheese", "tomato"])),
    ("u_p08", "budget student", dict(spice=6, sweet=5, sour=4, bitter=2, umami=7, richness=7, crunch=6, creaminess=5),
     dict(price_pref=1.0, vibe_pref={"quick": 0.6, "casual": 0.4}, liked_ingredients=["pepperoni", "cheese", "chicken", "rice"])),
    ("u_p09", "crunch & savoury snacker", dict(spice=5, sweet=3, sour=4, bitter=2, umami=8, richness=5, crunch=9, creaminess=3),
     dict(price_pref=2.0, vibe_pref={"quick": 0.5, "casual": 0.5}, disliked_ingredients=["mushroom"], liked_ingredients=["potato", "gram flour", "onion", "garlic"])),
]
out = []
for pid, label, sliders, kw in P:
    first = label.split()[0].capitalize()
    p = new_profile(pid, sliders, first_name=first, last_name="Persona", persona=label, **kw)
    out.append(p.model_dump())
(config.DATA_PROCESSED / "personas.json").write_text(json.dumps(out, indent=1))
print(f"wrote {len(out)} personas")
