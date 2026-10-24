"""Build data/processed/corpus.parquet (+ mock menu) from Kaggle sources.

Sources (data/raw/<name>/): uber_eats, foodcom, archana, indian_food_101, whats_cooking (optional).
Run: python -m tastelayer.data.ingest
Idempotent: rebuilds outputs from raw files each run; downloads are handled by scripts/download.sh.
"""
from __future__ import annotations

import ast
import html
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.process import cdist

from tastelayer import config
from tastelayer.canon import canonicalize, canonicalize_list, singularize, synonyms
from tastelayer.schema import Dish, Restaurant

RAW = config.DATA_RAW
OUT = config.DATA_PROCESSED
SEED = config.SEED

UE_CAP_INDIAN = 40      # max items per Indian restaurant
UE_CAP_ITALIAN = 12     # max items per Italian/pizza restaurant (there are ~4x more of them)
UE_MAX_ITEMS = 60_000   # overall cap on Uber Eats rows before dedupe
VOCAB_MIN_FREQ = 5
USD_TO_GBP = 0.79

# ---------- text helpers ----------

def clean_text(s) -> str:
    if not isinstance(s, str):
        return ""
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_name(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\brecipe\b", " ", s)
    s = re.sub(r"\([^)]*\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_price(p) -> float | None:
    if not isinstance(p, str):
        return None
    m = re.search(r"[\d.]+", p.replace(",", ""))
    try:
        v = float(m.group()) if m else None
    except ValueError:
        return None
    return v if v and 0 < v < 500 else None


# ---------- filters for Uber Eats menu items ----------

DROP_NAME = re.compile(
    r"\b(?:coke|coca[- ]cola|pepsi|sprite|fanta|soda|pop|water|juice|lemonade|iced tea|sweet tea|coffee|espresso|"
    r"latte|cappuccino|beer|wine|bottle|can of|napkin|utensil|fork|spoon|plate|cutlery|straw|gift card|"
    r"red bull|snapple|gatorade|ginger ale|seltzer|tip|donation|bag|box of|catering|tray|party pack|"
    r"family meal for|combo for|ice cubes?)\b",
    re.I,
)
DROP_PREFIX = re.compile(r"^\s*(?:side of|extra|add|add-on|substitute|sub |choice of|build your own)\b", re.I)
DROP_CONDIMENT = re.compile(r"\b(?:dip|dressing|sauce|ketchup|ranch|mayo|marinara cup|dipping|gravy only|chutney)\s*$", re.I)
DROP_SECTION = re.compile(r"(?:beverage|drink|soda|sauce|dressing|extras?\b|add[- ]?ons?|utensil|catering|party|"
                          r"alcohol|beer|wine|cocktail|gift|merch|kids? menu)", re.I)
NON_ITALIAN = re.compile(r"\b(?:wings?|burgers?|hoagies?|philly|cheesesteaks?|tacos?|burritos?|quesadillas?|nachos|fries|fajitas?|bbq|buffalo|"
                         r"tenders?|nuggets?|hot dog|gyro|falafel|shawarma|hummus|club sandwich|blt|wrap)\b", re.I)
NON_INDIAN = re.compile(r"\b(?:burgers?|fries|wings?|hot dogs?|tacos?|burritos?|pizzas?|chow mein|lo mein|hakka|manchurian|gyros?|"
                        r"falafel|shawarma|baklava|pita|quesadillas?|brisket|noodles?|"
                        r"fried rice|szechuan|schezwan|chilli chicken|momo)\b", re.I)


# ---------- dish family / category ----------

FAMILY_RULES: list[tuple[str, str]] = [
    ("biryani", r"biry?ani|biriyani|pulao|pilau"),
    ("pizza", r"pizza|margherita|margarita pie|\bpie\b|calzone|stromboli"),
    ("pasta", r"pasta|spaghetti|penne|fettuc+ine|linguine|rigatoni|lasagn|ravioli|gnocchi|tortellini|tagliatelle|"
              r"pappardelle|carbonara|bolognese|alfredo|ziti|orecchiette|bucatini|macaroni|manicotti|cannelloni|"
              r"farfalle|fusilli|cavatelli|vongole|puttanesca|amatriciana|cacio e pepe|arrabbiata|primavera"),
    ("risotto", r"risotto"),
    ("curry", r"tikka masala|butter chicken|makhani|korma|vindaloo|madras|jalfrezi|curry|kadai|karahi|kadhai|"
              r"rogan josh|saag|palak|bhuna|dopiaza|pasanda|chettinad|kofta|do pyaza|masala\b|keema|chana|chole|"
              r"paneer|malai|mughlai|nihari|haleem|xacuti|phaal|balti|methi"),
    ("dal", r"\bdal\b|\bdaal\b|dhal|lentil|sambar|rasam|rajma"),
    ("tandoori", r"tandoor|tikka\b|kebab|kabab|seekh|boti|reshmi|tangri|shashlik|chicken 65|lamb chops?"),
    ("indian_bread", r"\bnaan\b|\bnan\b|roti|paratha|kulcha|chapati|puri\b|bhatura|papadum|papad"),
    ("indian_snack", r"samosa|pakora|bhaji|bhajji|bhajia|mirchi|chaat|pani puri|bhel|vada|tikki|dosa|idli|uttapam|kachori|dhokla|"
                     r"pav bhaji|vada pav|pakoda|cutlet"),
    ("antipasti", r"bruschetta|arancini|antipast|calamari|caprese|carpaccio|mozzarella sticks|garlic bread|focaccia|"
                  r"crostini|meatballs?|polpette|frittata|burrata|prosciutto"),
    ("meat_main", r"parmigiana|parmesan\b|piccata|marsala|saltimbocca|scaloppine|osso buco|milanese|francese|"
                  r"cacciatore|chicken parm|veal|branzino|salmon|steak"),
    ("salad", r"salad|insalata"),
    ("soup", r"soup|minestrone|zuppa|shorba|pasta e fagioli|ribollita"),
    ("rice", r"\brice\b"),
    ("dessert", r"tiramisu|cannoli|panna cotta|gelato|sorbet|affogato|zeppole|cheesecake|cake|brownie|gulab|kulfi|"
                r"rasmalai|ras malai|kheer|halwa|jalebi|barfi|ladoo|laddu|rasgulla|payasam|mithai|dessert|cookie|"
                r"tart|pudding|ice cream|semifreddo|biscotti"),
    ("drink", r"lassi|chai|masala tea|shake|smoothie|falooda"),
    ("sandwich", r"panini|sandwich|sub\b|hero\b|grinder"),
]
_FAMILY_RE = [(f, re.compile(p, re.I)) for f, p in FAMILY_RULES]


def dish_family(name: str, description: str = "", section: str = "") -> str:
    """Coarse dish family from keyword rules; name wins over description/section."""
    for text in (name, section, description):
        for fam, rx in _FAMILY_RE:
            if text and rx.search(text):
                return fam
    return "other"


def dish_category(name: str, section: str, family: str) -> str:
    s = f"{section} {name}".lower()
    if family == "dessert" or re.search(r"dessert|sweet", s):
        return "dessert"
    if family == "drink":
        return "drink"
    if re.search(r"appeti|starter|small plate|antipast|chaat|soup|shareable|tapas", s) or family in {
        "indian_snack", "antipasti", "soup"}:
        return "starter"
    if re.search(r"\bside|bread|accompan|raita|naan|rice\b", s) or family in {"indian_bread", "rice"}:
        return "side"
    return "main"


# ---------- loaders ----------

def _row(uid, source, cuisine, name, description, ing_raw, category_hint="", price=None, menu_style=False,
         extra=None) -> dict:
    return dict(uid=uid, source=source, cuisine=cuisine, name=name, description=(description or "")[:600],
                ingredients_raw=list(ing_raw or []), category_hint=category_hint or "", price_usd=price,
                menu_style=menu_style, extra=json.dumps(extra or {}))


def load_indian_food_101() -> list[dict]:
    p = RAW / "indian_food_101" / "indian_food.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p).replace({-1: np.nan, "-1": np.nan})
    rows = []
    for i, r in df.iterrows():
        ings = [x.strip() for x in str(r.get("ingredients", "")).split(",") if x.strip() and x.strip() != "nan"]
        region = ", ".join(str(x) for x in [r.get("state"), r.get("region")] if isinstance(x, str))
        desc = f"{r.get('course', '')} from {region}".strip() if region else str(r.get("course", "") or "")
        rows.append(_row(f"if_{i:04d}", "indian_food_101", "indian", clean_text(r["name"]), clean_text(desc), ings,
                         category_hint=str(r.get("course", "") or ""),
                         extra={"flavor_profile": r.get("flavor_profile") if isinstance(r.get("flavor_profile"), str) else None,
                                "diet": r.get("diet") if isinstance(r.get("diet"), str) else None,
                                "region": region}))
    return rows


ARCHANA_NON_INDIAN = re.compile(r"continental|mexican|asian|thai|chinese|french|middle eastern|mediterranean|"
                                r"european|african|greek|japanese|korean|american|british|lebanese|indonesian|"
                                r"vietnamese|malaysian|singapore|sri lankan|nepalese|burmese|caribbean|"
                                r"indo chinese|fusion|world|arab|afghan|jewish|spanish|german|scandinavian|"
                                r"brazilian|hungarian|swedish|russian|turkish|egyptian|moroccan|persian|"
                                r"pakistani|bangladeshi|dessert|snack|side|lunch|dinner|breakfast|brunch|"
                                r"appetizer|healthy|vegan|diabetic|baby|kid|toddler", re.I)


def load_archana() -> list[dict]:
    d = RAW / "archana"
    files = list(d.glob("*.csv")) if d.exists() else []
    if not files:
        return []
    df = pd.read_csv(files[0])
    rows = []
    for i, r in df.iterrows():
        cz = str(r.get("Cuisine", "") or "").replace("﻿", "").strip()
        if "italian" in cz.lower():
            cuisine = "italian"
        elif cz and not ARCHANA_NON_INDIAN.search(cz):
            cuisine = "indian"  # "Indian", "North Indian Recipes", regional names, etc.
        else:
            continue
        name = re.sub(r"\s*recipe\s*$", "", clean_text(r.get("TranslatedRecipeName", "")), flags=re.I)
        name = re.sub(r"\s*-\s*.*$", "", name) if len(name) > 60 else name
        ings_s = r.get("Cleaned-Ingredients") if isinstance(r.get("Cleaned-Ingredients"), str) else r.get(
            "TranslatedIngredients", "")
        ings = [x.strip() for x in str(ings_s).split(",") if x.strip()]
        instr = clean_text(r.get("TranslatedInstructions", ""))
        rows.append(_row(f"ak_{i:05d}", "archana", cuisine, name, instr[:400], ings, category_hint=cz,
                         extra={"region": cz, "url": r.get("URL")}))
    return rows


def _lit(s):
    try:
        return ast.literal_eval(s) if isinstance(s, str) else []
    except (ValueError, SyntaxError):
        return []


def load_foodcom() -> list[dict]:
    p = RAW / "foodcom" / "RAW_recipes.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p, usecols=["name", "id", "tags", "nutrition", "description", "ingredients"])
    t = df["tags"].fillna("")
    mask = t.str.contains(r"'(?:indian|italian|pasta|pizza|curries)'", regex=True)
    df = df[mask]
    rows = []
    for r in df.itertuples(index=False):
        tags = set(_lit(r.tags))
        other = tags & {"mexican", "asian", "thai", "chinese", "japanese", "korean", "greek", "middle-eastern",
                        "african", "caribbean", "south-american", "vietnamese", "indonesian", "pakistani"}
        if "indian" in tags and "italian" not in tags:
            cuisine = "indian"
        elif "italian" in tags and "indian" not in tags:
            cuisine = "italian"
        elif tags & {"pasta", "pizza"} and "indian" not in tags and not other:
            cuisine = "italian"
        elif "curries" in tags and not other and "italian" not in tags and "indian" in tags:
            cuisine = "indian"
        else:
            continue
        nut = _lit(r.nutrition)
        nutrition = dict(zip(["calories", "total_fat_pdv", "sugar_pdv", "sodium_pdv", "protein_pdv", "sat_fat_pdv",
                              "carbs_pdv"], nut)) if len(nut) == 7 else None
        course = next((c for c in ["main-dish", "desserts", "appetizers", "side-dishes", "soups-stews", "salads",
                                   "breads", "beverages", "snacks", "condiments-etc"] if c in tags), "")
        if course in {"beverages", "condiments-etc"}:
            continue
        name = clean_text(str(r.name)).title()
        rows.append(_row(f"fc_{r.id}", "foodcom", cuisine, name, clean_text(r.description), _lit(r.ingredients),
                         category_hint=course, extra={"nutrition": nutrition, "tags": sorted(tags)[:40]}))
    return rows


def load_uber_eats() -> tuple[list[dict], pd.DataFrame]:
    d = RAW / "uber_eats"
    if not (d / "restaurants.csv").exists():
        return [], pd.DataFrame()
    rest = pd.read_csv(d / "restaurants.csv")
    cat = rest["category"].fillna("").str.lower()
    is_ind = cat.str.contains("indian")
    is_ita = cat.str.contains("italian|pizza")
    rest = rest[is_ind ^ is_ita].copy()
    rest["cuisine"] = np.where(cat[rest.index].str.contains("indian"), "indian", "italian")
    rest["italian_proper"] = cat[rest.index].str.contains("italian")
    ids = set(rest["id"])
    chunks = []
    for ch in pd.read_csv(d / "restaurant-menus.csv", chunksize=500_000,
                          usecols=["restaurant_id", "category", "name", "description", "price"]):
        chunks.append(ch[ch["restaurant_id"].isin(ids)])
    m = pd.concat(chunks, ignore_index=True)
    m["name"] = m["name"].map(clean_text)
    m["description"] = m["description"].map(clean_text)
    m["category"] = m["category"].map(clean_text)
    rest["name"] = rest["name"].map(clean_text)
    m = m[m["description"].str.len() >= 15]
    m = m[~m["name"].str.contains(DROP_NAME) & ~m["name"].str.contains(DROP_PREFIX)]
    m = m[~(m["name"].str.contains(DROP_CONDIMENT) & (m["name"].str.split().str.len() <= 3))]
    m = m[~m["category"].str.contains(DROP_SECTION)]
    m = m.merge(rest[["id", "cuisine", "name", "price_range", "italian_proper", "lat", "lng", "full_address",
                      "category"]].rename(columns={"id": "restaurant_id", "name": "restaurant_name",
                                                   "category": "rest_category"}),
                on="restaurant_id")
    ita = m["cuisine"] == "italian"
    m = m[~(ita & m["name"].str.contains(NON_ITALIAN))]
    m = m[~(~ita & m["name"].str.contains(NON_INDIAN))]
    m["price_usd"] = m["price"].map(parse_price)
    m = m.reset_index(drop=True)
    m["uid"] = [f"ue_{i:06d}" for i in range(len(m))]
    full = m.copy()  # uncapped (used to pick mock-menu restaurants)
    # Stratified cap per restaurant: prefer longer descriptions, deterministic shuffle within.
    rng = np.random.default_rng(SEED)
    m["_r"] = rng.random(len(m))
    m["_dl"] = m["description"].str.len().clip(upper=200) + m["_r"] * 40
    m = m.sort_values(["restaurant_id", "_dl"], ascending=[True, False])
    m["_rank"] = m.groupby("restaurant_id").cumcount()
    cap = np.where(m["cuisine"] == "indian", UE_CAP_INDIAN, UE_CAP_ITALIAN)
    m = m[m["_rank"] < cap].drop(columns=["_r", "_dl", "_rank"])
    if len(m) > UE_MAX_ITEMS:
        m = m.sample(UE_MAX_ITEMS, random_state=SEED)
    rows = [ue_row(r) for _, r in m.iterrows()]
    return rows, full


def ue_row(r) -> dict:
    return _row(r["uid"], "uber_eats", r["cuisine"], r["name"], r["description"], [],
                         category_hint=r["category"], price=r["price_usd"], menu_style=True,
                         extra={"restaurant_id": int(r["restaurant_id"]), "restaurant_name": r["restaurant_name"],
                                "price_range": r["price_range"] if isinstance(r["price_range"], str) else None})


def load_whats_cooking() -> pd.DataFrame:
    p = RAW / "whats_cooking" / "train.json"
    if not p.exists():
        return pd.DataFrame(columns=["uid", "cuisine", "ingredients_canonical"])
    data = json.loads(p.read_text())
    rows = [dict(uid=f"wc_{d['id']}", cuisine=d["cuisine"], ingredients_canonical=canonicalize_list(d["ingredients"]))
            for d in data if d.get("cuisine") in {"indian", "italian"}]
    return pd.DataFrame(rows)


# ---------- ingredient extraction from free text (Uber Eats) ----------

class Extractor:
    """Match known canonical ingredients (1-3 grams) and synonym keys in free text."""

    def __init__(self, vocab: set[str]):
        self.lookup: dict[str, str] = {v: v for v in vocab}
        for k, v in synonyms().items():
            if v in vocab:
                self.lookup.setdefault(k, v)
        # skip overly generic / misleading single tokens
        for bad in ["oil", "water", "salt", "all-purpose flour", "sugar", "black pepper", "spice", "masala", "sauce",
                    "seasoning", "herb", "cheese", "meat", "vegetable", "topping", "side", "fresh", "hot", "sweet", "powder",
                    "ice", "nut", "curry", "cream", "bread", "sauce", "assorted vegetable", "pizza", "pasta"]:
            self.lookup.pop(bad, None)
        self.lookup["black pepper"] = "black pepper"

    def __call__(self, text: str) -> list[str]:
        words = re.findall(r"[a-z']+", text.lower())
        words = [singularize(w) for w in words]
        found, seen = [], set()
        i = 0
        while i < len(words):
            hit = None
            for n in (3, 2, 1):
                g = " ".join(words[i:i + n])
                if len(words[i:i + n]) == n and g in self.lookup:
                    hit = (self.lookup[g], n)
                    break
            if hit:
                if hit[0] not in seen:
                    seen.add(hit[0])
                    found.append(hit[0])
                i += hit[1]
            else:
                i += 1
        return found


# ---------- dedupe ----------

def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Exact normalized name+cuisine, then fuzzy ratio>95 within first-4-char blocks. Protected rows never dropped."""
    df = df.copy()
    df["_n"] = df["name"].map(norm_name)
    df = df[df["_n"].str.len() >= 3]
    # priority: protected menu dishes, then menu-style, then rows with more info
    df["_prio"] = (df["protected"].astype(int) * 4 + df["menu_style"].astype(int) * 2
                   + (df["description"].str.len() > 40).astype(int))
    df = df.sort_values(["_prio", "uid"], ascending=[False, True])
    keep = ~df.duplicated(["_n", "cuisine"], keep="first") | df["protected"]
    df = df[keep]
    drop = set()
    for (cz, blk), g in df.groupby([df["cuisine"], df["_n"].str[:4]]):
        if len(g) < 2:
            continue
        names = g["_n"].tolist()
        uids = g["uid"].tolist()
        prot = g["protected"].tolist()
        mat = cdist(names, names, scorer=fuzz.ratio, score_cutoff=96, dtype=np.uint8, workers=-1)
        removed = np.zeros(len(names), bool)
        for i in range(len(names)):
            if removed[i]:
                continue
            js = np.nonzero(mat[i, i + 1:])[0] + i + 1
            for j in js:
                if not prot[j]:
                    removed[j] = True
        drop.update(u for u, r in zip(uids, removed) if r)
    df = df[~df["uid"].isin(drop)]
    return df.drop(columns=["_n", "_prio"])


# ---------- mock menu ----------

TARGET_FAMILIES = {"italian": ["pizza", "pasta", "antipasti", "dessert", "risotto", "salad", "meat_main"],
                   "indian": ["curry", "biryani", "tandoori", "indian_bread", "dal", "indian_snack", "dessert"]}
CUISINE_FAMILIES = {"italian": {"pizza", "pasta", "risotto", "antipasti", "meat_main", "salad", "soup", "dessert"},
                    "indian": {"curry", "biryani", "dal", "tandoori", "indian_bread", "indian_snack", "rice",
                               "dessert", "drink"}}
VENUE_FAMILIES = {"italian": ["pizza", "pasta"], "indian": ["curry", "biryani"]}
VIBES = {1: ["quick", "casual"], 2: ["casual", "family"], 3: ["traditional", "family"], 4: ["date-night", "trendy"]}


def build_mock_menu(ue: pd.DataFrame) -> tuple[list[dict], list[dict], dict[str, str]]:
    """Pick 10 Italian + 10 Indian Uber Eats restaurants, 12-20 items each. Returns dishes, restaurants, ue_uid->d_id."""
    ue = ue.copy()
    ue["family"] = [dish_family(n, d, c) for n, d, c in zip(ue["name"], ue["description"], ue["category"])]
    ue = ue[ue["price_usd"].notna()]
    dishes, rests, uid_map = [], [], {}
    did = 1
    for ci, cuisine in enumerate(["italian", "indian"]):
        sub = ue[ue["cuisine"] == cuisine]
        if cuisine == "italian":
            sub = sub[sub["italian_proper"]]  # proper Italian restaurants, not pizza-only chains
        stats = sub.groupby("restaurant_id").agg(n=("uid", "size"), fams=("family", lambda s: set(s)),
                                                 med=("price_usd", "median"),
                                                 pr=("price_range", "first"), rname=("restaurant_name", "first"))
        fam_set = CUISINE_FAMILIES[cuisine]
        stats["purity"] = sub.groupby("restaurant_id")["family"].apply(lambda s: s.isin(fam_set).mean())
        stats["pizza_share"] = sub.groupby("restaurant_id")["family"].apply(lambda s: (s == "pizza").mean())
        stats["n_pasta"] = sub.groupby("restaurant_id")["family"].apply(lambda s: (s == "pasta").sum())
        stats = stats[(stats["n"] >= 14) & (stats["purity"] >= 0.8)]
        if cuisine == "italian":
            stats = stats[(stats["pizza_share"] <= 0.5) & (stats["n_pasta"] >= 3)]
        stats["cover"] = stats["fams"].map(lambda f: len(f & set(TARGET_FAMILIES[cuisine])))
        stats["venue"] = stats["fams"].map(lambda f: set(VENUE_FAMILIES[cuisine]) <= f)
        stats = stats[stats["venue"]]
        # unique names (avoid picking 3 branches of one chain)
        stats["base"] = stats["rname"].str.replace(r"\s*\(.*\)", "", regex=True).str.lower()
        stats = stats.sort_values(["cover", "n"], ascending=False).drop_duplicates("base")
        # spread across median price: pick from price deciles
        stats = stats.sort_values("med")
        if len(stats) < 10:
            raise RuntimeError(f"not enough {cuisine} restaurants for mock menu: {len(stats)}")
        qs = np.linspace(0, len(stats) - 1, 10).round().astype(int)
        picked = stats.iloc[qs]
        tiers = [1, 1, 2, 2, 2, 3, 3, 3, 4, 4]  # rank-based tiers by median item price
        for k, (rid, srow) in enumerate(picked.iterrows()):
            r_id = f"r_{ci * 10 + k + 1:02d}"
            items = sub[sub["restaurant_id"] == rid].copy()
            items = items.drop_duplicates("name")
            chosen = []
            for fam in TARGET_FAMILIES[cuisine]:  # at least one per target family where available
                hit = items[(items["family"] == fam) & ~items["uid"].isin(chosen)]
                chosen += hit["uid"].head(2 if fam in VENUE_FAMILIES[cuisine] else 1).tolist()
            pool = items[~items["uid"].isin(chosen) & items["family"].isin(CUISINE_FAMILIES[cuisine])
                         & ~items["name"].str.contains(r"create your own|build your own|\bbyo\b", case=False)]
            groups = [g["uid"].tolist() for _, g in pool.groupby("family")]
            while len(chosen) < 15 and any(groups):
                for g in groups:
                    if g and len(chosen) < 15:
                        chosen.append(g.pop(0))
            chosen = chosen[:20]
            items = items.set_index("uid").loc[chosen].reset_index()
            first = items.iloc[0]
            addr = str(first.get("full_address", "") or "")
            r_dish_ids = []
            for _, it in items.iterrows():
                d_id = f"d_{did:04d}"
                did += 1
                uid_map[it["uid"]] = d_id
                r_dish_ids.append(d_id)
                fam = it["family"]
                dishes.append(Dish(
                    id=d_id, restaurant_id=r_id, name=it["name"], description=it["description"][:600],
                    cuisine=cuisine, category=dish_category(it["name"], it["category"], fam), dish_family=fam,
                    price_gbp=round(float(it["price_usd"]) * USD_TO_GBP, 2), ingredients=[], allergens=[],
                    taste=None, serving_temp="cold" if fam in {"salad", "dessert", "drink"} else "hot",
                    adventurousness=0, tagged_by="human", verified=False, is_mock=True,
                ).model_dump())
            rests.append(Restaurant(
                id=r_id, name=str(srow["rname"]), cuisine=cuisine, price_tier=tiers[k], vibe_tags=VIBES[tiers[k]],
                area=", ".join(addr.split(",")[-3:-1]).strip() if addr else "",
                lat=float(first["lat"]) if pd.notna(first["lat"]) else 0.0,
                lng=float(first["lng"]) if pd.notna(first["lng"]) else 0.0,
                halal_status="unknown", menu_source_url="uber-eats-kaggle-mock", dish_ids=r_dish_ids,
            ).model_dump())
    return dishes, rests, uid_map


# ---------- main ----------

def main() -> None:
    config.seed_everything()
    OUT.mkdir(parents=True, exist_ok=True)
    config.ARTIFACTS.mkdir(exist_ok=True)
    config.REPORTS.mkdir(exist_ok=True)

    print("loading recipes ...")
    recipe_rows = load_indian_food_101() + load_archana() + load_foodcom()
    for r in recipe_rows:
        r["ingredients_canonical"] = canonicalize_list(r["ingredients_raw"])
    rc = Counter(i for r in recipe_rows for i in r["ingredients_canonical"])
    recipe_vocab = {k for k, v in rc.items() if v >= VOCAB_MIN_FREQ}
    print(f"  recipes: {len(recipe_rows)}; recipe vocab (freq>={VOCAB_MIN_FREQ}): {len(recipe_vocab)}")

    print("loading uber eats ...")
    ue_rows, ue_df = load_uber_eats()
    extract = Extractor(recipe_vocab)
    for r in ue_rows:
        r["ingredients_raw"] = extract(f"{r['name']} {r['description']}")
        r["ingredients_canonical"] = list(r["ingredients_raw"])
    print(f"  uber eats items after filters/caps: {len(ue_rows)}")

    print("building mock menu ...")
    dishes, rests, uid_map = build_mock_menu(ue_df)
    have = {r["uid"] for r in ue_rows}
    for u in uid_map:
        if u not in have:
            r = ue_row(ue_df[ue_df["uid"] == u].iloc[0])
            r["ingredients_raw"] = extract(f"{r['name']} {r['description']}")
            r["ingredients_canonical"] = list(r["ingredients_raw"])
            ue_rows.append(r)
    ue_by_uid = {r["uid"]: r for r in ue_rows}
    for d in dishes:  # mock dishes get extracted ingredients too
        src = next(u for u, v in uid_map.items() if v == d["id"])
        d["ingredients"] = ue_by_uid[src]["ingredients_raw"]
        d["ingredients_canonical"] = ue_by_uid[src]["ingredients_canonical"]
    for r in ue_rows:
        if r["uid"] in uid_map:
            r["uid"] = uid_map[r["uid"]]
            r["extra"] = json.dumps({**json.loads(r["extra"]), "mock_menu": True})

    df = pd.DataFrame(recipe_rows + ue_rows)
    df["protected"] = df["uid"].str.startswith("d_")
    n_before = len(df)
    df = dedupe(df)
    print(f"  dedupe: {n_before} -> {len(df)}")

    vc = Counter(i for lst in df["ingredients_canonical"] for i in lst)
    vocab = {k: v for k, v in sorted(vc.items(), key=lambda kv: (-kv[1], kv[0])) if v >= VOCAB_MIN_FREQ}
    df["ingredients_canonical"] = df["ingredients_canonical"].map(lambda xs: [x for x in xs if x in vocab])
    df = df.drop(columns=["protected"])
    cols = ["uid", "source", "cuisine", "name", "description", "ingredients_raw", "ingredients_canonical",
            "category_hint", "price_usd", "menu_style", "extra"]
    df = df[cols].sort_values("uid").reset_index(drop=True)
    df.to_parquet(OUT / "corpus.parquet", index=False)
    (config.ARTIFACTS / "ingredient_vocab.json").write_text(json.dumps(vocab, indent=0))

    for d in dishes:
        d["ingredients_canonical"] = [x for x in d["ingredients_canonical"] if x in vocab]
        Dish.model_validate(d)
    (OUT / "menu_dishes.json").write_text(json.dumps(dishes, indent=1))
    (OUT / "restaurants_mock.json").write_text(json.dumps(rests, indent=1))

    wc = load_whats_cooking()
    wc.to_parquet(OUT / "cooc_extra.parquet", index=False)

    # ---- summary ----
    tab = df.groupby(["source", "cuisine"]).size().unstack(fill_value=0)
    tab["total"] = tab.sum(axis=1)
    print(tab.to_string())
    print(f"total rows: {len(df)}  | vocab: {len(vocab)} | menu dishes: {len(dishes)} | what's cooking: {len(wc)}")
    fam = pd.Series([d["dish_family"] for d in dishes]).value_counts()
    venue = {}
    for f in ["pizza", "pasta", "curry", "biryani"]:
        venue[f] = sorted({(d["restaurant_id"]) for d in dishes if d["dish_family"] == f})
    tiers = {r["id"]: r["price_tier"] for r in rests}
    lines = [
        "# Corpus summary", "",
        f"Total rows: **{len(df)}** (before dedupe {n_before}). Ingredient vocab (freq>={VOCAB_MIN_FREQ}): **{len(vocab)}**.",
        f"Rows with >=1 canonical ingredient: {int((df['ingredients_canonical'].str.len() > 0).sum())}.",
        f"What's Cooking rows (co-occurrence only): {len(wc)}.", "",
        "## Rows per source x cuisine", "", tab.to_markdown() if hasattr(tab, "to_markdown") else tab.to_string(), "",
        "## Mock menu (is_mock=true, from Uber Eats US)", "",
        f"{len(dishes)} dishes across {len(rests)} restaurants (tiers assigned by median item price rank).", "",
        "Dish families: " + ", ".join(f"{k} {v}" for k, v in fam.items()), "",
        "Venue-round families (restaurant: tier):", "",
    ]
    for f, rs in venue.items():
        lines.append(f"- {f}: " + ", ".join(f"{r}:T{tiers[r]}" for r in rs))
    (config.REPORTS / "corpus_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
