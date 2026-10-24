"""Builds tests/fixtures/mini_corpus.parquet (40 dishes) and mini_vocab.json (60 ingredients)."""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
D = [
    ("indian", "Chicken Tikka Masala", "Chargrilled chicken in a creamy spiced tomato sauce", "chicken,tomato,cream,garam masala,ginger,garlic"),
    ("indian", "Lamb Vindaloo", "Fiery Goan curry with vinegar, chilli and potato", "lamb,vinegar,red chilli,potato,garlic"),
    ("indian", "Vegetable Samosa", "Crisp fried pastry filled with spiced potato and peas", "potato,pea,all-purpose flour,cumin"),
    ("indian", "Mango Lassi", "Chilled yoghurt drink blended with alphonso mango", "yoghurt,mango,sugar"),
    ("indian", "Tarka Dal", "Yellow lentils tempered with garlic, cumin and ghee", "lentil,garlic,cumin,ghee,turmeric"),
    ("indian", "Onion Bhaji", "Crispy onion fritters in gram flour batter", "onion,gram flour,chilli,coriander"),
    ("indian", "Palak Paneer", "Cottage cheese cubes in a smooth spinach gravy", "paneer,spinach,cream,garlic"),
    ("indian", "Chicken Biryani", "Fragrant basmati layered with spiced chicken and saffron", "chicken,basmati rice,saffron,onion,yoghurt"),
    ("indian", "Gulab Jamun", "Milk dumplings soaked in rose cardamom syrup", "milk powder,sugar,cardamom,rose water"),
    ("indian", "Aloo Gobi", "Dry curry of potato and cauliflower with turmeric", "potato,cauliflower,turmeric,cumin"),
    ("indian", "Pani Puri", "Crisp hollow puris with tamarind water and potato", "semolina,tamarind,potato,mint,chilli"),
    ("indian", "Butter Chicken", "Tandoori chicken in a buttery tomato makhani sauce", "chicken,butter,tomato,cream,fenugreek"),
    ("indian", "Garlic Naan", "Leavened flatbread baked in the tandoor with garlic butter", "all-purpose flour,garlic,butter"),
    ("indian", "Lamb Rogan Josh", "Kashmiri lamb curry with aromatic spices", "lamb,kashmiri chilli,yoghurt,onion"),
    ("indian", "Chana Masala", "Chickpeas in a tangy onion tomato gravy", "chickpea,tomato,onion,amchur"),
    ("indian", "Prawn Malabar Curry", "Prawns in coconut milk with curry leaves", "prawn,coconut milk,curry leaf,tamarind"),
    ("indian", "Masala Dosa", "Crisp rice crepe with spiced potato, sambar and chutney", "rice,urad dal,potato,mustard seed"),
    ("indian", "Chicken 65", "Deep-fried spicy chicken bites with curry leaves", "chicken,red chilli,curry leaf,cornflour"),
    ("indian", "Raita", "Whisked yoghurt with cucumber and roasted cumin", "yoghurt,cucumber,cumin"),
    ("indian", "Phaal Curry", "The hottest curry on the menu, scotch bonnet heat", "chicken,scotch bonnet,tomato,onion"),
    ("italian", "Margherita Pizza", "San Marzano tomato, fior di latte, basil", "tomato,mozzarella,basil,all-purpose flour"),
    ("italian", "Spaghetti Carbonara", "Guanciale, egg yolk, pecorino, black pepper", "spaghetti,guanciale,egg,pecorino,black pepper"),
    ("italian", "Tiramisu", "Savoiardi soaked in espresso with mascarpone and cocoa", "mascarpone,espresso,ladyfinger,cocoa,sugar"),
    ("italian", "Arancini", "Fried risotto balls with mozzarella centre", "arborio rice,mozzarella,breadcrumb,parmesan"),
    ("italian", "Penne Arrabbiata", "Tomato sauce with garlic and dried chilli", "penne,tomato,garlic,chilli flake"),
    ("italian", "Rocket and Parmesan Salad", "Peppery rocket, shaved parmesan, balsamic", "rocket,parmesan,balsamic vinegar,olive oil"),
    ("italian", "Lasagne al Forno", "Layers of beef ragu, bechamel and parmesan", "beef,tomato,bechamel,parmesan,pasta"),
    ("italian", "Bruschetta", "Toasted sourdough with tomato, garlic and basil", "sourdough,tomato,garlic,basil,olive oil"),
    ("italian", "Risotto ai Funghi", "Creamy arborio rice with porcini mushrooms", "arborio rice,porcini,parmesan,butter"),
    ("italian", "Panna Cotta", "Vanilla cream set with berry coulis", "cream,vanilla,gelatin,berry"),
    ("italian", "Diavola Pizza", "Spicy salami, chilli, tomato and mozzarella", "salami,chilli,tomato,mozzarella"),
    ("italian", "Spaghetti alle Vongole", "Clams, white wine, garlic and parsley", "spaghetti,clam,white wine,garlic,parsley"),
    ("italian", "Fritto Misto", "Lightly fried calamari and prawns with lemon", "squid,prawn,flour,lemon"),
    ("italian", "Cacio e Pepe", "Tonnarelli with pecorino and cracked black pepper", "pasta,pecorino,black pepper"),
    ("italian", "Insalata di Radicchio", "Bitter radicchio with orange and walnuts", "radicchio,orange,walnut,olive oil"),
    ("italian", "Osso Buco", "Braised veal shank with gremolata", "veal,white wine,tomato,lemon zest"),
    ("italian", "Cannoli", "Crisp pastry shells filled with sweet ricotta", "ricotta,sugar,pastry,pistachio"),
    ("italian", "Minestrone", "Clear vegetable soup with beans and pasta", "carrot,celery,bean,tomato,pasta"),
    ("italian", "Pesto Genovese Trofie", "Trofie with basil pesto, potato and green beans", "basil,pine nut,parmesan,potato,green bean"),
    ("italian", "Affogato", "Vanilla gelato drowned in espresso", "gelato,espresso"),
]
rows = []
for i, (c, n, desc, ings) in enumerate(D, 1):
    il = ings.split(",")
    rows.append(dict(uid=f"fx_{i:03d}", source="fixture", cuisine=c, name=n, description=desc,
                     ingredients_raw=il, ingredients_canonical=il, category_hint="main", price_usd=None,
                     menu_style=True, extra="{}"))
pd.DataFrame(rows).to_parquet(HERE / "mini_corpus.parquet", index=False)
(HERE / "mini_uids.txt").write_text("\n".join(r["uid"] for r in rows))
vocab = {}
for r in rows:
    for x in r["ingredients_canonical"]:
        vocab[x] = vocab.get(x, 0) + 1
(HERE / "mini_vocab.json").write_text(json.dumps(vocab, indent=1))
print(len(rows), "dishes,", len(vocab), "ingredients")
