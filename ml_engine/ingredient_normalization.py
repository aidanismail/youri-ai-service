from __future__ import annotations

import re

SPACE_RE = re.compile(r"\s+")
PUNCT_RE = re.compile(r"[^\w\s-]", flags=re.UNICODE)


DIRECT_ALIASES = {
    # Protein aliases
    "chicken": "ayam",
    "ayam negeri": "ayam",
    "dada ayam sesuai selera": "ayam",
    "ayam bgian": "ayam",
    "ayam bagian paha bagi": "ayam",
    "fillet ayam dadu": "ayam",
    "ayam ambil dagingnya": "ayam",
    "sapi": "daging sapi",
    "beef": "daging sapi",
    "sapi khas dalam": "daging sapi",
    "sapi has dalam": "daging sapi",
    "g daging sapi": "daging sapi",
    "ons daging sapi": "daging sapi",
    "daging sapi diiris": "daging sapi",
    "daging sapi pipih": "daging sapi",
    "daging sapi has dalam": "daging sapi",
    "daging sapi giling": "daging sapi",
    "munjung daging sapi giling": "daging sapi",
    "tetelanpipi sapi": "daging sapi",
    "daging sapikambing": "daging sapi",
    "g daging kambing": "daging kambing",
    "g daging kambing dadu": "daging kambing",
    "daging kambing dipotong": "daging kambing",
    "daging paha kambing": "daging kambing",
    "sesuai selera daging kambing": "daging kambing",
    "iga kambing": "daging kambing",
    "kaki kambing": "daging kambing",
    "daging kambing boleh sapiayam": "daging kambing",
    "tongkol": "ikan tongkol",
    "tuna": "ikan tuna",
    "patin": "ikan patin",
    "tenggiri": "ikan tenggiri",
    "bandeng": "ikan bandeng",
    "bawal": "ikan bawal",

    # Egg and dairy aliases
    "egg": "telur",
    "scrumble egg": "telur",
    "telor": "telur",
    "telor ayam": "telur",
    "telur ayam": "telur",
    "telor ayam kocok lepas": "telur",
    "telur kocok lepas": "telur",
    "telurkocok lepas": "telur",
    "telur kocok": "telur",
    "telur di ceplok": "telur",
    "telur dadar": "telur",
    "telur lalu dari kulit": "telur",
    "telur bebek": "telur",
    "telor puyuh": "telur puyuh",
    "keju parut": "keju",
    "cheddar parut": "keju",
    "parmesan": "keju",
    "dancow putih": "susu",
    "susu cair putih plain": "susu",
    "susu cair putih": "susu",
    "whipping cream": "krim",
    "gravy": "saus",
    "saori": "saus",

    # Aromatic aliases and common typos
    "bawang putin": "bawang putih",
    "bawang merch": "bawang merah",
    "bj bawang merah": "bawang merah",
    "bj bawang putih": "bawang putih",
    "bawang merah sekitar": "bawang merah",
    "bawang merah tipis": "bawang merah",
    "bawang merah haluskan": "bawang merah",
    "irisan bawang merah": "bawang merah",
    "bawang putih haluskan": "bawang putih",
    "bawang putih kasar": "bawang putih",
    "bawang putih tipis": "bawang putih",
    "bawang putih cacah": "bawang putih",
    "bawang putih dicincang": "bawang putih",
    "bawang putih lalu": "bawang putih",
    "bawang putih bubuk": "bawang putih",
    "bawang merah bawang putih": "bawang merah bawang putih",
    "bawang merah baput": "bawang merah bawang putih",
    "bawang bombai": "bawang bombay",
    "bombay": "bawang bombay",
    "bombay dicincang": "bawang bombay",
    "bawang bombay rajang": "bawang bombay",
    "bawang bombai tipis": "bawang bombay",
    "bawang bombay tipis": "bawang bombay",
    "bawang bombay kotak": "bawang bombay",
    "bawang bombay lalu": "bawang bombay",
    "bawang hijau": "daun bawang",
    "bawang daun": "daun bawang",
    "bawang daunpre": "daun bawang",
    "bawangprei": "daun bawang",
    "bawang prei": "daun bawang",
    "bawang prey memperindah tampilan": "daun bawang",
    "daon bawang": "daun bawang",
    "daun prei": "daun bawang",
    "daun solong": "daun bawang",
    "daun kucai": "daun bawang",
    "bonggol daun bawang": "daun bawang",
    "helai daun bawang": "daun bawang",
    "potongan daun bawang": "daun bawang",
    "daun bawang tipis": "daun bawang",
    "daun bawang rajang": "daun bawang",
    "daun bawang pre": "daun bawang",
    "daun bawang preh": "daun bawang",
    "daun bawang daun sledri": "daun bawang",
    "daun bawang sledri": "daun bawang",
    "seledri": "seledri",
    "helai daun ketumbar": "daun ketumbar",
    "jae": "jahe",
    "jahe saja": "jahe",
    "jahe lebih": "jahe",
    "jari jahe": "jahe",
    "seruas jahe": "jahe",
    "jahe bubuk": "jahe",
    "laos": "lengkuas",
    "laos di": "lengkuas",
    "seruas laos": "lengkuas",
    "lengkus": "lengkuas",
    "jempol laja": "lengkuas",
    "laja": "lengkuas",
    "lengkuas memarkan": "lengkuas",
    "seruas lengkuas": "lengkuas",
    "sere": "serai",
    "sereh": "serai",
    "serehambil putihnya": "serai",
    "sereh memarkan": "serai",
    "sereh simpulkan": "serai",
    "sereh ambil putihnyageprek": "serai",
    "serai memarkan": "serai",
    "serai ambil putihnya memarkan": "serai",
    "btng serai": "serai",
    "daun serai": "serai",
    "kunir": "kunyit",
    "kunyit bakar": "kunyit",
    "sejempol kunyit": "kunyit",
    "jari kunyit": "kunyit",
    "jari kunyit bakar": "kunyit",
    "seruas kunyit": "kunyit",
    "serbuk kunyit": "kunyit",
    "kunyit seujung jari": "kunyit",
    "kira kunyit": "kunyit",
    "telunjuk jari kunyit": "kunyit",
    "jahe kunyit": "kunyit",
    "tumbar": "ketumbar",
    "ketumbar bubuk": "ketumbar",
    "ketumbar bubuk ketumbar": "ketumbar",
    "miri": "kemiri",
    "mrica": "merica",
    "bubuk merica": "merica",
    "merica bubuk": "merica",
    "merica butiran": "merica",
    "sd teh mrica": "merica",
    "lada": "merica",
    "lada bubuk": "merica",
    "lada butiran": "merica",
    "lada putih": "merica",
    "lada putih bubuk": "merica",
    "ladaku": "merica",
    "black pepper": "merica hitam",
    "lada hitam": "merica hitam",
    "lada hitam kasar": "merica hitam",
    "merica hitam gerus": "merica hitam",
    "bumbu lada hitam": "merica hitam",
    "pala bubuk": "pala",
    "bubuk pala": "pala",
    "kengkeh": "cengkeh",
    "cengkih": "cengkeh",
    "jintan": "jinten",
    "jintem": "jinten",
    "sd teh jinten": "jinten",
    "bubuk jinten": "jinten",
    "kayu manis bubuk": "kayu manis",
    "telunjuk jari kayu manis": "kayu manis",
    "pekak bunga lawang": "bunga lawang",
    "bumbu kari indofood": "bumbu kari",
    "bumbu kari bubuk": "bumbu kari",
    "bubuk kari": "bumbu kari",

    # Leaf aliases
    "salam": "daun salam",
    "lmbr daun salam": "daun salam",
    "lmbar daun salam": "daun salam",
    "helai daun salam": "daun salam",
    "salam koja": "daun salam",
    "daun jeruk purut": "daun jeruk",
    "daun jeruk optional": "daun jeruk",
    "daun jeruk tipis": "daun jeruk",
    "daun jeruksobek": "daun jeruk",
    "helai daun jeruk": "daun jeruk",
    "lmbr daun jeruk": "daun jeruk",
    "daun jeruk buang tulangnya": "daun jeruk",
    "daun jeruk nipis": "daun jeruk",
    "daun jeruk potek": "daun jeruk",
    "daun jeruk remas": "daun jeruk",
    "daun pandan simpulkan": "daun pandan",

    # Vegetables and fruits
    "cabai": "cabai",
    "cabe": "cabai",
    "lombok": "cabai",
    "rawit domba": "cabai rawit",
    "cili pedas": "cabai",
    "abon cabe": "cabai bubuk",
    "cabe bubuk": "cabai bubuk",
    "saus cabai": "saus sambal",
    "saos pedas": "saus sambal",
    "saus sambel": "saus sambal",
    "saos sambal": "saus sambal",
    "saos sambel": "saus sambal",
    "cabe ijo": "cabai hijau",
    "cabe hijau": "cabai hijau",
    "cabai hijau keriting": "cabai hijau",
    "cabe ijo keriting": "cabai hijau",
    "cabe setan": "cabai rawit",
    "cabe gendot": "cabai hijau",
    "tomat ukkecil": "tomat",
    "tomat uksedang": "tomat",
    "tomat setengah dadu": "tomat",
    "tomat dadu": "tomat",
    "tomatiris bagi": "tomat",
    "irisan tomat": "tomat",
    "sambel tomat": "tomat",
    "saos tomat": "saus tomat",
    "kentang dadu": "kentang",
    "kentang goreng": "kentang",
    "kentang dipotong panjang": "kentang",
    "ons kentangpotong gorengsisihkan": "kentang",
    "wortel korek": "wortel",
    "wortel korek api": "wortel",
    "wortel tipis": "wortel",
    "wortel parut": "wortel",
    "sesuai selera wortel": "wortel",
    "kol": "kol",
    "kubis": "kol",
    "helai kol": "kol",
    "sayur kol": "kol",
    "daun kol": "kol",
    "daun kol tipis": "kol",
    "sesuai selera sayur kol": "kol",
    "kol sobek": "kol",
    "letuce": "selada",
    "pokchoi": "sawi",
    "lobak panjang": "lobak",
    "bawang goreng": "bawang merah",
    "brambang goreng tomat garnish": "bawang merah",
    "bunga pepaya": "daun pepaya",
    "buncir kira": "buncis",
    "buncis ptong": "buncis",
    "jagung pipil serut": "jagung",
    "jagung manis": "jagung",
    "caisim": "sawi",
    "sawi hijau": "sawi",
    "sawi putih": "sawi",
    "daun kemangi": "kemangi",
    "daun kunyit": "daun kunyit",
    "daun pepaya": "daun pepaya",
    "daun pisang": "daun pisang",
    "pisang tanduk": "pisang",
    "jeruk nipis lemon": "jeruk nipis",
    "perasan jeruk nipis": "jeruk nipis",
    "irisan jeruk limau": "jeruk limau",
    "jeruk lemon": "lemon",
    "lemon juice": "lemon",
    "air lemon": "lemon",
    "belimbing air matang": "belimbing wuluh",
    "nanas parutblender tanpa air": "nanas",
    "nanas manis dadu": "nanas",
    "parutan nanas": "nanas",
    "nanas parut mengempukkan daging": "nanas",

    # Legumes, carbs, fats, sauces
    "tofu": "tahu",
    "tahu putih": "tahu",
    "tahu putih belah segitiga": "tahu",
    "tahu goreng": "tahu",
    "tahu matang": "tahu",
    "tahu ukxcm": "tahu",
    "tahu sesuai selera": "tahu",
    "separo tahu putih": "tahu",
    "pcs tahu sutra": "tahu",
    "tahu goreng kering": "tahu",
    "tahu cina dadu": "tahu",
    "tahu kuning": "tahu",
    "tahu coklat": "tahu",
    "tahu kulitcoklat": "tahu",
    "tahu segitiga coklat": "tahu",
    "plastik tahu coklat": "tahu",
    "tempe semangit": "tempe",
    "papam tempe": "tempe",
    "tempe ukxcm": "tempe",
    "tempe dadu": "tempe",
    "kotak tempe": "tempe",
    "kedelai calon tempe": "tempe",
    "tempe kotak panjang": "tempe",
    "tempe sesuai selera": "tempe",
    "tempe mendoan": "tempe",
    "tempe busuk": "tempe",
    "tempe kacang": "tempe",
    "mie instan": "mie",
    "mie urai": "mie",
    "mie burung dara": "mie",
    "bahan kuah mie": "mie",
    "bihun": "bihun",
    "pack soun": "soun",
    "soun telah diseduh ditiriskan": "soun",
    "cup beras": "beras",
    "canting beras": "beras",
    "terigu": "tepung terigu",
    "terigu serbaguna": "tepung terigu",
    "adonan tepung krispi": "tepung bumbu",
    "tepung bumbu sajiku": "tepung bumbu",
    "tepung sajiku": "tepung bumbu",
    "maizena": "tepung maizena",
    "larutan maizena": "tepung maizena",
    "maizena larutkan air": "tepung maizena",
    "sagu": "tepung sagu",
    "tapioka": "tepung tapioka",
    "santal kental": "santan",
    "santan kara": "santan",
    "santan kental": "santan",
    "santan instan": "santan",
    "santan kelapa": "santan",
    "gayung santan kelapa": "santan",
    "kara": "santan",
    "sun kara": "santan",
    "olive oil": "minyak",
    "minyak goreng": "minyak",
    "minyak menggoreng": "minyak",
    "minyak menumis": "minyak",
    "minyak goreng menumis": "minyak",
    "minyak menumis bumbu": "minyak",
    "minyak sayur": "minyak",
    "mentega butter": "mentega",
    "mentega tawar": "mentega",
    "mentega blueband": "mentega",
    "munjung mentega": "mentega",
    "margarin": "mentega",
    "wijen hiasan": "wijen",
    "kecap bango": "kecap manis",
    "sendok kecap manis": "kecap manis",
    "sesuai selera kecap manis": "kecap manis",
    "kecap manis tropicana slim": "kecap manis",
    "kecap manis kecap asin": "kecap manis",
    "sesuai selera kecap": "kecap",
    "kecap ikan": "kecap ikan",
    "sch saos tiram": "saus tiram",
    "saos tiram": "saus tiram",
    "saori saus tiram": "saus tiram",
    "saori saos tiram": "saus tiram",
    "penuh saus tiram": "saus tiram",
    "saus teriyaki": "saus teriyaki",
    "saos teriyaki": "saus teriyaki",
    "sendok saus teriyaki": "saus teriyaki",
    "saus raja rasa": "saus raja rasa",
    "saus inggris": "saus inggris",
    "saori saus lada hitam": "saus lada hitam",
    "saori saus asam manis": "saus asam manis",
    "mayonaise": "mayones",
    "cuka dapurjeruk nipislemon": "cuka",
    "air asam jawa": "asam jawa",
    "asam larutkan dg air": "asam jawa",
    "asem jawa diberi air": "asam jawa",
    "asam gelugur": "asam gelugur",
    "trasi": "terasi",
    "terasi abc": "terasi",
    "gulput": "gula",
    "madu": "madu",
    "baking powder": "baking powder",
    "kismis": "kismis",
    "kurma buang": "kurma",
    "bambu": "bumbu",
    "kulit kakap": "kulit ikan",
    "teh celup": "teh",
    "es batu": "air",
    "air": "air",
    "beri air": "air",
    "air matang": "air",
    "air putih": "air",
    "air secukup nya": "air",
    "air es": "air",
    "air hangat": "air",
    "air merebus daging": "air",
    "sesuai selera air secukupny": "air",
    "satu": "bahan",
    "lainnya": "bahan",
}

COMMON_NOISE_TERMS = {
    "bahan",
    "bumbu",
    "pelengkap",
    "tambahan",
    "secukupnya",
    "sesuai selera",
    "optional",
    "haluskan",
    "dicincang",
    "iris",
    "irisan",
    "tipis",
    "kasar",
    "bubuk",
    "memarkan",
    "geprek",
    "simpulkan",
    "ambil putihnya",
    "buang tulangnya",
    "dipotong",
    "dadu",
    "rajang",
    "korek api",
}

CHICKEN_EXCLUDE_TERMS = {
    "kaldu",
    "masako",
    "royco",
    "penyedap",
    "telur",
    "telor",
    "pangsit",
    "sosis",
    "bumbu",
    "racik",
    "minyak ayam",
}

BEEF_EXCLUDE_TERMS = {
    "kaldu",
    "royco",
    "penyedap",
    "bakso",
    "sosis",
    "hati",
    "lidah",
}


def normalize_ingredient(value: object) -> str:
    text = PUNCT_RE.sub(" ", str(value or "").casefold())
    return SPACE_RE.sub(" ", text).strip(" -_")


def _has_any(text: str, terms: set[str] | tuple[str, ...] | list[str]) -> bool:
    return any(term in text for term in terms)


def _canonical_from_keyword(ingredient: str) -> str | None:
    # Sauces and packaged/basic seasonings first, because many contain meat words.
    if _has_any(ingredient, ("royco", "masako", "sasa", "magic lezat", "knorr", "vetsin")):
        return "penyedap rasa"
    if ingredient in {"air", "beri air", "air matang", "air putih", "air secukup nya", "air es", "air hangat", "air merebus daging", "sesuai selera air secukupny"}:
        return "air"
    if "es batu" in ingredient:
        return "air"
    if "kaldu" in ingredient:
        return "kaldu bubuk"
    if "penyedap" in ingredient:
        return "penyedap rasa"
    if "garam" in ingredient and "gula" in ingredient:
        return "garam gula"
    if "garam" in ingredient and ("lada" in ingredient or "merica" in ingredient or "mrica" in ingredient):
        return "garam merica"
    if "gula" in ingredient:
        if "merah" in ingredient or "jawa" in ingredient:
            return "gula merah"
        return "gula"
    if "garam" in ingredient:
        return "garam"

    if "kecap asin" in ingredient:
        return "kecap asin"
    if "kecap ikan" in ingredient:
        return "kecap ikan"
    if "kecap manis" in ingredient or "kecap bango" in ingredient:
        return "kecap manis"
    if ingredient == "kecap" or ingredient.startswith("kecap "):
        return "kecap"
    if "saus tiram" in ingredient or "saos tiram" in ingredient:
        return "saus tiram"
    if "teriyaki" in ingredient:
        return "saus teriyaki"
    if "saus sambal" in ingredient or "saos sambal" in ingredient or "saus sambel" in ingredient or "saos sambel" in ingredient:
        return "saus sambal"
    if "saus tomat" in ingredient or "saos tomat" in ingredient:
        return "saus tomat"
    if "saus" in ingredient or "saos" in ingredient:
        return ingredient
    if ingredient in {"gravy", "saori"}:
        return "saus"
    if "terasi" in ingredient or ingredient == "trasi":
        return "terasi"
    if "tauco" in ingredient:
        return "tauco"
    if "taousi" in ingredient:
        return "taousi"
    if "cuka" in ingredient:
        return "cuka"
    if "asam jawa" in ingredient or "asem jawa" in ingredient:
        return "asam jawa"
    if "asam gelugur" in ingredient:
        return "asam gelugur"

    # Proteins
    if "telur" in ingredient or "telor" in ingredient or "egg" in ingredient:
        if "asin" in ingredient:
            return "telur asin"
        if "puyuh" in ingredient:
            return "telur puyuh"
        return "telur"
    if "ayam" in ingredient and not _has_any(ingredient, CHICKEN_EXCLUDE_TERMS):
        return "ayam"
    if "daging ayam" in ingredient or "ayam fillet" in ingredient or "dada ayam" in ingredient or "fillet ayam" in ingredient:
        return "ayam"
    if "ati ampela ayam" in ingredient:
        return "ati ampela ayam"
    if "kulit ayam" in ingredient:
        return "kulit ayam"
    if ("daging sapi" in ingredient or ingredient.startswith("sapi")) and not _has_any(ingredient, BEEF_EXCLUDE_TERMS):
        return "daging sapi"
    if "tetelan" in ingredient and "sapi" in ingredient:
        return "daging sapi"
    if "hati sapi" in ingredient:
        return "hati sapi"
    if "lidah sapi" in ingredient:
        return "lidah sapi"
    if "daging kambing" in ingredient or "iga kambing" in ingredient or "kaki kambing" in ingredient:
        return "daging kambing"
    if "bakso" in ingredient:
        return "bakso"
    if "sosis" in ingredient:
        return "sosis"
    if "ham" in ingredient or "pastrami" in ingredient:
        return "daging asap"
    if "kikil" in ingredient:
        return "kikil"

    # Seafood
    if "udang" in ingredient:
        return "udang"
    if "tongkol" in ingredient:
        return "ikan tongkol"
    if "tuna" in ingredient:
        return "ikan tuna"
    if "patin" in ingredient:
        return "ikan patin"
    if "tenggiri" in ingredient:
        return "ikan tenggiri"
    if "gabus" in ingredient:
        return "ikan gabus"
    if "gurame" in ingredient:
        return "ikan gurame"
    if "nila" in ingredient:
        return "ikan nila"
    if "bawal" in ingredient:
        return "ikan bawal"
    if "bandeng" in ingredient:
        return "ikan bandeng"
    if "dencis" in ingredient:
        return "ikan dencis"
    if "ikan" in ingredient:
        return "ikan"

    # Aromatics, spices, leaves
    if "bawang merah" in ingredient and "bawang putih" in ingredient:
        return "bawang merah bawang putih"
    if "bawang merah" in ingredient or "bawang merch" in ingredient:
        return "bawang merah"
    if "bawang putih" in ingredient or "bawang putin" in ingredient:
        return "bawang putih"
    if "bawang bomb" in ingredient or ingredient == "bombay" or "bombai" in ingredient:
        return "bawang bombay"
    if "daun bawang" in ingredient or "bawang daun" in ingredient or "bawangprei" in ingredient or "daun prei" in ingredient or "daon bawang" in ingredient:
        return "daun bawang"
    if "daun kucai" in ingredient:
        return "daun kucai"
    if "seledri" in ingredient or "sledri" in ingredient:
        return "seledri"
    if "daun ketumbar" in ingredient:
        return "daun ketumbar"
    if "daun pisang" in ingredient:
        return "daun pisang"
    if "daun salam" in ingredient or ingredient == "salam":
        return "daun salam"
    if "daun jeruk" in ingredient:
        return "daun jeruk"
    if "daun pandan" in ingredient:
        return "daun pandan"
    if "daun kunyit" in ingredient:
        return "daun kunyit"
    if "daun kemangi" in ingredient or ingredient == "kemangi":
        return "kemangi"
    if "daun pepaya" in ingredient:
        return "daun pepaya"
    if "serai" in ingredient or "sereh" in ingredient or ingredient.startswith("sere"):
        return "serai"
    if "lengkuas" in ingredient or "laos" in ingredient or "laja" in ingredient:
        return "lengkuas"
    if "kunyit" in ingredient or "kunir" in ingredient:
        return "kunyit"
    if "jahe" in ingredient or ingredient == "jae":
        return "jahe"
    if "ketumbar" in ingredient or ingredient == "tumbar":
        return "ketumbar"
    if "kemiri" in ingredient or ingredient == "miri":
        return "kemiri"
    if "merica hitam" in ingredient or "lada hitam" in ingredient or "black pepper" in ingredient:
        return "merica hitam"
    if "merica" in ingredient or "mrica" in ingredient or "lada" in ingredient:
        return "merica"
    if "pala" in ingredient:
        return "pala"
    if "cengkeh" in ingredient or "cengkih" in ingredient or "kengkeh" in ingredient:
        return "cengkeh"
    if "jinten" in ingredient or "jintan" in ingredient or "jintem" in ingredient:
        return "jinten"
    if "kayu manis" in ingredient:
        return "kayu manis"
    if "kapulaga" in ingredient:
        return "kapulaga"
    if "adas" in ingredient:
        return "adas"
    if "kencur" in ingredient:
        return "kencur"
    if "bunga lawang" in ingredient or "pekak" in ingredient:
        return "bunga lawang"
    if "kari" in ingredient:
        return "bumbu kari"

    # Vegetables, fruits, legumes, carbs, fats
    if "cabai" in ingredient or "cabe" in ingredient or "rawit" in ingredient or "lombok" in ingredient or "cili" in ingredient:
        if "hijau" in ingredient or "ijo" in ingredient or "gendot" in ingredient:
            return "cabai hijau"
        if "rawit" in ingredient or "setan" in ingredient or "domba" in ingredient:
            return "cabai rawit"
        if "bubuk" in ingredient or "abon" in ingredient:
            return "cabai bubuk"
        return "cabai"
    if "tomat" in ingredient:
        return "tomat"
    if "kentang" in ingredient:
        return "kentang"
    if "wortel" in ingredient:
        return "wortel"
    if "kol" in ingredient or "kubis" in ingredient:
        return "kol"
    if "sawi" in ingredient or "pokchoi" in ingredient or "pakcoy" in ingredient or "caisim" in ingredient:
        return "sawi"
    if "buncis" in ingredient or "buncir" in ingredient:
        return "buncis"
    if "jagung" in ingredient:
        return "jagung"
    if "timun" in ingredient:
        return "timun"
    if "paprika" in ingredient:
        return "paprika"
    if "labu siam" in ingredient:
        return "labu siam"
    if "labu" in ingredient:
        return "labu"
    if "pare" in ingredient:
        return "pare"
    if "terung" in ingredient:
        return "terung"
    if "petai" in ingredient or "pete" in ingredient:
        return "petai"
    if "lobak" in ingredient:
        return "lobak"
    if "jamur" in ingredient:
        return "jamur"
    if "jeruk nipis" in ingredient:
        return "jeruk nipis"
    if "jeruk limau" in ingredient:
        return "jeruk limau"
    if "lemon" in ingredient:
        return "lemon"
    if "belimbing" in ingredient:
        return "belimbing wuluh"
    if "kismis" in ingredient:
        return "kismis"
    if "kurma" in ingredient:
        return "kurma"
    if "nanas" in ingredient:
        return "nanas"
    if ingredient.startswith("pisang"):
        return "pisang"
    if "tahu" in ingredient or "tofu" in ingredient:
        return "tahu"
    if "tempe" in ingredient:
        return "tempe"
    if "kacang panjang" in ingredient:
        return "kacang panjang"
    if "kacang tanah" in ingredient or "bumbu kacang" in ingredient:
        return "kacang tanah"
    if "kacang koro" in ingredient:
        return "kacang koro"
    if "mie" in ingredient:
        return "mie"
    if "bihun" in ingredient:
        return "bihun"
    if "soun" in ingredient:
        return "soun"
    if "beras" in ingredient:
        return "beras"
    if "roti" in ingredient:
        return "roti"
    if "kulit pangsit" in ingredient or "kulit pangsi" in ingredient:
        return "kulit pangsit"
    if "kulit lumpia" in ingredient:
        return "kulit lumpia"
    if "tepung terigu" in ingredient or ingredient == "terigu":
        return "tepung terigu"
    if "maizena" in ingredient:
        return "tepung maizena"
    if "tepung tapioka" in ingredient or "tapioka" in ingredient:
        return "tepung tapioka"
    if "tepung sagu" in ingredient or ingredient == "sagu":
        return "tepung sagu"
    if "tepung beras" in ingredient:
        return "tepung beras"
    if "tepung panir" in ingredient:
        return "tepung panir"
    if "tepung" in ingredient:
        return "tepung"
    if "emping" in ingredient:
        return "emping"
    if "krupuk" in ingredient or "kerupuk" in ingredient:
        return "kerupuk"
    if "santan" in ingredient or "santen" in ingredient or ingredient in {"kara", "sun kara"}:
        return "santan"
    if "minyak" in ingredient or "olive oil" in ingredient:
        return "minyak"
    if "mentega" in ingredient or "margarin" in ingredient:
        return "mentega"
    if "wijen" in ingredient:
        return "wijen"
    if "susu" in ingredient or "dancow" in ingredient:
        return "susu"
    if "keju" in ingredient or "cheddar" in ingredient or "parmesan" in ingredient:
        return "keju"
    if "cream" in ingredient:
        return "krim"
    if "baking powder" in ingredient:
        return "baking powder"
    if "madu" in ingredient:
        return "madu"
    if "teh" in ingredient:
        return "teh"

    return None


def canonical_ingredient(value: object) -> str:
    ingredient = normalize_ingredient(value)
    if not ingredient:
        return ""

    alias = DIRECT_ALIASES.get(ingredient)
    if alias:
        return alias

    canonical = _canonical_from_keyword(ingredient)
    if canonical:
        return canonical

    return ingredient


def ingredient_variants(value: object) -> list[str]:
    normalized = normalize_ingredient(value)
    canonical = canonical_ingredient(normalized)
    variants = [normalized]
    if canonical and canonical != normalized:
        variants.append(canonical)
    if canonical == "bawang merah bawang putih":
        variants.extend(["bawang merah", "bawang putih"])
    if canonical == "garam gula":
        variants.extend(["garam", "gula"])
    if canonical == "garam merica":
        variants.extend(["garam", "merica"])
    return [variant for variant in variants if variant]


def ingredient_to_token(ingredient: str) -> str:
    return ingredient.replace(" ", "_")


def substitution_group(value: object) -> str:
    ingredient = canonical_ingredient(value)
    normalized = normalize_ingredient(value)

    if not ingredient:
        return "unknown"

    if ingredient == "daun pisang":
        return "wrapper_leaf"

    if ingredient == "air":
        return "basic_liquid"

    if ingredient in {
        "bawang merah",
        "bawang putih",
        "bawang merah bawang putih",
        "bawang bombay",
        "daun bawang",
        "daun kucai",
    }:
        return "allium"

    if ingredient in {"seledri", "daun ketumbar"}:
        return "fresh_herb"

    if ingredient in {
        "jahe",
        "kunyit",
        "lengkuas",
        "serai",
        "ketumbar",
        "kemiri",
        "pala",
        "jinten",
        "cengkeh",
        "kayu manis",
        "kapulaga",
        "adas",
        "kencur",
        "bunga lawang",
        "bumbu kari",
    }:
        return "aromatic_spice"

    if ingredient in {"merica", "merica hitam", "cabai", "cabai hijau", "cabai rawit", "cabai bubuk"}:
        return "hot_spice"

    if ingredient in {
        "daun salam",
        "daun jeruk",
        "daun kunyit",
        "daun pandan",
        "kemangi",
        "daun pepaya",
    }:
        return "aromatic_leaf"

    if ingredient in {"telur", "telur puyuh", "telur asin"}:
        return "egg"

    if ingredient in {"ayam", "daging sapi", "daging kambing"}:
        return "meat"

    if ingredient in {"hati sapi", "lidah sapi", "ati ampela ayam", "kikil", "kulit ayam"}:
        return "offal"

    if ingredient in {"bakso", "sosis", "daging asap", "pangsit ayam", "kekyan panjang"}:
        return "processed_meat"

    if (
        ingredient.startswith("ikan")
        or ingredient in {"udang", "kecap ikan"}
    ):
        return "seafood"

    if ingredient in {"tahu", "tempe"}:
        return "plant_protein"

    if ingredient in {"kacang tanah", "kacang panjang", "kacang koro"}:
        return "legume"

    if ingredient in {"santan", "susu", "keju", "krim"}:
        return "creamy_fat"

    if ingredient in {"minyak", "mentega", "wijen", "minyak wijen", "minyak samin"}:
        return "cooking_fat"

    if ingredient in {"beras", "mie", "bihun", "soun", "roti", "kulit pangsit", "kulit lumpia"}:
        return "carb"

    if ingredient.startswith("tepung") or ingredient in {"maizena", "tapioka", "sagu"}:
        return "flour_starch"

    if ingredient in {"kentang", "jagung", "labu", "labu siam"}:
        return "starchy_vegetable"

    if ingredient in {"tomat", "wortel", "kol", "sawi", "buncis", "timun", "paprika", "pare", "terung", "lobak", "selada"}:
        return "vegetable"

    if ingredient in {"petai"}:
        return "strong_vegetable"

    if ingredient == "jamur":
        return "mushroom"

    if ingredient in {"jeruk nipis", "jeruk limau", "lemon", "belimbing wuluh", "asam jawa", "asam gelugur", "cuka", "acar"}:
        return "acid"

    if ingredient in {"nanas", "pisang", "kismis", "kurma"}:
        return "fruit"

    if ingredient in {
        "kecap",
        "kecap manis",
        "kecap asin",
        "saus tiram",
        "saus teriyaki",
        "saus sambal",
        "saus tomat",
        "saus inggris",
        "saus raja rasa",
        "saus lada hitam",
        "saus asam manis",
        "mayones",
        "saus",
    }:
        return "condiment_sauce"

    if ingredient in {"terasi", "tauco", "taousi", "tempoyak"}:
        return "fermented_condiment"

    if ingredient in {"garam", "gula", "gula merah", "garam gula", "garam merica", "kaldu bubuk", "penyedap rasa", "madu"}:
        return "basic_seasoning"

    if ingredient in {"baking powder"}:
        return "leavening"

    if ingredient in {"teh"}:
        return "beverage"

    if ingredient in {"kulit ikan"}:
        return "seafood"

    if ingredient in {"emping", "kerupuk"}:
        return "crunchy_topping"

    if ingredient in {"bahan", "bumbu"} or _has_any(normalized, COMMON_NOISE_TERMS):
        return "generic_noise"

    return "unknown"
