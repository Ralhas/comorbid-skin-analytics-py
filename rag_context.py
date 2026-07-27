# ── rag_context.py ──────────────────────────────────────────────
# v3.0 — RAG Context Layer (Python port of R/09_rag_context.R)
#
# TASARIM PRENSIBI (R versiyonundan degismedi):
#   Retrieval  = tamamen deterministik Python kodu (bu dosya).
#   LLM        = SADECE context'i dogal dile ceviren son katman.
#   LLM hicbir zaman risk_level / risk_score URETMEZ, KARAR VERMEZ.
#
# Bu dosya R/09_rag_context.R'nin davranissal olarak birebir
# esdegeridir. Fonksiyon isimleri ve mantik bilerek ayni tutuldu ki
# iki versiyon karsilastirilabilir ve dogrulanabilir olsun.

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv
load_dotenv()


DATA_DIR = Path(__file__).parent / "data"

ingredient_db_v2 = pd.read_parquet(DATA_DIR / "ingredient_db_v2.parquet")
condition_map = pd.read_parquet(DATA_DIR / "condition_map.parquet")
condition_parent = pd.read_parquet(DATA_DIR / "condition_parent_map.parquet")
inci_aliases = pd.read_parquet(DATA_DIR / "inci_aliases.parquet")


# ── 1. Soru metninden cilt durumlarini cikar ───────────────────
def extract_conditions_from_text(text: str) -> list[str]:
    q = text.lower()

    mask = condition_map["condition_tr"].str.lower().apply(lambda v: v in q) | \
        condition_map["condition_en"].str.lower().apply(lambda v: v in q)
    matched_en = condition_map.loc[mask, "condition_en"].tolist()

    via_parent = condition_parent.loc[
        condition_parent["condition_en"].isin(matched_en), "maps_to_condition"
    ].tolist()

    seen = []
    for c in matched_en + via_parent:
        if c not in seen:
            seen.append(c)
    return seen


# ── 2. Soru metninden icerik adlarini cikar ────────────────────
def extract_ingredients_from_text(text: str) -> list[str]:
    q = text.lower()

    hits_alias = inci_aliases.loc[
        inci_aliases["alias"].str.lower().apply(lambda v: v in q), "inci_name"
    ].tolist()

    hits_direct = ingredient_db_v2.loc[
        ingredient_db_v2["ingredient"].str.lower().apply(lambda v: v in q), "ingredient"
    ].tolist()

    seen = []
    for i in hits_alias + hits_direct:
        if i not in seen:
            seen.append(i)
    return seen


# ── 3. Yapilandirilmis context olustur ─────────────────────────
@dataclass
class QueryContext:
    question: str
    matched_conditions: list[str]
    matched_ingredients: list[str]
    ingredient_hits: pd.DataFrame
    condition_hits: pd.DataFrame
    has_context: bool = field(init=False)

    def __post_init__(self):
        self.has_context = len(self.ingredient_hits) > 0 or len(self.condition_hits) > 0


def build_query_context(question: str) -> QueryContext:
    conds = extract_conditions_from_text(question)
    ingredients = extract_ingredients_from_text(question)

    # 3a. Soruda gecen spesifik icerikler icin dogrudan kayit
    if ingredients:
        ingredient_hits = ingredient_db_v2[ingredient_db_v2["ingredient"].isin(ingredients)]
    else:
        ingredient_hits = ingredient_db_v2.iloc[0:0]

    # 3b. Durum bazli TAM liste SADECE soruda spesifik bir icerik
    # GECMIYORSA calisir (R'daki ayni mantik, ayni yorum).
    if conds and not ingredients:
        df = ingredient_db_v2[ingredient_db_v2["conditions"].notna()].copy()
        df["cond_list"] = df["conditions"].str.split(";")
        df = df.explode("cond_list").rename(columns={"cond_list": "condition"})
        df = df[
            df["condition"].isin(conds) & df["risk_level"].isin(["banned", "restricted"])
        ]
        condition_hits = df.sort_values("risk_score", ascending=False).head(25)
    else:
        condition_hits = ingredient_db_v2.iloc[0:0]

    return QueryContext(
        question=question,
        matched_conditions=conds,
        matched_ingredients=ingredients,
        ingredient_hits=ingredient_hits,
        condition_hits=condition_hits,
    )


# ── 4. Context'i LLM icin duz metne cevir ──────────────────────
def format_context_for_llm(ctx: QueryContext) -> str:
    if not ctx.has_context:
        return (
            "Soruda gecen icerik veya cilt durumu veritabaninda bulunamadi. "
            "Bu net sekilde belirtilmeli: 'literatur veritabaninda eslesme yok'."
        )

    parts = []

    if len(ctx.ingredient_hits) > 0:
        lines = ["ICERIK KAYITLARI:"]
        for _, row in ctx.ingredient_hits.iterrows():
            lines.append(
                f"- {row['ingredient']} | risk_level: {row['risk_level'] or 'unknown'} | "
                f"mekanizma: {row['mechanism'] or 'literaturde tanimlanmamis'} | "
                f"fonksiyonel kategori: {row['functional_category'] or 'unclassified'} | "
                f"ilgili durumlar: {row['conditions'] or 'belirtilmemis'}"
            )
        parts.append("\n".join(lines))

    if len(ctx.condition_hits) > 0:
        lines = ["\nCILT DURUMU BAZLI KAYITLAR:"]
        dedup = ctx.condition_hits.drop_duplicates(
            subset=["ingredient", "condition", "risk_level", "risk_score", "mechanism"]
        )
        for _, row in dedup.iterrows():
            lines.append(
                f"- [{row['condition']}] {row['ingredient']} | "
                f"risk_level: {row['risk_level']} (skor {int(row['risk_score'])}) | "
                f"mekanizma: {row['mechanism'] or '-'}"
            )
        parts.append("\n".join(lines))

    return "\n".join(parts)


# ── 5. LLM cagrisi (Gemini API) — SADECE yorumlama, karar verme YOK ─
SYSTEM_PROMPT = (
    "Sen bir kozmetik icerik analiz asistanisin. GOREVIN SADECE:\n"
    "asagida verilen KAYITLARI dogal, anlasilir bir dile cevirmek.\n"
    "KESINLIKLE YAPMAYACAKLARIN:\n"
    "- Kayitlarda olmayan bir risk_level veya mekanizma UYDURMA.\n"
    "- Kayitlarda olmayan bir icerik hakkinda yorum yapma.\n"
    "- Tibbi tavsiye verme; bu bir bilgi ozeti, tani/tedavi degil.\n"
    "Eger kayitlarda 'eslesme bulunamadi' yaziyorsa, bunu acikca soyle,\n"
    "kendi bilginle bosluk doldurmaya calisma."
)


def ask_llm_interpreter(question: str, api_key: str | None = None,
                         model: str = "gemini-2.5-flash") -> str:
    import requests

    api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY ortam degiskeni tanimli degil. "
            "export GEMINI_API_KEY=xxx ile ayarlayabilirsin."
        )

    ctx = build_query_context(question)
    ctx_text = format_context_for_llm(ctx)

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"Soru: {question}\n\nKAYITLAR:\n{ctx_text}"}],
            }
        ],
    }
    resp = requests.post(url, headers={"x-goog-api-key": api_key}, json=payload, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    return body["candidates"][0]["content"]["parts"][0]["text"]


if __name__ == "__main__":
    ctx = build_query_context("rozasea icin niacinamide guvenli mi")
    print(format_context_for_llm(ctx))
