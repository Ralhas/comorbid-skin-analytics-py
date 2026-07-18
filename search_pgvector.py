import os
import sys
import requests
from sqlalchemy import create_engine, text

DB_URL = "postgresql://postgres:comorbid123@localhost:5432/comorbid"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
EMBED_MODEL = "gemini-embedding-001"

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY tanimli degil. export GEMINI_API_KEY=... yap.")

engine = create_engine(DB_URL)


def get_embedding(text_input):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:embedContent"
    payload = {
        "content": {"parts": [{"text": text_input}]},
        "outputDimensionality": 768,
    }
    resp = requests.post(url, headers={"x-goog-api-key": GEMINI_API_KEY}, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["embedding"]["values"]


def semantic_search(query, top_k=3):
    query_emb = get_embedding(query)
    emb_str = str(query_emb)

    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT ingredient, risk_level, mechanism, conditions,
                       embedding <=> :query_emb AS distance
                FROM ingredient_embeddings
                ORDER BY distance ASC
                LIMIT :top_k
            """),
            {"query_emb": emb_str, "top_k": top_k},
        ).fetchall()

    print(f"\nSoru: \"{query}\"")
    print(f"En yakin {top_k} sonuc (mesafe kucuk = daha benzer):\n")
    for r in rows:
        similarity_pct = (1 - r.distance) * 100
        print(f"  [{similarity_pct:.1f}% benzer] {r.ingredient} | risk: {r.risk_level}")
        print(f"      mekanizma: {r.mechanism}")
        print(f"      durumlar: {r.conditions}\n")


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "B3 vitamini rozasea icin iyi mi"
    semantic_search(query)
