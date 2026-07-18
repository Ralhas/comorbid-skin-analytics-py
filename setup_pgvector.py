
# ── setup_pgvector.py ───────────────────────────────────────────
# v3 Aşama 2: pgvector kurulumu
#
# Bu script BİR KEZ çalıştırılır:
#   1. "ingredient_embeddings" tablosunu oluşturur (pgvector destekli)
#   2. ingredient_db_v2.parquet'teki her satırın "mechanism" alanını
#      Gemini embedding API'sine gönderir
#   3. Sonuçları PostgreSQL'e yazar
#
# Amaç: kullanıcı "B3 vitamini" gibi dolaylı ifadeyle sorduğunda da
# doğru ingredient kaydını bulabilmek (semantic search).
 
import os
import time
import requests
import pandas as pd
from sqlalchemy import create_engine, text
 
DB_URL = "postgresql://postgres:comorbid123@localhost:5432/comorbid"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
EMBED_MODEL = "gemini-embedding-001"  # Guncel Gemini embedding modeli (2026)
 
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY tanımlı değil. export GEMINI_API_KEY=... yap.")
 
engine = create_engine(DB_URL)
 
 
# ── 1. Tabloyu oluştur ──────────────────────────────────────────
def create_schema():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ingredient_embeddings (
                id SERIAL PRIMARY KEY,
                ingredient TEXT NOT NULL,
                risk_level TEXT,
                risk_score FLOAT,
                mechanism TEXT,
                conditions TEXT,
                embedding vector(768)
            );
        """))
        conn.commit()
    print("✓ Şema oluşturuldu (ingredient_embeddings tablosu + vector extension)")
 
 
# ── 2. Gemini'den embedding al ──────────────────────────────────
def get_embedding(text_input: str, max_retries: int = 5) -> list[float]:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL}:embedContent"
    payload = {
        "content": {"parts": [{"text": text_input}]},
        "outputDimensionality": 768,  # semamizdaki vector(768) ile eslessin diye
    }
    for attempt in range(max_retries):
        resp = requests.post(url, headers={"x-goog-api-key": GEMINI_API_KEY}, json=payload, timeout=30)
        if resp.status_code == 429:
            wait = 15 * (attempt + 1)  # 15s, 30s, 45s, 60s, 75s
            print(f"    Rate limit (429), {wait}s bekleniyor...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]
    raise RuntimeError("Rate limit asilamadi, cok fazla deneme yapildi.")
 
 
# ── 3. Veriyi işle ve yükle ──────────────────────────────────────
def load_data():
    df = pd.read_parquet("data/ingredient_db_v2.parquet")
    df = df[df["mechanism"].notna() & (df["mechanism"].str.strip() != "")]
 
    with engine.connect() as conn:
        already_done = {
            row[0] for row in conn.execute(text("SELECT ingredient FROM ingredient_embeddings"))
        }
 
    remaining = df[~df["ingredient"].isin(already_done)]
    print(f"{len(df)} toplam satır, {len(already_done)} zaten islenmis, {len(remaining)} kaldi...")
 
    with engine.connect() as conn:
        for idx, (i, row) in enumerate(remaining.iterrows(), start=1):
            emb = get_embedding(row["mechanism"])
            conn.execute(
                text("""
                    INSERT INTO ingredient_embeddings
                    (ingredient, risk_level, risk_score, mechanism, conditions, embedding)
                    VALUES (:ingredient, :risk_level, :risk_score, :mechanism, :conditions, :embedding)
                """),
                {
                    "ingredient": row["ingredient"],
                    "risk_level": row["risk_level"],
                    "risk_score": float(row["risk_score"]) if pd.notna(row["risk_score"]) else None,
                    "mechanism": row["mechanism"],
                    "conditions": row["conditions"],
                    "embedding": str(emb),
                },
            )
            conn.commit()  # her satirda commit -> yarida kesilirse ilerleme kaybolmaz
            done_count = len(already_done) + idx
            if done_count % 10 == 0:
                print(f"  {done_count}/{len(df)} tamamlandi...")
            time.sleep(1.2)  # ucretsiz kota rate limitine takilmamak icin
 
    print("✓ Tüm veriler embed edildi ve yüklendi")
 
 
if __name__ == "__main__":
    create_schema()
    load_data()
    print("\nBitti. Şimdi search_pgvector.py ile semantic arama test edilebilir.")
 