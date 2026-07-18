import os
import json
import requests
from rag_context import build_query_context, format_context_for_llm

SYSTEM_PROMPT = (
    "Sen bir kozmetik icerik analiz asistanisin. GOREVIN SADECE:\n"
    "asagida verilen KAYITLARI, belirtilen JSON semasina uygun sekilde ozetlemek.\n"
    "KESINLIKLE YAPMAYACAKLARIN:\n"
    "- Kayitlarda olmayan bir risk_level veya mekanizma UYDURMA.\n"
    "- Kayitlarda eslesme yoksa, risk_level'i 'unknown' yap ve confidence'i 'low' yap.\n"
    "- Tibbi tavsiye verme; bu bir bilgi ozeti, tani/tedavi degil."
)

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "risk_level": {"type": "string", "enum": ["banned", "restricted", "flagged", "unknown"]},
        "mechanism": {"type": "string"},
        "summary": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["risk_level", "mechanism", "summary", "confidence"],
}


def _build_prompt(question):
    ctx = build_query_context(question)
    ctx_text = format_context_for_llm(ctx)
    return f"Soru: {question}\n\nKAYITLAR:\n{ctx_text}"


def _ask_gemini(question, model="gemini-2.5-flash"):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY tanimli degil.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": _build_prompt(question)}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        },
    }
    resp = requests.post(url, headers={"x-goog-api-key": api_key}, json=payload, timeout=30)
    resp.raise_for_status()
    text_out = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text_out)


def _ask_openai(question):
    raise NotImplementedError("OpenAI API anahtari henuz tanimli degil.")


def _ask_anthropic(question):
    raise NotImplementedError("Anthropic API anahtari henuz tanimli degil.")


PROVIDERS = {
    "gemini": _ask_gemini,
    "openai": _ask_openai,
    "anthropic": _ask_anthropic,
}


def ask_structured(question, provider="gemini"):
    if provider not in PROVIDERS:
        raise ValueError(f"Bilinmeyen provider: {provider}")
    return PROVIDERS[provider](question)


if __name__ == "__main__":
    result = ask_structured("rozasea icin niacinamide guvenli mi", provider="gemini")
    print(json.dumps(result, indent=2, ensure_ascii=False))
