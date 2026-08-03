# agent_graph.py
#
# LangGraph orchestration for multi-ingredient queries.
# v3 flow: question -> retrieval -> LLM -> answer (single step).
# This adds: ingredient count check -> conditional conflict check
# -> answer -> deterministic verification against the DB.

import os
import json
from typing import TypedDict
import requests
import pandas as pd

from rag_context import (
    ingredient_db_v2,
    extract_ingredients_from_text,
    extract_conditions_from_text,
)

from langgraph.graph import StateGraph, END

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
EMBED_MODEL_CHAT = "gemini-2.5-flash"

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "risk_level": {"type": "string", "enum": ["banned", "restricted", "flagged", "unknown"]},
        "mechanism": {"type": "string"},
        "summary": {"type": "string"},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "conflict_warning": {"type": "string"},
    },
    "required": ["risk_level", "mechanism", "summary", "confidence"],
}

SYSTEM_PROMPT = (
    "Sen bir kozmetik icerik analiz asistanisin. GOREVIN SADECE:\n"
    "asagida verilen KAYITLARI ve varsa CAKISMA UYARISINI, belirtilen JSON\n"
    "semasina uygun sekilde ozetlemek.\n"
    "KESINLIKLE YAPMAYACAKLARIN:\n"
    "- Kayitlarda olmayan bir risk_level veya mekanizma UYDURMA.\n"
    "- Kayitlarda eslesme yoksa, risk_level'i 'unknown' yap.\n"
    "- Tibbi tavsiye verme; bu bir bilgi ozeti, tani/tedavi degil."
)


# State tanimi
class AgentState(TypedDict):
    question: str
    ingredients: list[str]
    records: list[dict]
    conflict_note: str
    answer: dict


# Node 1: sorudaki ingredient'lari tespit et
def identify_ingredients(state: AgentState) -> AgentState:
    ingredients = extract_ingredients_from_text(state["question"])
    return {**state, "ingredients": ingredients}


# Node 2: her ingredient icin kaydi getir
def lookup_records(state: AgentState) -> AgentState:
    ingredients = state["ingredients"]
    if not ingredients:
        return {**state, "records": []}

    df = ingredient_db_v2[ingredient_db_v2["ingredient"].isin(ingredients)]
    records = df.to_dict("records")
    return {**state, "records": records}


# Yonlendirme: 2+ ingredient varsa cakisma kontroluune git
def route_after_lookup(state: AgentState) -> str:
    if len(state["ingredients"]) >= 2:
        return "check_conflict"
    return "final_answer"


# Node 3 (kosullu): cakisma var mi kontrol et
def check_conflict(state: AgentState) -> AgentState:
    records = state["records"]
    risky_levels = {"banned", "restricted", "flagged"}

    risky_records = [r for r in records if r.get("risk_level") in risky_levels]

    if len(risky_records) >= 2:
        names = ", ".join(r["ingredient"] for r in risky_records)
        note = (
            f"UYARI: Bu sorguda birden fazla riskli icerik ayni anda geciyor "
            f"({names}). Her biri ayri ayri degerlendirilmeli, birlikte "
            f"kullanimlarinin ek bir riski literaturde ayrica belirtilmedigi "
            f"surece varsayilmamali."
        )
    else:
        note = ""

    return {**state, "conflict_note": note}


# Node 4: final structured LLM cagrisi
def final_answer(state: AgentState) -> AgentState:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY tanimli degil.")

    records = state["records"]
    conflict_note = state.get("conflict_note", "")

    if not records:
        context_text = "Soruda gecen icerik veritabaninda bulunamadi."
    else:
        lines = ["ICERIK KAYITLARI:"]
        for r in records:
            lines.append(
                f"- {r['ingredient']} | risk_level: {r.get('risk_level', 'unknown')} | "
                f"mekanizma: {r.get('mechanism', '-')}"
            )
        context_text = "\n".join(lines)
        if conflict_note:
            context_text += f"\n\n{conflict_note}"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{EMBED_MODEL_CHAT}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {"role": "user", "parts": [{"text": f"Soru: {state['question']}\n\n{context_text}"}]}
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": RESPONSE_SCHEMA,
        },
    }
    resp = requests.post(url, headers={"x-goog-api-key": GEMINI_API_KEY}, json=payload, timeout=30)
    resp.raise_for_status()
    text_out = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    answer = json.loads(text_out)

    return {**state, "answer": answer}


# Node 5: verification against the DB, not a second LLM call.
# A second LLM judging the first doesn't fix hallucination -- it just
# adds a second chance to hallucinate. This compares the LLM's
# risk_level directly against the ground-truth record.

RISK_SEVERITY = {"banned": 3, "restricted": 2, "flagged": 1, "unknown": 0}


def verify_answer(state: AgentState) -> AgentState:
    records = state["records"]
    answer = dict(state["answer"])

    if not records:
        answer["verified"] = True
        answer["correction_note"] = ""
        return {**state, "answer": answer}

    # highest-severity risk_level among matched records is ground truth
    true_risk = max(
        (r.get("risk_level", "unknown") for r in records),
        key=lambda level: RISK_SEVERITY.get(level, 0),
    )

    llm_risk = answer.get("risk_level", "unknown")

    if llm_risk == true_risk:
        answer["verified"] = True
        answer["correction_note"] = ""
    else:
        # LLM was wrong -- override with DB value, don't re-query the LLM.
        # tutarsizlik tespit edildi.
        answer["verified"] = False
        answer["correction_note"] = (
            f"LLM '{llm_risk}' dedi, veritabani '{true_risk}' diyor. "
            f"Veritabani esas alinarak duzeltildi."
        )
        answer["risk_level"] = true_risk
        answer["confidence"] = "medium"

    return {**state, "answer": answer}



def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("identify_ingredients", identify_ingredients)
    graph.add_node("lookup_records", lookup_records)
    graph.add_node("check_conflict", check_conflict)
    graph.add_node("final_answer", final_answer)
    graph.add_node("verify_answer", verify_answer)

    graph.set_entry_point("identify_ingredients")
    graph.add_edge("identify_ingredients", "lookup_records")
    graph.add_conditional_edges(
        "lookup_records",
        route_after_lookup,
        {"check_conflict": "check_conflict", "final_answer": "final_answer"},
    )
    graph.add_edge("check_conflict", "final_answer")
    graph.add_edge("final_answer", "verify_answer")
    graph.add_edge("verify_answer", END)

    return graph.compile()


def ask_agent(question: str) -> dict:
    app = build_graph()
    result = app.invoke({"question": question, "ingredients": [], "records": [], "conflict_note": "", "answer": {}})
    return result["answer"]


if __name__ == "__main__":
    # Tek ingredient - conflict node atlanmali
    print("=== Tek ingredient testi ===")
    print(json.dumps(ask_agent("niacinamide rozasea icin guvenli mi"), indent=2, ensure_ascii=False))

    # Iki ingredient - conflict node calismali
    print("\n=== Iki ingredient testi (cakisma beklenir) ===")
    print(json.dumps(
        ask_agent("salicylic acid ve retinyl palmitate'i birlikte kullanabilir miyim"),
        indent=2, ensure_ascii=False
    ))
