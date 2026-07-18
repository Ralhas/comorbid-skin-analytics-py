import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rag_context import (
    extract_conditions_from_text,
    extract_ingredients_from_text,
    build_query_context,
    format_context_for_llm,
)


def test_extract_conditions_finds_rosacea():
    conditions = extract_conditions_from_text("rozasea icin niacinamide guvenli mi")
    assert "rosacea" in conditions


def test_extract_ingredients_finds_niacinamide():
    ingredients = extract_ingredients_from_text("rozasea icin niacinamide guvenli mi")
    assert any("NIACINAMIDE" in i.upper() for i in ingredients)


def test_extract_ingredients_returns_empty_for_unknown():
    ingredients = extract_ingredients_from_text("bugun hava nasil")
    assert ingredients == []


def test_build_query_context_has_context_for_known_ingredient():
    ctx = build_query_context("niacinamide rozasea icin iyi mi")
    assert ctx.has_context is True


def test_build_query_context_no_context_for_irrelevant_question():
    ctx = build_query_context("bugun hava nasil")
    assert ctx.has_context is False


def test_format_context_reports_no_match_when_empty():
    ctx = build_query_context("bugun hava nasil")
    output = format_context_for_llm(ctx)
    assert "bulunamadi" in output.lower()


def test_salicylic_acid_flagged_for_rosacea_and_atopic_dermatitis():
    ingredients = extract_ingredients_from_text("salisilik asit rozasea icin guvenli mi")
    assert any("SALICYLIC ACID" in i.upper() for i in ingredients)
