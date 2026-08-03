# tests/test_verification.py
# v4: verify_answer node'unun GERCEKTEN calistigini kanitlayan test.
#
# Gercek LLM'i "yaniltmaya calismak" yerine (rastgele, API paralı,
# tekrarlanamaz), verify_answer fonksiyonunu DOGRUDAN, kurgulanmis
# bir "yanlis LLM cevabi" ile cagiriyoruz. Bu hem ucretsiz hem de
# her calistirildiginda ayni sonucu verir.

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_graph import verify_answer


def test_verify_answer_passes_through_correct_answer():
    """LLM dogru cevap verdiyse, hicbir sey degismemeli."""
    state = {
        "question": "test",
        "ingredients": ["NIACINAMIDE"],
        "records": [{"ingredient": "NIACINAMIDE", "risk_level": "flagged"}],
        "conflict_note": "",
        "answer": {
            "risk_level": "flagged",
            "mechanism": "B3/flushing risk",
            "summary": "test summary",
            "confidence": "high",
        },
    }
    result = verify_answer(state)
    assert result["answer"]["verified"] is True
    assert result["answer"]["correction_note"] == ""
    assert result["answer"]["risk_level"] == "flagged"


def test_verify_answer_catches_and_corrects_wrong_llm_output():
    """
    Gercek senaryo: LLM, veritabaninda 'restricted' olan bir icerigi
    yanlislikla 'unknown' olarak rapor etti (orn. mechanism metni
    belirsiz oldugu icin LLM riski hafife aldi). Bu, kullanicinin
    aslinda riskli bir icerigi guvenli sanmasina yol acabilecek,
    saglik acisindan onemli bir hata sinifidir.

    verify_answer, bunu LLM'e tekrar SORMADAN, dogrudan veritabani
    kaydiyla karsilastirarak yakalayip duzeltmeli.
    """
    state = {
        "question": "test",
        "ingredients": ["SALICYLIC ACID"],
        "records": [{"ingredient": "SALICYLIC ACID", "risk_level": "restricted"}],
        "conflict_note": "",
        "answer": {
            "risk_level": "unknown",  # <-- LLM'in YANLIS urettigi deger
            "mechanism": "BHA/keratolytic",
            "summary": "yanlislikla hafife alinmis bir ozet",
            "confidence": "high",
        },
    }
    result = verify_answer(state)

    # Duzeltme gerceklesmis olmali
    assert result["answer"]["verified"] is False
    assert result["answer"]["risk_level"] == "restricted"  # veritabani esas alindi
    assert "unknown" in result["answer"]["correction_note"]
    assert "restricted" in result["answer"]["correction_note"]
    # Bir tutarsizlik yakalandigi icin guven seviyesi dusurulmeli
    assert result["answer"]["confidence"] == "medium"


def test_verify_answer_picks_most_severe_when_multiple_records():
    """
    Iki ingredient var, biri 'flagged' biri 'banned'. LLM sadece
    'flagged' demis (daha hafif olani rapor etmis). Dogrulama,
    en ciddi seviyeyi (banned) esas almali.
    """
    state = {
        "question": "test",
        "ingredients": ["A", "B"],
        "records": [
            {"ingredient": "A", "risk_level": "flagged"},
            {"ingredient": "B", "risk_level": "banned"},
        ],
        "conflict_note": "conflict detected",
        "answer": {
            "risk_level": "flagged",  # <-- eksik, daha ciddi olani (banned) kacirmis
            "mechanism": "mixed",
            "summary": "test",
            "confidence": "high",
        },
    }
    result = verify_answer(state)

    assert result["answer"]["verified"] is False
    assert result["answer"]["risk_level"] == "banned"


def test_verify_answer_no_records_marks_verified_trivially():
    """Kayit yoksa (unknown durumu), dogrulama otomatik gecerli sayilir."""
    state = {
        "question": "alakasiz soru",
        "ingredients": [],
        "records": [],
        "conflict_note": "",
        "answer": {
            "risk_level": "unknown",
            "mechanism": "N/A",
            "summary": "eslesme yok",
            "confidence": "low",
        },
    }
    result = verify_answer(state)
    assert result["answer"]["verified"] is True
