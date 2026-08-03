# compare_v3_v4.py
# v4'un tezi: "Orkestrasyon eklemek karmasikligi artirir -- ama bu
# karmasiklik gercekten daha guvenilir bir sistem uretiyor mu?"
#
# Bu script, AYNI soru setini hem v3'un tek-adimli sistemine
# (structured_llm.ask_structured) hem v4'un agent'ina (agent_graph.
# ask_agent) sorar ve su dortunu karsilastirir:
#   1. Dogruluk (ground truth ile eslesme)
#   2. API cagri sayisi (maliyet gostergesi)
#   3. Gecikme (latency)
#   4. Cakisma tespiti (SADECE v4'un yapabildigi bir sey)

import time
import json
from structured_llm import ask_structured
from agent_graph import ask_agent


def call_with_retry(func, *args, max_retries=4, **kwargs):
    """
    429 (rate limit) hatalarinda otomatik bekleyip tekrar dener.
    Bekleme suresini ayri dondurur ki gecikme olcumune karismasin --
    yoksa "v4 daha yavas" sonucu aslinda "rate limit'e denk geldi"
    demek olabilir, bu yaniltici olur.
    """
    total_wait = 0.0
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            return result, total_wait
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 20 * (attempt + 1)
                print(f"    Rate limit, {wait}s bekleniyor (gecikme olcumune dahil edilmeyecek)...")
                time.sleep(wait)
                total_wait += wait
                continue
            raise


# Ayni eval setinden (evals.py) + coklu-ingredient senaryolari eklendi.
TEST_CASES = [
    {"question": "diethyl phthalate guvenli mi", "expected_risk_level": "flagged", "multi": False},
    {"question": "retinyl palmitate kullanabilir miyim", "expected_risk_level": "restricted", "multi": False},
    {"question": "propylparaben endokrin sorunu var mi", "expected_risk_level": "flagged", "multi": False},
    {"question": "niacinamide rozasea icin iyi mi", "expected_risk_level": "flagged", "multi": False},
    {"question": "hic alakasiz bir soru: bugun hava nasil", "expected_risk_level": "unknown", "multi": False},
    # Coklu ingredient - sadece v4'un cakisma tespiti yapabilecegi senaryolar
    {"question": "salicylic acid ve retinyl palmitate'i birlikte kullanabilir miyim",
     "expected_risk_level": "restricted", "multi": True},
    {"question": "diethyl phthalate ve propylparaben ayni urunde olursa sorun olur mu",
     "expected_risk_level": "flagged", "multi": True},
]


def run_comparison():
    results = {"v3": [], "v4": []}

    for case in TEST_CASES:
        q = case["question"]

        # v3: tek adimli, dogrudan structured_llm cagrisi
        t0 = time.time()
        try:
            v3_answer, v3_wait = call_with_retry(ask_structured, q, provider="gemini")
            v3_calls = 1  # tek LLM cagrisi
        except Exception as e:
            v3_answer = {"risk_level": "ERROR", "summary": str(e)}
            v3_wait = 0
            v3_calls = 1
        v3_latency_raw = time.time() - t0
        v3_latency_clean = round(v3_latency_raw - v3_wait, 2)

        results["v3"].append({
            "question": q,
            "expected": case["expected_risk_level"],
            "got": v3_answer.get("risk_level"),
            "correct": v3_answer.get("risk_level") == case["expected_risk_level"],
            "latency_sec": v3_latency_clean,
            "rate_limit_wait_sec": v3_wait,
            "api_calls": v3_calls,
            "conflict_detected": False,  # v3 bunu hic yapamaz
        })

        time.sleep(10)  # rate limit'e takilmamak icin daha da arttirildi

        # v4: coklu adimli agent
        t0 = time.time()
        try:
            v4_answer, v4_wait = call_with_retry(ask_agent, q)
            v4_calls = 1
        except Exception as e:
            v4_answer = {"risk_level": "ERROR", "summary": str(e)}
            v4_wait = 0
            v4_calls = 1
        v4_latency_raw = time.time() - t0
        v4_latency_clean = round(v4_latency_raw - v4_wait, 2)

        results["v4"].append({
            "question": q,
            "expected": case["expected_risk_level"],
            "got": v4_answer.get("risk_level"),
            "correct": v4_answer.get("risk_level") == case["expected_risk_level"],
            "latency_sec": v4_latency_clean,
            "rate_limit_wait_sec": v4_wait,
            "api_calls": v4_calls,
            "conflict_detected": bool(v4_answer.get("conflict_warning")),
            "verified": v4_answer.get("verified"),
        })

        time.sleep(10)

    return results


def print_report(results):
    print(f"\n{'='*70}")
    print("v3 (tek adimli) vs v4 (LangGraph agent) karsilastirmasi")
    print(f"{'='*70}\n")

    v3_correct = sum(r["correct"] for r in results["v3"])
    v4_correct = sum(r["correct"] for r in results["v4"])
    v3_avg_latency = sum(r["latency_sec"] for r in results["v3"]) / len(results["v3"])
    v4_avg_latency = sum(r["latency_sec"] for r in results["v4"]) / len(results["v4"])
    v4_conflicts_found = sum(r["conflict_detected"] for r in results["v4"])
    expected_conflicts = sum(1 for c in TEST_CASES if c["multi"])

    print(f"Dogruluk:        v3 {v3_correct}/{len(TEST_CASES)}  |  v4 {v4_correct}/{len(TEST_CASES)}")
    print(f"Ort. gecikme:    v3 {v3_avg_latency:.2f}s  |  v4 {v4_avg_latency:.2f}s")
    print(f"(gecikme, rate-limit bekleme suresi HARIC olacak sekilde hesaplandi)")
    print(f"Cakisma tespiti: v3 0/{expected_conflicts} (yapamaz)  |  v4 {v4_conflicts_found}/{expected_conflicts}")
    print()

    print("Soru bazinda detay:")
    for v3r, v4r in zip(results["v3"], results["v4"]):
        print(f"\n  \"{v3r['question'][:50]}...\"" if len(v3r['question']) > 50 else f"\n  \"{v3r['question']}\"")
        print(f"    v3: {v3r['got']} ({'OK' if v3r['correct'] else 'YANLIS'}, {v3r['latency_sec']}s)")
        print(f"    v4: {v4r['got']} ({'OK' if v4r['correct'] else 'YANLIS'}, {v4r['latency_sec']}s, "
              f"verified={v4r['verified']}, conflict={v4r['conflict_detected']})")

    with open("comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nDetayli sonuclar comparison_results.json dosyasina kaydedildi.")


if __name__ == "__main__":
    results = run_comparison()
    print_report(results)
