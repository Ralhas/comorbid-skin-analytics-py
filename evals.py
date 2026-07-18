import json
import time
from structured_llm import ask_structured

TEST_CASES = [
    {"question": "diethyl phthalate guvenli mi", "expected_risk_level": "flagged"},
    {"question": "retinyl palmitate kullanabilir miyim", "expected_risk_level": "restricted"},
    {"question": "homosalate icerikli gunes kremi riskli mi", "expected_risk_level": "restricted"},
    {"question": "propylparaben endokrin sorunu var mi", "expected_risk_level": "flagged"},
    {"question": "limonene alerjik mi", "expected_risk_level": "flagged"},
    {"question": "salisilik asit rozasea icin guvenli mi", "expected_risk_level": "restricted"},
    {"question": "niacinamide rozasea icin iyi mi", "expected_risk_level": "flagged"},
    {"question": "hic alakasiz bir soru: bugun hava nasil", "expected_risk_level": "unknown"},
]


def run_evals():
    results = []
    for case in TEST_CASES:
        output = None
        for attempt in range(4):
            try:
                output = ask_structured(case["question"], provider="gemini")
                break
            except Exception as e:
                if "429" in str(e) and attempt < 3:
                    wait = 15 * (attempt + 1)
                    print(f"  Rate limit, {wait}s bekleniyor...")
                    time.sleep(wait)
                    continue
                output = {"risk_level": "ERROR", "summary": str(e)}
                break

        correct = output["risk_level"] == case["expected_risk_level"]
        results.append({
            "question": case["question"],
            "expected": case["expected_risk_level"],
            "got": output["risk_level"],
            "correct": correct,
            "summary": output.get("summary", ""),
        })
        time.sleep(3)

    return results


def print_report(results):
    correct_count = sum(r["correct"] for r in results)
    total = len(results)

    print(f"\n{'='*60}")
    print(f"EVAL RAPORU: {correct_count}/{total} dogru ({correct_count/total*100:.1f}%)")
    print(f"{'='*60}\n")

    for r in results:
        status = "OK" if r["correct"] else "HATA"
        print(f"{status} \"{r['question']}\"")
        print(f"    beklenen: {r['expected']} | gelen: {r['got']}")
        if not r["correct"]:
            print(f"    ozet: {r['summary']}")
        print()


if __name__ == "__main__":
    results = run_evals()
    print_report(results)

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Detayli sonuclar eval_results.json dosyasina kaydedildi.")
