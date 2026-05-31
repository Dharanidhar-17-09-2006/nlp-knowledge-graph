"""
evaluate_hybrid.py
==================
Compares REBEL, Groq, and Hybrid extraction using:
  1. Exact F1  — canonicalized (subj, rel, obj) string match
  2. Partial F1 — subject/object substring match with relation synonym check

Same evaluation methodology as Test_pipeline.py for fair comparison.

Run:
  python evaluate_hybrid.py
  python evaluate_hybrid.py --verbose
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from typing import List, Tuple, Dict

from dotenv import load_dotenv
load_dotenv()

# ══════════════════════════════════════════════════════════════════
# EVALUATION DATASET — 30 sentences with gold triples
# ══════════════════════════════════════════════════════════════════

EVAL_DATA = [
    {"text": "Albert Einstein was born in Ulm, Germany.",
     "gold": [("albert einstein", "born in", "ulm"), ("albert einstein", "born in", "germany")]},
    {"text": "Marie Curie was born in Warsaw.",
     "gold": [("marie curie", "born in", "warsaw")]},
    {"text": "Google was founded by Larry Page and Sergey Brin.",
     "gold": [("google", "founded by", "larry page"), ("google", "founded by", "sergey brin")]},
    {"text": "Elon Musk founded SpaceX in California.",
     "gold": [("elon musk", "founded", "spacex"), ("spacex", "located in", "california")]},
    {"text": "SpaceX is headquartered in Hawthorne, California.",
     "gold": [("spacex", "headquartered in", "hawthorne"), ("spacex", "located in", "california")]},
    {"text": "Marie Curie studied at the University of Paris.",
     "gold": [("marie curie", "studied at", "university of paris")]},
    {"text": "Apple was founded by Steve Jobs.",
     "gold": [("apple", "founded by", "steve jobs")]},
    {"text": "The Eiffel Tower is located in Paris, France.",
     "gold": [("eiffel tower", "located in", "paris"), ("eiffel tower", "located in", "france")]},
    {"text": "Virat Kohli is a member of the Royal Challengers Bangalore.",
     "gold": [("virat kohli", "member of", "royal challengers bangalore")]},
    {"text": "Microsoft was founded by Bill Gates and Paul Allen.",
     "gold": [("microsoft", "founded by", "bill gates"), ("microsoft", "founded by", "paul allen")]},
    {"text": "Amazon is headquartered in Seattle.",
     "gold": [("amazon", "headquartered in", "seattle")]},
    {"text": "Barack Obama was born in Hawaii.",
     "gold": [("barack obama", "born in", "hawaii")]},
    {"text": "Tesla was founded by Elon Musk and Martin Eberhard.",
     "gold": [("tesla", "founded by", "elon musk"), ("tesla", "founded by", "martin eberhard")]},
    {"text": "Newton was born in Lincolnshire, England.",
     "gold": [("newton", "born in", "lincolnshire"), ("newton", "born in", "england")]},
    {"text": "Facebook is headquartered in Menlo Park.",
     "gold": [("facebook", "headquartered in", "menlo park")]},
    {"text": "Srishwan is a student at IIT Kharagpur.",
     "gold": [("srishwan", "studied at", "iit kharagpur")]},
    {"text": "Aaron Jason Baptist studies in the AI Department.",
     "gold": [("aaron jason baptist", "member of", "ai department")]},
    {"text": "The AI Department is part of IIT Kharagpur.",
     "gold": [("ai department", "part of", "iit kharagpur")]},
    {"text": "Rahul studies Computer Science at IIT Kharagpur.",
     "gold": [("rahul", "studied at", "iit kharagpur"), ("rahul", "studies", "computer science")]},
    {"text": "IIT Kharagpur is located in West Bengal, India.",
     "gold": [("iit kharagpur", "located in", "west bengal"), ("iit kharagpur", "located in", "india")]},
    {"text": "Virat Kohli is married to Anushka Sharma.",
     "gold": [("virat kohli", "married to", "anushka sharma")]},
    {"text": "Sundar Pichai is the CEO of Google.",
     "gold": [("sundar pichai", "ceo of", "google")]},
    {"text": "Jeff Bezos founded Amazon.",
     "gold": [("jeff bezos", "founded", "amazon")]},
    {"text": "Python was created by Guido van Rossum.",
     "gold": [("python", "created by", "guido van rossum")]},
    {"text": "The theory of relativity was developed by Albert Einstein.",
     "gold": [("albert einstein", "developed", "theory of relativity")]},
    {"text": "Mark Zuckerberg co-founded Facebook.",
     "gold": [("mark zuckerberg", "founded", "facebook")]},
    {"text": "Ratan Tata is the chairman of Tata Group.",
     "gold": [("ratan tata", "chairman of", "tata group")]},
    {"text": "Warren Buffett is the CEO of Berkshire Hathaway.",
     "gold": [("warren buffett", "ceo of", "berkshire hathaway")]},
    {"text": "The Taj Mahal is located in Agra, India.",
     "gold": [("taj mahal", "located in", "agra"), ("taj mahal", "located in", "india")]},
    {"text": "Sachin Tendulkar played for Mumbai Indians.",
     "gold": [("sachin tendulkar", "played for", "mumbai indians")]},
]

# ══════════════════════════════════════════════════════════════════
# EVALUATION METRICS (same as Test_pipeline.py)
# ══════════════════════════════════════════════════════════════════

def _canonicalise(rel: str) -> str:
    return re.sub(r"\s+", " ", rel.lower().replace("_", " ").replace("-", " ")).strip()

def _relations_match(r1: str, r2: str) -> bool:
    c1, c2 = _canonicalise(r1), _canonicalise(r2)
    return c1 == c2 or c1 in c2 or c2 in c1

def _norm_triple(s, r, o):
    return (s.lower().strip(), _canonicalise(r), o.lower().strip())

def exact_f1(
    predicted: List[Tuple[str, str, str]],
    gold:      List[Tuple[str, str, str]],
) -> Tuple[float, float, float, int, int, int]:
    """Exact + soft match F1 — same method as Test_pipeline.py"""
    pred_norm = [_norm_triple(*t) for t in predicted]
    gold_norm = [_norm_triple(*t) for t in gold]
    pred_set  = set(pred_norm)
    gold_set  = set(gold_norm)

    # Exact matches
    exact_tp = len(pred_set & gold_set)

    # Soft matches
    unmatched_gold = [t for t in gold_norm if t not in pred_set]
    unmatched_pred = [t for t in pred_norm if t not in gold_set]
    soft_tp, used  = 0, set()
    for gs, gr, go in unmatched_gold:
        for ps, pr, po in unmatched_pred:
            if (ps, pr, po) in used:
                continue
            if _relations_match(gr, pr) and (gs in ps or ps in gs) and (go in po or po in go):
                soft_tp += 1
                used.add((ps, pr, po))
                break

    tp = exact_tp + soft_tp
    fp = max(len(pred_norm) - tp, 0)
    fn = max(len(gold_norm) - tp, 0)
    P  = tp / (tp + fp) if (tp + fp) else 0.0
    R  = tp / (tp + fn) if (tp + fn) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return round(P,4), round(R,4), round(F1,4), tp, fp, fn


def partial_f1(
    predicted: List[Tuple[str, str, str]],
    gold:      List[Tuple[str, str, str]],
) -> float:
    """Partial F1 — same method as Test_pipeline.py"""
    if not gold:
        return 1.0 if not predicted else 0.0
    scores = []
    for gs, gr, go in gold:
        best = 0.0
        for ps, pr, po in predicted:
            if not _relations_match(gr, pr):
                continue
            subj_ok = gs.lower() in ps.lower() or ps.lower() in gs.lower()
            obj_ok  = go.lower() in po.lower() or po.lower() in go.lower()
            best = max(best, (subj_ok + obj_ok) / 2.0)
        scores.append(best)
    return round(sum(scores) / len(scores), 4)


# ══════════════════════════════════════════════════════════════════
# EXTRACTORS
# ══════════════════════════════════════════════════════════════════

def extract_rebel(pipeline, text: str) -> List[Tuple[str, str, str]]:
    try:
        triplets = pipeline.process(text, verbose=False)
        return [(t.subject.text, t.relation, t.obj.text) for t in triplets
                if t.subject.text.lower() != t.obj.text.lower()
                and t.subject.text.lower() in text.lower()]
    except Exception as e:
        print(f"  [REBEL error] {e}")
        return []


def extract_groq(groq_extractor, text: str) -> List[Tuple[str, str, str]]:
    try:
        return groq_extractor.extract(text)
    except Exception as e:
        print(f"  [Groq error] {e}")
        return []


def extract_hybrid(pipeline, groq_triples, text: str) -> List[Tuple[str, str, str]]:
    try:
        from groq_extractor import fuse_triples
        rebel_raw = pipeline.process(text, verbose=False)
        rebel_for_fusion = [
            (t.subject.text, t.relation, t.obj.text, t.confidence)
            for t in rebel_raw
            if t.subject.text.lower() != t.obj.text.lower()
            and t.subject.text.lower() in text.lower()
        ]
        fused = fuse_triples(rebel_for_fusion, groq_triples, text)
        return [(t["subject"], t["relation"], t["object"]) for t in fused]
    except Exception as e:
        print(f"  [Hybrid error] {e}")
        return []


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def evaluate(verbose: bool = False):
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        print("❌ GROQ_API_KEY not set. Run: $env:GROQ_API_KEY='your_key'")
        sys.exit(1)

    print("Loading NLP pipeline...")
    from NLP_pipeline import NLPPipeline
    pipeline = NLPPipeline(num_beams=3, num_sequences=1, min_confidence=0.15)

    from groq_extractor import GroqExtractor
    groq = GroqExtractor(groq_key)

    results = {"rebel": [], "groq": [], "hybrid": []}
    partial = {"rebel": [], "groq": [], "hybrid": []}

    print(f"\n{'─'*80}")
    print(f"{'Sentence':<43} {'REBEL':>8} {'Groq':>8} {'Hybrid':>9}")
    print(f"{'─'*80}")

    for sample in EVAL_DATA:
        text = sample["text"]
        gold = sample["gold"]

        r_pred = extract_rebel(pipeline, text)
        g_pred = extract_groq(groq, text)
        h_pred = extract_hybrid(pipeline, g_pred, text)

        rp, rr, rf, *_ = exact_f1(r_pred, gold)
        gp, gr, gf, *_ = exact_f1(g_pred, gold)
        hp, hr, hf, *_ = exact_f1(h_pred, gold)

        r_part = partial_f1(r_pred, gold)
        g_part = partial_f1(g_pred, gold)
        h_part = partial_f1(h_pred, gold)

        results["rebel"].append((rp, rr, rf))
        results["groq"].append((gp, gr, gf))
        results["hybrid"].append((hp, hr, hf))
        partial["rebel"].append(r_part)
        partial["groq"].append(g_part)
        partial["hybrid"].append(h_part)

        short = text[:41] + ".." if len(text) > 43 else text
        best  = max(rf, gf, hf)
        h_str = f"{'*' if hf == best and hf > 0 else ' '}{hf:.2f}"
        print(f"{short:<43} {rf:>8.2f} {gf:>8.2f} {h_str:>9}")

        if verbose:
            print(f"  Gold:   {gold}")
            print(f"  REBEL:  {r_pred}  P={rp:.2f} R={rr:.2f} F1={rf:.2f} Partial={r_part:.2f}")
            print(f"  Groq:   {g_pred}  P={gp:.2f} R={gr:.2f} F1={gf:.2f} Partial={g_part:.2f}")
            print(f"  Hybrid: {h_pred}  P={hp:.2f} R={hr:.2f} F1={hf:.2f} Partial={h_part:.2f}")

    n = len(EVAL_DATA)
    def avg_col(lst, col): return sum(x[col] for x in lst) / n

    r_P,  r_R,  r_F1  = avg_col(results["rebel"],  0), avg_col(results["rebel"],  1), avg_col(results["rebel"],  2)
    g_P,  g_R,  g_F1  = avg_col(results["groq"],   0), avg_col(results["groq"],   1), avg_col(results["groq"],   2)
    h_P,  h_R,  h_F1  = avg_col(results["hybrid"], 0), avg_col(results["hybrid"], 1), avg_col(results["hybrid"], 2)
    r_pF1 = sum(partial["rebel"])  / n
    g_pF1 = sum(partial["groq"])   / n
    h_pF1 = sum(partial["hybrid"]) / n

    print(f"\n{'='*80}")
    print(f"  EVALUATION RESULTS ({n} sentences)")
    print(f"  Method: Exact F1 + Partial F1 (same as Test_pipeline.py)")
    print(f"{'='*80}")
    print(f"  {'Method':<20} {'Precision':>10} {'Recall':>8} {'Exact F1':>10} {'Partial F1':>12} {'vs REBEL':>10}")
    print(f"  {'─'*20} {'─'*10} {'─'*8} {'─'*10} {'─'*12} {'─'*10}")
    print(f"  {'REBEL only':<20} {r_P:>10.4f} {r_R:>8.4f} {r_F1:>10.4f} {r_pF1:>12.4f} {'baseline':>10}")
    print(f"  {'Groq only':<20} {g_P:>10.4f} {g_R:>8.4f} {g_F1:>10.4f} {g_pF1:>12.4f} {g_F1-r_F1:>+10.4f}")
    print(f"  {'Hybrid fusion':<20} {h_P:>10.4f} {h_R:>8.4f} {h_F1:>10.4f} {h_pF1:>12.4f} {h_F1-r_F1:>+10.4f}")
    print(f"{'='*80}")
    print(f"\n  Hybrid vs REBEL  — Exact F1:   {(h_F1-r_F1)*100:+.1f}%")
    print(f"  Hybrid vs REBEL  — Partial F1: {(h_pF1-r_pF1)*100:+.1f}%")
    print(f"  Hybrid vs Groq   — Exact F1:   {(h_F1-g_F1)*100:+.1f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    evaluate(verbose=args.verbose)
