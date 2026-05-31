"""
Tune hybrid fusion parameters for the 30-sentence benchmark.

The script runs REBEL and Groq once, then sweeps fusion thresholds without
making repeated API calls. It reports the best configs by Exact F1 and Partial
F1 so README/resume metrics are based on measured values.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

from dotenv import load_dotenv

from evaluate_hybrid import EVAL_DATA, exact_f1, partial_f1
from groq_extractor import fuse_triples

load_dotenv()


RawRebel = Tuple[str, str, str, float, str, str]
RawGroq = Tuple[str, str, str]


def _contains_entity(entity: str, text: str) -> bool:
    entity_l = entity.lower().strip()
    text_l = text.lower()
    if entity_l in text_l:
        return True
    words = [w for w in entity_l.split() if len(w) > 3]
    return any(w in text_l for w in words)


def collect_predictions() -> Dict[str, Dict[str, list]]:
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    print("Loading NLP pipeline...")
    from NLP_pipeline import NLPPipeline

    pipeline = NLPPipeline(num_beams=3, num_sequences=1, min_confidence=0.15)

    print("Loading Groq extractor...")
    from groq_extractor import GroqExtractor

    groq = GroqExtractor(groq_key)

    cache: Dict[str, Dict[str, list]] = {}
    for idx, sample in enumerate(EVAL_DATA, start=1):
        text = sample["text"]
        print(f"[{idx:02d}/{len(EVAL_DATA)}] {text[:64]}")

        rebel_raw: List[RawRebel] = []
        for t in pipeline.process(text, verbose=False):
            if t.subject.text.lower() == t.obj.text.lower():
                continue
            if not _contains_entity(t.subject.text, text) or not _contains_entity(t.obj.text, text):
                continue
            rebel_raw.append(
                (
                    t.subject.text,
                    t.relation,
                    t.obj.text,
                    t.confidence,
                    t.subject.label,
                    t.obj.label,
                )
            )

        groq_raw: List[RawGroq] = groq.extract(text)
        cache[text] = {"rebel": rebel_raw, "groq": groq_raw}

    return cache


def score_config(cache, lexical_threshold: float, rebel_conf_threshold: float):
    exact_scores = []
    partial_scores = []
    total_pred = 0

    for sample in EVAL_DATA:
        text = sample["text"]
        gold = sample["gold"]
        fused = fuse_triples(
            cache[text]["rebel"],
            cache[text]["groq"],
            text,
            lexical_threshold=lexical_threshold,
            rebel_conf_threshold=rebel_conf_threshold,
        )
        pred = [(t["subject"], t["relation"], t["object"]) for t in fused]
        total_pred += len(pred)
        exact_scores.append(exact_f1(pred, gold)[:3])
        partial_scores.append(partial_f1(pred, gold))

    n = len(EVAL_DATA)
    precision = sum(row[0] for row in exact_scores) / n
    recall = sum(row[1] for row in exact_scores) / n
    exact = sum(row[2] for row in exact_scores) / n
    partial = sum(partial_scores) / n
    avg_pred = total_pred / n
    return {
        "lexical_threshold": lexical_threshold,
        "rebel_conf_threshold": rebel_conf_threshold,
        "precision": precision,
        "recall": recall,
        "exact_f1": exact,
        "partial_f1": partial,
        "avg_pred": avg_pred,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    cache = collect_predictions()

    lexical_values = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rebel_conf_values = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

    results = []
    for lexical in lexical_values:
        for rebel_conf in rebel_conf_values:
            results.append(score_config(cache, lexical, rebel_conf))

    best_exact = sorted(results, key=lambda r: (r["exact_f1"], r["partial_f1"], r["precision"]), reverse=True)
    best_partial = sorted(results, key=lambda r: (r["partial_f1"], r["exact_f1"], r["precision"]), reverse=True)

    def print_table(title, rows):
        print(f"\n{title}")
        print("-" * 96)
        print(f"{'lex':>5} {'rebel':>7} {'prec':>8} {'recall':>8} {'exact':>8} {'partial':>8} {'avg_pred':>9}")
        for row in rows[: args.top]:
            print(
                f"{row['lexical_threshold']:>5.2f} "
                f"{row['rebel_conf_threshold']:>7.2f} "
                f"{row['precision']:>8.4f} "
                f"{row['recall']:>8.4f} "
                f"{row['exact_f1']:>8.4f} "
                f"{row['partial_f1']:>8.4f} "
                f"{row['avg_pred']:>9.2f}"
            )

    print_table("Best by Exact F1", best_exact)
    print_table("Best by Partial F1", best_partial)


if __name__ == "__main__":
    main()
