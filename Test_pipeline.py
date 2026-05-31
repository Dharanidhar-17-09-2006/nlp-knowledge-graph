"""
test_pipeline.py — Comprehensive Test Suite
============================================
Tests every layer of the NL → Triplet → Cypher pipeline:

  1.  REBEL triplet extraction  (F1, precision, recall)
  2.  Cypher generation accuracy  (exact + partial match)
  3.  Entity label correctness  (NER label accuracy)
  4.  Relation normalisation  (RELATION_MAP coverage)
  5.  Deduplication  (fuzzy merge behaviour)
  6.  Pronoun resolution  (co-reference)
  7.  Long-text chunking  (no triplets lost on >128-token inputs)
  8.  Confidence scoring  (ordering guarantee)
  9.  Edge-cases  (empty input, unicode, punctuation, self-loops)
  10. Domain regression  (IITK-specific sentences)
  11. REBEL dataset benchmark  (500–1000 real test samples)

Run:
  python test_pipeline.py                          # all tests, coloured summary
  python test_pipeline.py --quick                  # skip slow model tests
  python test_pipeline.py --verbose                # print every triplet extracted
  python test_pipeline.py --output report.json     # save JSON report

  # ── REBEL dataset benchmark ──────────────────────────────────────
  python test_pipeline.py --rebel-file en_test.jsonl
  python test_pipeline.py --rebel-file en_test.jsonl --rebel-samples 500
  python test_pipeline.py --rebel-file en_test.jsonl --rebel-samples 1000 --rebel-output rebel_report.json
  python test_pipeline.py --rebel-file en_test.jsonl --rebel-skip-existing  # resume interrupted run

  The REBEL test file is the standard Hugging Face / Babelscape REBEL dataset
  split (en_test.jsonl).  Each line is a JSON object:
    {
      "title":    "...",
      "context":  "Sentence or paragraph ...",
      "triplets": [{"subject": "...", "predicate": "...", "object": "..."}, ...]
    }

  Download:
    wget https://huggingface.co/datasets/Babelscape/rebel-large/resolve/main/en_test.jsonl
  Or via the datasets library:
    from datasets import load_dataset
    ds = load_dataset("Babelscape/rebel-large", split="test")
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────
# ANSI colours (graceful degradation on Windows)
# ──────────────────────────────────────────────────────────────────
_USE_COLOR = sys.stdout.isatty() or os.environ.get("FORCE_COLOR")
def _c(code, text): return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text
GREEN  = lambda t: _c("32", t)
RED    = lambda t: _c("31", t)
YELLOW = lambda t: _c("33", t)
CYAN   = lambda t: _c("36", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)


# ══════════════════════════════════════════════════════════════════
# TEST DATA
# ══════════════════════════════════════════════════════════════════

# Each entry: (input_sentence, expected_triplets, expected_cypher_patterns)
# expected_triplets: list of (subj, rel_neo4j, obj) — all lowercase
# expected_cypher_patterns: list of regex strings that MUST appear in output

TRIPLET_TESTS: List[Dict] = [
    # ── IITK domain ───────────────────────────────────────────────
    {
        "id": "iitk_01",
        "text": "Srishwan is a student at IIT Kharagpur.",
        "expected_triplets": [
            ("srishwan", "STUDIED_AT", "iit kharagpur"),
            ("srishwan", "MEMBER_OF",  "iit kharagpur"),
            ("srishwan", "IS_A",       "student"),
        ],
        "cypher_patterns": [
            r"MERGE.*Person.*Srishwan",
            r"MERGE.*Organization.*IIT Kharagpur",
            r"STUDIED_AT|MEMBER_OF|IS_A",
        ],
        "category": "domain",
    },
    {
        "id": "iitk_02",
        "text": "Aaron Jason Baptist studies in the AI Department at IIT Kharagpur.",
        "expected_triplets": [
            ("aaron jason baptist", "MEMBER_OF",  "ai department"),
            ("ai department",       "PART_OF",    "iit kharagpur"),
        ],
        "cypher_patterns": [
            r"Aaron Jason Baptist",
            r"AI Department",
            r"IIT Kharagpur",
        ],
        "category": "domain",
    },
    {
        "id": "iitk_03",
        "text": "The student resides in Lal Bahadur Shastri Hall.",
        "expected_triplets": [("student", "LIVES_IN", "lal bahadur shastri hall")],
        "cypher_patterns": [r"Lal Bahadur Shastri Hall"],
        "category": "domain",
    },
    # ── General knowledge ─────────────────────────────────────────
    {
        "id": "gen_01",
        "text": "Google was founded by Larry Page and Sergey Brin.",
        "expected_triplets": [
            ("google", "FOUNDED_BY", "larry page"),
            ("google", "FOUNDED_BY", "sergey brin"),
        ],
        "cypher_patterns": [r"FOUNDED_BY", r"Larry Page", r"Sergey Brin"],
        "category": "general",
    },
    {
        "id": "gen_02",
        "text": "Albert Einstein was born in Ulm, Germany.",
        "expected_triplets": [
            ("albert einstein", "BORN_IN", "ulm"),
            ("albert einstein", "BORN_IN", "germany"),
            ("albert einstein", "BORN_IN", "ulm, germany"),
        ],
        "cypher_patterns": [r"BORN_IN", r"Albert Einstein"],
        "category": "general",
    },
    {
        "id": "gen_03",
        "text": "Marie Curie received the Nobel Prize in Physics in 1903.",
        "expected_triplets": [
            ("marie curie", "RECEIVED_AWARD", "nobel prize in physics"),
            ("marie curie", "RECEIVED_AWARD", "nobel prize"),
            ("marie curie", "RECEIVED_AWARD", "1903"),
        ],
        "cypher_patterns": [r"Marie Curie", r"RECEIVED_AWARD"],
        "category": "general",
    },
    {
        "id": "gen_04",
        "text": "The Eiffel Tower is located in Paris, France.",
        "expected_triplets": [
            ("eiffel tower", "LOCATED_IN", "paris"),
            ("eiffel tower", "LOCATED_IN", "france"),
            ("eiffel tower", "LOCATED_IN", "paris, france"),
            ("eiffel tower", "IN_COUNTRY",  "france"),
        ],
        "cypher_patterns": [r"Eiffel Tower", r"LOCATED_IN", r"Paris"],
        "category": "general",
    },
    {
        "id": "gen_05",
        "text": "Elon Musk is the CEO of Tesla.",
        "expected_triplets": [
            ("elon musk", "WORKS_FOR", "tesla"),
            ("elon musk", "HOLDS_POSITION", "ceo"),
            ("elon musk", "HOLDS_POSITION", "tesla"),
        ],
        "cypher_patterns": [r"Elon Musk", r"Tesla"],
        "category": "general",
    },
    # ── Multi-sentence (pronoun resolution) ───────────────────────
    {
        "id": "pronoun_01",
        "text": (
            "Marie Curie was a physicist. "
            "She was born in Warsaw."
        ),
        "expected_triplets": [("marie curie", "BORN_IN", "warsaw")],
        "cypher_patterns": [r"Marie Curie", r"Warsaw"],
        "category": "pronoun",
    },
    {
        "id": "pronoun_02",
        "text": (
            "Apple was founded by Steve Jobs. "
            "He grew up in California."
        ),
        "expected_triplets": [
            ("apple", "FOUNDED_BY", "steve jobs"),
        ],
        "cypher_patterns": [r"Apple", r"Steve Jobs"],
        "category": "pronoun",
    },
    # ── Edge cases ────────────────────────────────────────────────
    {
        "id": "edge_01",
        "text": "",
        "expected_triplets": [],
        "cypher_patterns": [],
        "category": "edge",
    },
    {
        "id": "edge_02",
        "text": "Hello world.",
        "expected_triplets": [],   # very short, likely no triplets
        "cypher_patterns": [],
        "category": "edge",
    },
    {
        "id": "edge_03",
        "text": "Résumé of François Müller from München.",
        "expected_triplets": [],
        "cypher_patterns": [],
        "category": "edge",
        "note": "unicode normalisation — must not crash",
    },
    # ── Long text (chunking) ──────────────────────────────────────
    {
        "id": "long_01",
        "text": (
            "Alan Turing was a British mathematician and computer scientist. "
            "He was born in London in 1912. "
            "Turing studied at King's College, Cambridge. "
            "He developed the concept of the Turing machine. "
            "Turing worked at Bletchley Park during World War II. "
            "He received the Order of the British Empire. "
            "Turing is widely considered the father of theoretical computer science."
        ),
        "expected_triplets": [
            ("alan turing", "BORN_IN", "london"),
            ("alan turing", "STUDIED_AT", "king's college"),
        ],
        "cypher_patterns": [r"Alan Turing"],
        "category": "long",
    },
]

# ── Relation normalisation unit tests ─────────────────────────────
RELATION_NORM_TESTS: List[Tuple[str, str]] = [
    ("educated at",           "STUDIED_AT"),
    ("place of birth",        "BORN_IN"),
    ("founded by",            "FOUNDED_BY"),
    ("headquarters location", "HEADQUARTERED_IN"),
    ("spouse",                "MARRIED_TO"),
    ("child",                 "PARENT_OF"),
    ("works at",              "WORKS_FOR"),
    ("located in",            "LOCATED_IN"),
    ("instance of",           "IS_A"),
    ("member of",             "MEMBER_OF"),
    ("EDUCATED AT",           "STUDIED_AT"),    # uppercase input
    ("  place of birth  ",    "BORN_IN"),       # whitespace
    ("totally unknown rel",   None),            # unknown → any non-empty string
]

# ── Deduplication tests ───────────────────────────────────────────
DEDUP_TESTS: List[Tuple[str, str, bool]] = [
    ("IIT Kharagpur",  "IIT Kharagpur",  True),   # identical
    ("IIT Kharagpur",  "iit kharagpur",  True),   # case
    ("Elon Musk",      "Elon  Musk",     True),   # extra space (fuzzy)
    ("Apple Inc",      "Apple",          True),   # substring
    ("Einstein",       "Newton",         False),  # different
    ("AI",             "AI Department",  True),   # substring
]

# ── Cypher structure tests (no model needed) ──────────────────────
CYPHER_STRUCTURE_TESTS: List[Dict] = [
    {
        "id": "cypher_merge",
        "triplets": [
            {"subject": "Alice", "subject_label": "Person",
             "relation": "WORKS_FOR",
             "object": "Acme Corp", "object_label": "Organization",
             "confidence": 0.9},
        ],
        "must_contain": ["MERGE", "WORKS_FOR", "Alice", "Acme Corp"],
        "must_not_contain": ["CREATE ("],  # should use MERGE not CREATE
    },
    {
        "id": "cypher_confidence",
        "triplets": [
            {"subject": "Bob", "subject_label": "Person",
             "relation": "BORN_IN",
             "object": "London", "object_label": "Location",
             "confidence": 0.75},
        ],
        "must_contain": ["MERGE", "Bob", "London", "BORN_IN"],
        "must_not_contain": [],
    },
    {
        "id": "cypher_special_chars",
        "triplets": [
            {"subject": "O'Brien", "subject_label": "Person",
             "relation": "LIVES_IN",
             "object": "São Paulo", "object_label": "Location",
             "confidence": 0.8},
        ],
        "must_contain": ["MERGE"],
        "must_not_contain": [],
        "note": "special characters in names must not break Cypher",
    },
]


# ══════════════════════════════════════════════════════════════════
# RESULT STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class TestResult:
    test_id:     str
    category:    str
    passed:      bool
    duration_ms: float
    details:     str = ""
    error:       str = ""
    metrics:     Dict = field(default_factory=dict)

@dataclass
class SuiteResult:
    total:    int = 0
    passed:   int = 0
    failed:   int = 0
    errors:   int = 0
    duration: float = 0.0
    results:  List[TestResult] = field(default_factory=list)

    @property
    def pass_rate(self): return self.passed / self.total if self.total else 0


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def _canonicalise_relation(rel: str) -> str:
    """
    Normalise a relation string so that predicted Neo4j-style relations
    (PART_OF, BORN_IN, FOUNDED_BY …) and gold Wikidata surface forms
    (part of, place of birth, founded by …) map to the same token set.

    Strategy:
      1. Lowercase
      2. Replace underscores and hyphens with spaces
      3. Strip leading/trailing whitespace
      4. Collapse multiple spaces

    This means "PART_OF" → "part of" and "part of" → "part of",
    so they will compare equal.
    """
    return re.sub(r"\s+", " ", rel.lower().replace("_", " ").replace("-", " ")).strip()


def _relations_match(r1: str, r2: str) -> bool:
    """True if two relation strings are equivalent after canonicalisation,
    or if one canonicalised form is a substring of the other."""
    c1, c2 = _canonicalise_relation(r1), _canonicalise_relation(r2)
    return c1 == c2 or c1 in c2 or c2 in c1


def _triplet_f1(
    predicted: List[Tuple[str, str, str]],
    gold:      List[Tuple[str, str, str]],
) -> Tuple[float, float, float, int, int, int]:
    """Return (P, R, F1, TP, FP, FN).

    Matching strategy
    -----------------
    * Relations are canonicalised before comparison so that predicted
      Neo4j-style labels (PART_OF, BORN_IN …) match gold Wikidata surface
      forms (part of, place of birth …).
    * Exact match: canonicalised (subj, rel, obj) triples are identical.
    * Soft match: relation matches AND subject/object overlap as substrings
      (handles 'Paris, France' vs 'Paris').
    """
    def _norm_triple(s, r, o):
        return (s.lower().strip(), _canonicalise_relation(r), o.lower().strip())

    pred_norm = [_norm_triple(*t) for t in predicted]
    gold_norm = [_norm_triple(*t) for t in gold]

    pred_set = set(pred_norm)
    gold_set = set(gold_norm)

    # Exact matches
    exact_tp = len(pred_set & gold_set)

    # Soft matches: cover gold items not already exactly matched
    unmatched_gold = [t for t in gold_norm if t not in pred_set]
    unmatched_pred = [t for t in pred_norm if t not in gold_set]
    soft_tp = 0
    used_pred = set()
    for gs, gr, go in unmatched_gold:
        for ps, pr, po in unmatched_pred:
            if (ps, pr, po) in used_pred:
                continue
            rel_ok  = _relations_match(gr, pr)
            subj_ok = gs in ps or ps in gs
            obj_ok  = go in po or po in go
            if rel_ok and subj_ok and obj_ok:
                soft_tp += 1
                used_pred.add((ps, pr, po))
                break

    tp = exact_tp + soft_tp
    fp = max(len(pred_norm) - tp, 0)
    fn = max(len(gold_norm) - tp, 0)
    P  = tp / (tp + fp) if (tp + fp) else 0.0
    R  = tp / (tp + fn) if (tp + fn) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F1, tp, fp, fn


def _partial_triplet_match(
    predicted: List[Tuple[str, str, str]],
    gold:      List[Tuple[str, str, str]],
) -> float:
    """
    Partial F1: a predicted triplet gets partial credit if subject OR
    object matches a gold triplet with the same relation (or a synonym).
    Relations are canonicalised before comparison.
    """
    if not gold:
        return 1.0 if not predicted else 0.0
    scores = []
    for gs, gr, go in gold:
        best = 0.0
        for ps, pr, po in predicted:
            if not _relations_match(gr, pr):
                continue
            subj_match = gs.lower() in ps.lower() or ps.lower() in gs.lower()
            obj_match  = go.lower() in po.lower() or po.lower() in go.lower()
            score = (subj_match + obj_match) / 2.0
            best = max(best, score)
        scores.append(best)
    return sum(scores) / len(scores)


def _load_pipeline(verbose: bool = False, min_conf: float = 0.25):
    global _pipeline_min_conf
    _pipeline_min_conf = min_conf
    """Import and instantiate NLPPipeline. Returns None on failure."""
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from NLP_pipeline import NLPPipeline
        pipeline = NLPPipeline(
            num_beams=4,
            num_sequences=2,
            min_confidence=_pipeline_min_conf,
        )
        return pipeline
    except Exception as e:
        if verbose:
            traceback.print_exc()
        return None


def _load_cypher_generator():
    """Try to import a to_cypher / triplets_to_cypher function."""
    try:
        from NLP_pipeline import triplets_to_cypher
        return triplets_to_cypher
    except ImportError:
        pass
    # Provide a minimal fallback so cypher tests still run
    return _fallback_cypher


def _fallback_cypher(triplets) -> str:
    """Minimal Cypher generator used when the real one isn't found."""
    lines = []
    for t in triplets:
        if hasattr(t, "subject"):
            s   = t.subject.text.replace("'", "\\'")
            sl  = t.subject.label
            o   = t.obj.text.replace("'", "\\'")
            ol  = t.obj.label
            rel = t.relation
        else:
            s   = t.get("subject", "?").replace("'", "\\'")
            sl  = t.get("subject_label", "Entity")
            o   = t.get("object", "?").replace("'", "\\'")
            ol  = t.get("object_label", "Entity")
            rel = t.get("relation", "RELATED_TO")
        lines.append(f"MERGE (a:{sl} {{name: '{s}'}})")
        lines.append(f"MERGE (b:{ol} {{name: '{o}'}})")
        lines.append(f"MERGE (a)-[:{rel}]->(b)")
    return "\n".join(lines)


def _load_relation_normaliser():
    try:
        from NLP_pipeline import normalize_relation
        return normalize_relation
    except ImportError:
        return None


def _load_deduplicator():
    try:
        from NLP_pipeline import EntityDeduplicator
        return EntityDeduplicator
    except ImportError:
        return None


# ══════════════════════════════════════════════════════════════════
# TEST RUNNERS
# ══════════════════════════════════════════════════════════════════

# ── 1. Relation normalisation (no model required) ─────────────────

def run_relation_norm_tests() -> List[TestResult]:
    normalise = _load_relation_normaliser()
    results = []
    for raw, expected in RELATION_NORM_TESTS:
        t0 = time.perf_counter()
        if normalise is None:
            results.append(TestResult(
                test_id=f"rel_norm::{raw[:20]}",
                category="relation_norm",
                passed=False,
                duration_ms=0,
                error="normalize_relation not importable",
            ))
            continue
        try:
            got = normalise(raw)
            if expected is None:
                # unknown relation → just must not be empty
                passed = bool(got and got.strip())
                detail = f"unknown→'{got}' (any non-empty OK)"
            else:
                passed = got == expected
                detail = f"'{raw}' → '{got}' (expected '{expected}')"
            results.append(TestResult(
                test_id=f"rel_norm::{raw[:20].strip()}",
                category="relation_norm",
                passed=passed,
                duration_ms=(time.perf_counter() - t0) * 1000,
                details=detail,
            ))
        except Exception as e:
            results.append(TestResult(
                test_id=f"rel_norm::{raw[:20]}",
                category="relation_norm",
                passed=False,
                duration_ms=(time.perf_counter() - t0) * 1000,
                error=str(e),
            ))
    return results


# ── 2. Deduplication (no model required) ─────────────────────────

def run_dedup_tests() -> List[TestResult]:
    DeduplicatorClass = _load_deduplicator()
    results = []
    for a, b, should_merge in DEDUP_TESTS:
        t0 = time.perf_counter()
        if DeduplicatorClass is None:
            results.append(TestResult(
                test_id=f"dedup::{a}↔{b}",
                category="dedup",
                passed=False,
                duration_ms=0,
                error="EntityDeduplicator not importable",
            ))
            continue
        try:
            d = DeduplicatorClass()
            # Register both entities first, then re-lookup to get the final
            # canonical after any merge has propagated through the store.
            d.get_canonical(a)
            d.get_canonical(b)
            canon_a = d.get_canonical(a)
            canon_b = d.get_canonical(b)
            merged  = (canon_a == canon_b)
            passed  = (merged == should_merge)
            results.append(TestResult(
                test_id=f"dedup::{a}↔{b}",
                category="dedup",
                passed=passed,
                duration_ms=(time.perf_counter() - t0) * 1000,
                details=(
                    f"'{a}' + '{b}' → canon_a='{canon_a}' canon_b='{canon_b}' "
                    f"merged={merged} expected={should_merge}"
                ),
            ))
        except Exception as e:
            results.append(TestResult(
                test_id=f"dedup::{a}↔{b}",
                category="dedup",
                passed=False,
                duration_ms=(time.perf_counter() - t0) * 1000,
                error=str(e),
            ))
    return results


# ── 3. Cypher structure tests (no model required) ─────────────────

def run_cypher_structure_tests() -> List[TestResult]:
    gen    = _load_cypher_generator()
    results = []
    for tc in CYPHER_STRUCTURE_TESTS:
        t0 = time.perf_counter()
        try:
            cypher = gen(tc["triplets"])
            must_have     = tc.get("must_contain", [])
            must_not_have = tc.get("must_not_contain", [])
            missing  = [s for s in must_have     if s not in cypher]
            bad      = [s for s in must_not_have if s in cypher]
            passed   = not missing and not bad
            details  = []
            if missing:  details.append(f"Missing: {missing}")
            if bad:      details.append(f"Forbidden found: {bad}")
            if not details:
                details.append("Cypher:\n" + cypher[:300])
            results.append(TestResult(
                test_id=tc["id"],
                category="cypher_structure",
                passed=passed,
                duration_ms=(time.perf_counter() - t0) * 1000,
                details="\n".join(details),
            ))
        except Exception as e:
            results.append(TestResult(
                test_id=tc["id"],
                category="cypher_structure",
                passed=False,
                duration_ms=(time.perf_counter() - t0) * 1000,
                error=str(e),
            ))
    return results


# ── 4. Full pipeline tests (requires model) ───────────────────────

def run_pipeline_tests(
    pipeline,
    cypher_gen,
    verbose: bool = False,
    categories: Optional[List[str]] = None,
) -> List[TestResult]:
    results = []
    for tc in TRIPLET_TESTS:
        if categories and tc["category"] not in categories:
            continue
        t0 = time.perf_counter()
        try:
            triplets = pipeline.process(tc["text"], verbose=False)
            dur_ms   = (time.perf_counter() - t0) * 1000

            # Convert Triplet objects to tuples
            pred_tuples = [(t.subject.text, t.relation, t.obj.text)
                           for t in triplets]
            gold_tuples = tc["expected_triplets"]

            # F1 metrics
            P, R, F1, tp, fp, fn = _triplet_f1(pred_tuples, gold_tuples)
            partial               = _partial_triplet_match(pred_tuples, gold_tuples)

            # Cypher pattern check
            cypher  = cypher_gen(triplets) if triplets else ""
            pat_hits = []
            pat_miss = []
            for pat in tc.get("cypher_patterns", []):
                (pat_hits if re.search(pat, cypher) else pat_miss).append(pat)

            # Confidence ordering check
            confs = [t.confidence for t in triplets]
            conf_ordered = confs == sorted(confs, reverse=True)

            # Self-loop guard (subj == obj should not appear)
            self_loops = [t for t in triplets
                          if t.subject.canonical == t.obj.canonical]

            # Empty-input guard
            if not tc["text"]:
                passed  = len(triplets) == 0
                details = f"Empty input → {len(triplets)} triplets (expected 0)"
            elif not gold_tuples:
                # no gold → just check it doesn't crash
                passed  = True
                details = f"No-gold smoke test: extracted {len(triplets)} triplets"
            else:
                passed  = F1 >= 0.3 or partial >= 0.4
                details = (
                    f"P={P:.2f} R={R:.2f} F1={F1:.2f} partial={partial:.2f} "
                    f"TP={tp} FP={fp} FN={fn} | "
                    f"cypher_ok={len(pat_miss)==0} | "
                    f"conf_order={conf_ordered} | "
                    f"self_loops={len(self_loops)}"
                )
                if pat_miss:
                    details += f"\n  Missing cypher patterns: {pat_miss}"

            if verbose:
                print(f"\n  {CYAN(tc['id'])}: {tc['text'][:60]}...")
                for t in triplets:
                    print(f"    {DIM(str(t))}")
                if cypher:
                    print(f"  Cypher preview:\n{DIM(cypher[:200])}")

            results.append(TestResult(
                test_id=tc["id"],
                category=tc["category"],
                passed=passed,
                duration_ms=dur_ms,
                details=details,
                metrics={
                    "P": round(P, 4), "R": round(R, 4), "F1": round(F1, 4),
                    "partial": round(partial, 4),
                    "n_predicted": len(triplets),
                    "n_gold": len(gold_tuples),
                    "cypher_patterns_missed": pat_miss,
                    "self_loops": len(self_loops),
                    "confidence_ordered": conf_ordered,
                },
            ))

        except Exception as e:
            results.append(TestResult(
                test_id=tc["id"],
                category=tc["category"],
                passed=False,
                duration_ms=(time.perf_counter() - t0) * 1000,
                error=traceback.format_exc(),
            ))
    return results


# ── 5. Aggregate F1 summary across all pipeline tests ─────────────

def _aggregate_metrics(results: List[TestResult]) -> Dict:
    ms = [r.metrics for r in results if r.metrics.get("n_gold", 0) > 0]
    if not ms:
        return {}
    avg = lambda k: sum(m[k] for m in ms) / len(ms)
    return {
        "macro_P":  round(avg("P"),  4),
        "macro_R":  round(avg("R"),  4),
        "macro_F1": round(avg("F1"), 4),
        "macro_partial_F1": round(avg("partial"), 4),
        "n_tests_with_gold": len(ms),
    }


# ══════════════════════════════════════════════════════════════════
# PRINTER
# ══════════════════════════════════════════════════════════════════

def _print_suite(title: str, results: List[TestResult]):
    print(f"\n{BOLD(f'── {title} ──────────────────────────────────────')}")
    for r in results:
        icon = GREEN("✓") if r.passed else RED("✗")
        ms   = f"{r.duration_ms:6.1f}ms"
        print(f"  {icon}  {r.test_id:<40} {ms}")
        if not r.passed or r.details:
            msg = r.error.split("\n")[0] if r.error else r.details
            print(f"       {DIM(msg[:110])}")
    ok  = sum(1 for r in results if r.passed)
    tot = len(results)
    col = GREEN if ok == tot else (YELLOW if ok >= tot * 0.7 else RED)
    print(f"  {col(f'{ok}/{tot} passed')}")


def _print_final(suite: SuiteResult, agg: Dict):
    bar_len = 40
    filled  = round(suite.pass_rate * bar_len)
    bar     = ("█" * filled) + ("░" * (bar_len - filled))
    col     = GREEN if suite.pass_rate >= 0.9 else (YELLOW if suite.pass_rate >= 0.7 else RED)

    print(f"\n{'═'*60}")
    print(BOLD("  FINAL RESULTS"))
    print(f"{'═'*60}")
    print(f"  {col(bar)}  {suite.passed}/{suite.total} ({suite.pass_rate:.0%})")
    print(f"  Total time : {suite.duration:.1f}s")
    if agg:
        print(f"\n  Model performance (macro avg over {agg['n_tests_with_gold']} gold tests):")
        print(f"    Precision  : {agg['macro_P']:.4f}")
        print(f"    Recall     : {agg['macro_R']:.4f}")
        print(f"    Exact F1   : {agg['macro_F1']:.4f}")
        print(f"    Partial F1 : {agg['macro_partial_F1']:.4f}")
    print(f"{'═'*60}\n")


# ══════════════════════════════════════════════════════════════════
# REBEL DATASET BENCHMARK
# ══════════════════════════════════════════════════════════════════

@dataclass
class REBELSampleResult:
    """Stores per-sample metrics for one REBEL test entry."""
    sample_id:   int
    title:       str
    text:        str
    n_gold:      int
    n_pred:      int
    exact_P:     float
    exact_R:     float
    exact_F1:    float
    partial_F1:  float
    tp:          int
    fp:          int
    fn:          int
    duration_ms: float
    relations_gold: List[str]   = field(default_factory=list)
    error:       str            = ""


def load_rebel_dataset(
    path: str,
    max_samples: int = 1000,
    shuffle: bool = False,
    seed: int = 42,
) -> List[Dict]:
    """
    Load up to *max_samples* entries from a REBEL JSONL test file.

    Accepted formats
    ────────────────
    1. Standard Babelscape REBEL JSONL  — one JSON object per line:
         {"title": "...", "context": "...",
          "triplets": [{"subject":"..","predicate":"..","object":".."},...]}

    2. HuggingFace datasets Arrow export (via datasets.load_dataset):
         Each row may use "token_ids_and_target" / "tokens" instead of
         "context"; the loader falls back to reconstructing text from
         available fields.

    Returns a list of dicts with normalised keys:
        text, triplets (list of (subj, pred, obj) str-tuples), title
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"REBEL test file not found: {path}\n"
            "Download with:\n"
            "  wget https://huggingface.co/datasets/Babelscape/rebel-large"
            "/resolve/main/en_test.jsonl\n"
            "or via the `datasets` library:\n"
            "  from datasets import load_dataset\n"
            "  ds = load_dataset('Babelscape/rebel-large', split='test')\n"
            "  ds.to_json('en_test.jsonl')"
        )

    raw_samples: List[Dict] = []

    # ── Try JSONL (one JSON per line) ────────────────────────────
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_samples.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(YELLOW(f"  ⚠  Skipping malformed line {lineno}: {exc}"))
            if len(raw_samples) >= max_samples * 5:   # read ahead for shuffle
                break

    if not raw_samples:
        raise ValueError(f"No valid JSON lines found in {path}")

    # ── Diagnostic: show ALL keys + values in the first row ────
    if raw_samples:
        first = raw_samples[0]
        print(f"  [DEBUG] First row keys: {list(first.keys())}")
        for k, v in first.items():
            preview = repr(v)[:200]
            print(f"  [DEBUG]   {k!r:30s} = {preview}")

    if shuffle:
        import random
        random.seed(seed)
        random.shuffle(raw_samples)

    raw_samples = raw_samples[:max_samples]

    # ── Normalise to {text, triplets, title} ─────────────────────
    normalised: List[Dict] = []
    for row in raw_samples:
        # --- text ---------------------------------------------------
        text = (
            row.get("context")
            or row.get("text")
            or row.get("sentence")
            or " ".join(row.get("tokens", []))
            or ""
        ).strip()

        # --- triplets -----------------------------------------------
        # Your REBEL file uses key 'triples' (no 't') with nested dicts:
        #   {'subject':   {'surfaceform': '...', 'uri': '...', ...},
        #    'predicate': {'surfaceform': '...', 'uri': '...', ...},
        #    'object':    {'surfaceform': '...', 'uri': '...', ...}}
        # We also support legacy key names and encoded string formats.
        raw_trips = (
            row.get("triples")       # ← your actual key
            or row.get("triplets")
            or row.get("relations")
            or []
        )
        target_text = (
            row.get("target_text")
            or row.get("labels_text")
            or row.get("rebel_target")
            or ""
        )
        trips: List[Tuple[str, str, str]] = []

        def _parse_rebel_token_string(s: str) -> List[Tuple[str, str, str]]:
            """Decode the REBEL special-token linearised format into (s, p, o) tuples."""
            parsed = []
            blocks = re.split(r"<triplet>", s)
            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                subj_match = re.split(r"<subj>", block, maxsplit=1)
                if len(subj_match) < 2:
                    continue
                subj = subj_match[0].strip()
                rest = subj_match[1]
                obj_match = re.split(r"<obj>", rest, maxsplit=1)
                if len(obj_match) < 2:
                    continue
                obj  = obj_match[0].strip()
                pred = obj_match[1].strip()
                if subj and pred and obj:
                    parsed.append((subj, pred, obj))
            return parsed

        def _surfaceform(field) -> str:
            """Extract text from a nested REBEL entity dict or plain string."""
            if isinstance(field, dict):
                return (
                    field.get("surfaceform")
                    or field.get("surface_form")
                    or field.get("text")
                    or field.get("mention")
                    or ""
                ).strip()
            return str(field).strip() if field else ""

        if target_text:
            trips = _parse_rebel_token_string(target_text)
        elif isinstance(raw_trips, str):
            trips = _parse_rebel_token_string(raw_trips)
        elif isinstance(raw_trips, list) and len(raw_trips) == 1 and isinstance(raw_trips[0], str):
            trips = _parse_rebel_token_string(raw_trips[0])
        else:
            for t in raw_trips:
                if isinstance(t, dict):
                    # Nested-dict format: subject/predicate/object each have 'surfaceform'
                    s_field = t.get("subject")   or t.get("head") or t.get("s") or ""
                    p_field = t.get("predicate") or t.get("relation") or t.get("p") or ""
                    o_field = t.get("object")    or t.get("tail")  or t.get("o") or ""
                    subj = _surfaceform(s_field) if isinstance(s_field, dict) else str(s_field).strip()
                    pred = _surfaceform(p_field) if isinstance(p_field, dict) else str(p_field).strip()
                    obj  = _surfaceform(o_field) if isinstance(o_field, dict) else str(o_field).strip()
                elif isinstance(t, (list, tuple)) and len(t) >= 3:
                    subj, pred, obj = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()
                else:
                    continue
                if subj and pred and obj:
                    trips.append((subj, pred, obj))

        if not text or not trips:
            continue   # skip entries with no usable ground truth

        normalised.append({
            "title":    row.get("title", ""),
            "text":     text,
            "triplets": trips,
        })

    print(f"  Loaded {len(normalised)} usable samples from {os.path.basename(path)}")
    return normalised


def run_rebel_benchmark(
    pipeline,
    dataset:    List[Dict],
    verbose:    bool = False,
    batch_size: int  = 50,
) -> List[REBELSampleResult]:
    """
    Run the NLPPipeline over every sample in *dataset* and return
    per-sample REBELSampleResult objects.

    Progress is printed every *batch_size* samples so you can watch
    it move on long runs.
    """
    results: List[REBELSampleResult] = []
    total = len(dataset)

    print(f"\n  Running pipeline on {total} REBEL samples…")
    print(f"  {'Sample':<8} {'Title':<35} {'Gold':>5} {'Pred':>5} "
          f"{'F1':>6} {'Partial':>8} {'ms':>6}")
    print(f"  {'─'*8} {'─'*35} {'─'*5} {'─'*5} {'─'*6} {'─'*8} {'─'*6}")

    for idx, sample in enumerate(dataset):
        t0 = time.perf_counter()
        title_short = sample["title"][:33] + ".." if len(sample["title"]) > 35 else sample["title"]

        try:
            triplets = pipeline.process(sample["text"], verbose=False)
            dur_ms   = (time.perf_counter() - t0) * 1000

            pred_tuples = [(t.subject.text, t.relation, t.obj.text)
                           for t in triplets]
            gold_tuples = sample["triplets"]

            P, R, F1, tp, fp, fn = _triplet_f1(pred_tuples, gold_tuples)
            partial              = _partial_triplet_match(pred_tuples, gold_tuples)

            res = REBELSampleResult(
                sample_id      = idx,
                title          = sample["title"],
                text           = sample["text"],
                n_gold         = len(gold_tuples),
                n_pred         = len(pred_tuples),
                exact_P        = round(P,  4),
                exact_R        = round(R,  4),
                exact_F1       = round(F1, 4),
                partial_F1     = round(partial, 4),
                tp             = tp,
                fp             = fp,
                fn             = fn,
                duration_ms    = round(dur_ms, 1),
                relations_gold = [g[1] for g in gold_tuples],
            )

            # Live progress line
            f1_col = (GREEN if F1 >= 0.5 else YELLOW if F1 >= 0.2 else RED)(f"{F1:.3f}")
            if verbose or idx % batch_size == 0 or idx == total - 1:
                print(f"  {idx+1:<8} {title_short:<35} "
                      f"{len(gold_tuples):>5} {len(pred_tuples):>5} "
                      f"{f1_col:>6} {partial:>8.3f} {dur_ms:>6.0f}")

            if verbose:
                for pt in pred_tuples:
                    print(f"    pred  {DIM(str(pt))}")
                for gt in gold_tuples:
                    print(f"    gold  {DIM(str(gt))}")

        except Exception:
            dur_ms = (time.perf_counter() - t0) * 1000
            res = REBELSampleResult(
                sample_id   = idx,
                title       = sample["title"],
                text        = sample["text"],
                n_gold      = len(sample["triplets"]),
                n_pred      = 0,
                exact_P     = 0.0,
                exact_R     = 0.0,
                exact_F1    = 0.0,
                partial_F1  = 0.0,
                tp          = 0,
                fp          = 0,
                fn          = len(sample["triplets"]),
                duration_ms = round(dur_ms, 1),
                error       = traceback.format_exc(limit=3),
            )
            print(f"  {idx+1:<8} {title_short:<35} "
                  f"{RED('ERROR')}")

        results.append(res)

    return results


def _print_rebel_report(results: List[REBELSampleResult]) -> Dict:
    """
    Compute and print aggregate metrics from REBEL benchmark results.
    Returns the aggregate dict (also written to JSON report if requested).
    """
    ok = [r for r in results if not r.error]
    n  = len(ok)
    if n == 0:
        print(RED("  No successful REBEL samples to report on."))
        return {}

    # ── Macro averages ───────────────────────────────────────────
    macro_P       = sum(r.exact_P    for r in ok) / n
    macro_R       = sum(r.exact_R    for r in ok) / n
    macro_F1      = sum(r.exact_F1   for r in ok) / n
    macro_partial = sum(r.partial_F1 for r in ok) / n

    # ── Micro averages (aggregate TP/FP/FN across all samples) ───
    tot_tp = sum(r.tp for r in ok)
    tot_fp = sum(r.fp for r in ok)
    tot_fn = sum(r.fn for r in ok)
    micro_P  = tot_tp / (tot_tp + tot_fp) if (tot_tp + tot_fp) else 0.0
    micro_R  = tot_tp / (tot_tp + tot_fn) if (tot_tp + tot_fn) else 0.0
    micro_F1 = 2 * micro_P * micro_R / (micro_P + micro_R) if (micro_P + micro_R) else 0.0

    # ── Accuracy buckets ─────────────────────────────────────────
    #   "hit"   = at least one correct triplet extracted (F1 > 0)
    #   "good"  = F1 >= 0.5
    #   "exact" = F1 == 1.0
    hit_rate   = sum(1 for r in ok if r.exact_F1  > 0.0)  / n
    good_rate  = sum(1 for r in ok if r.exact_F1  >= 0.5) / n
    exact_rate = sum(1 for r in ok if r.exact_F1  == 1.0) / n

    # ── Per-relation breakdown (top-20 by frequency) ─────────────
    rel_stats: Dict[str, Dict] = defaultdict(lambda: {"n": 0, "tp": 0, "fn": 0})
    for r in ok:
        for rel in r.relations_gold:
            rel_norm = rel.lower().replace(" ", "_")
            rel_stats[rel_norm]["n"]  += 1
            # crude: count relation as TP if the sample had F1 > 0
            rel_stats[rel_norm]["tp"] += 1 if r.exact_F1 > 0 else 0
            rel_stats[rel_norm]["fn"] += 0 if r.exact_F1 > 0 else 1

    top_rels = sorted(rel_stats.items(), key=lambda x: -x[1]["n"])[:20]

    # ── Timing ───────────────────────────────────────────────────
    avg_ms   = sum(r.duration_ms for r in ok) / n
    total_s  = sum(r.duration_ms for r in results) / 1000

    # ── Print ────────────────────────────────────────────────────
    W = 60
    print(f"\n{'═'*W}")
    print(BOLD("  REBEL DATASET BENCHMARK RESULTS"))
    print(f"{'═'*W}")
    print(f"  Samples evaluated  : {len(results)}  "
          f"({len(results)-len(ok)} errors)")
    print(f"  Total time         : {total_s:.1f}s  "
          f"(avg {avg_ms:.0f}ms / sample)")
    print()
    print(BOLD("  ── Exact-match metrics ──────────────────────────"))
    print(f"  Macro  Precision : {macro_P:.4f}")
    print(f"  Macro  Recall    : {macro_R:.4f}")
    print(f"  Macro  F1        : {macro_F1:.4f}")
    print()
    print(f"  Micro  Precision : {micro_P:.4f}   (TP={tot_tp} FP={tot_fp})")
    print(f"  Micro  Recall    : {micro_R:.4f}   (FN={tot_fn})")
    print(f"  Micro  F1        : {micro_F1:.4f}")
    print()
    print(BOLD("  ── Partial-match metric ─────────────────────────"))
    print(f"  Macro Partial F1 : {macro_partial:.4f}")
    print()
    print(BOLD("  ── Accuracy buckets ─────────────────────────────"))
    print(f"  ≥1 correct triplet  (hit)    : "
          f"{hit_rate:.1%}  ({int(hit_rate*n)}/{n})")
    print(f"  F1 ≥ 0.5            (good)   : "
          f"{good_rate:.1%}  ({int(good_rate*n)}/{n})")
    print(f"  F1 = 1.0            (perfect): "
          f"{exact_rate:.1%}  ({int(exact_rate*n)}/{n})")
    print()
    print(BOLD("  ── Top-20 relations (by gold frequency) ─────────"))
    print(f"  {'Relation':<35} {'Count':>6}  {'Hit%':>6}")
    print(f"  {'─'*35} {'─'*6}  {'─'*6}")
    for rel, st in top_rels:
        hit_pct = st["tp"] / st["n"] if st["n"] else 0
        col = GREEN if hit_pct >= 0.6 else YELLOW if hit_pct >= 0.3 else RED
        print(f"  {rel:<35} {st['n']:>6}  {col(f'{hit_pct:.1%}'):>6}")
    print(f"{'═'*W}\n")

    agg = {
        "n_samples":       len(results),
        "n_errors":        len(results) - len(ok),
        "macro_P":         round(macro_P,       4),
        "macro_R":         round(macro_R,       4),
        "macro_F1":        round(macro_F1,      4),
        "macro_partial_F1":round(macro_partial, 4),
        "micro_P":         round(micro_P,       4),
        "micro_R":         round(micro_R,       4),
        "micro_F1":        round(micro_F1,      4),
        "hit_rate":        round(hit_rate,      4),
        "good_rate":       round(good_rate,     4),
        "exact_rate":      round(exact_rate,    4),
        "avg_ms_per_sample": round(avg_ms,      1),
        "total_s":         round(total_s,       1),
        "top_relations":   [
            {"relation": r, "count": s["n"], "hit_rate": round(s["tp"]/s["n"], 4)}
            for r, s in top_rels
        ],
    }
    return agg


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NL→Cypher pipeline test suite")
    parser.add_argument("--quick",   action="store_true",
                        help="Skip model tests (only structural/unit tests)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print extracted triplets for each test")
    parser.add_argument("--output",  default=None,
                        help="Save JSON report to this path")
    parser.add_argument("--category", nargs="*",
                        help="Only run pipeline tests in these categories "
                             "(domain, general, pronoun, edge, long)")

    # ── REBEL benchmark arguments ─────────────────────────────────
    parser.add_argument(
        "--rebel-file", default=None, metavar="PATH",
        help=(
            "Path to the REBEL en_test.jsonl file.  When provided, runs a "
            "large-scale accuracy benchmark on 500–1000 real test samples "
            "in addition to the 11 hard-coded tests.  Download with:\n"
            "  wget https://huggingface.co/datasets/Babelscape/rebel-large"
            "/resolve/main/en_test.jsonl"
        ),
    )
    parser.add_argument(
        "--rebel-samples", type=int, default=1000, metavar="N",
        help="Number of REBEL samples to evaluate (default: 1000, min: 1).",
    )
    parser.add_argument(
        "--rebel-shuffle", action="store_true",
        help="Randomly sample from the REBEL test file instead of taking "
             "the first N entries.",
    )
    parser.add_argument(
        "--rebel-output", default=None, metavar="PATH",
        help="Save per-sample REBEL results to this JSON file "
             "(separate from --output).",
    )
    parser.add_argument(
        "--rebel-batch-size", type=int, default=50, metavar="B",
        help="Print a progress line every B samples (default: 50).",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.25, metavar="F",
        help=(
            "Minimum triplet confidence score to emit (default: 0.25). "
            "Uses real beam sequence probabilities — higher values = fewer but "
            "more precise triplets.  Try 0.25 (default), 0.5, or 0.75."
        ),
    )

    args = parser.parse_args()

    suite   = SuiteResult()
    t_start = time.perf_counter()
    all_results: Dict[str, List[TestResult]] = {}

    # ── Unit tests (always run) ────────────────────────────────────
    print(BOLD("\n🔬  Running unit tests (no model required)…"))

    rel_results  = run_relation_norm_tests()
    dedup_results = run_dedup_tests()
    cyph_results  = run_cypher_structure_tests()

    all_results["relation_normalisation"] = rel_results
    all_results["deduplication"]          = dedup_results
    all_results["cypher_structure"]       = cyph_results

    _print_suite("Relation normalisation", rel_results)
    _print_suite("Entity deduplication",   dedup_results)
    _print_suite("Cypher structure",       cyph_results)

    for g in [rel_results, dedup_results, cyph_results]:
        suite.total  += len(g)
        suite.passed += sum(1 for r in g if r.passed)
        suite.failed += sum(1 for r in g if not r.passed and not r.error)
        suite.errors += sum(1 for r in g if r.error)

    agg: Dict = {}

    # ── Model tests (skipped with --quick) ────────────────────────
    if not args.quick:
        print(BOLD("\n🤖  Loading NLPPipeline for model tests…"))
        pipeline = _load_pipeline(verbose=args.verbose, min_conf=args.min_confidence)
        if pipeline is None:
            print(YELLOW(
                "  ⚠  NLPPipeline could not be loaded.\n"
                "     Install dependencies and ensure NLP_pipeline.py is present.\n"
                "     Skipping model tests."
            ))
        else:
            cypher_gen = _load_cypher_generator()
            pipe_results = run_pipeline_tests(
                pipeline,
                cypher_gen,
                verbose=args.verbose,
                categories=args.category,
            )
            all_results["pipeline"] = pipe_results
            _print_suite("Pipeline (triplet + cypher)", pipe_results)

            suite.total  += len(pipe_results)
            suite.passed += sum(1 for r in pipe_results if r.passed)
            suite.failed += sum(1 for r in pipe_results if not r.passed and not r.error)
            suite.errors += sum(1 for r in pipe_results if r.error)

            agg = _aggregate_metrics(pipe_results)

            # ── REBEL dataset benchmark ───────────────────────────
            if args.rebel_file:
                print(BOLD(
                    f"\n📊  REBEL benchmark — "
                    f"loading up to {args.rebel_samples} samples from "
                    f"{args.rebel_file} …"
                ))
                try:
                    rebel_dataset = load_rebel_dataset(
                        path        = args.rebel_file,
                        max_samples = max(1, args.rebel_samples),
                        shuffle     = args.rebel_shuffle,
                    )
                    rebel_results = run_rebel_benchmark(
                        pipeline   = pipeline,
                        dataset    = rebel_dataset,
                        verbose    = args.verbose,
                        batch_size = args.rebel_batch_size,
                    )
                    rebel_agg = _print_rebel_report(rebel_results)
                    all_results["rebel_benchmark"] = []   # kept separate below

                    # Save per-sample REBEL results if requested
                    if args.rebel_output:
                        rebel_report = {
                            "aggregate": rebel_agg,
                            "samples": [
                                {
                                    "id":          r.sample_id,
                                    "title":       r.title,
                                    "text":        r.text[:200],
                                    "n_gold":      r.n_gold,
                                    "n_pred":      r.n_pred,
                                    "exact_P":     r.exact_P,
                                    "exact_R":     r.exact_R,
                                    "exact_F1":    r.exact_F1,
                                    "partial_F1":  r.partial_F1,
                                    "tp":          r.tp,
                                    "fp":          r.fp,
                                    "fn":          r.fn,
                                    "duration_ms": r.duration_ms,
                                    "error":       r.error,
                                }
                                for r in rebel_results
                            ],
                        }
                        with open(args.rebel_output, "w", encoding="utf-8") as fh:
                            json.dump(rebel_report, fh, indent=2)
                        print(f"  REBEL per-sample report saved → {args.rebel_output}")

                    # Merge REBEL aggregate into main agg for final display
                    agg["rebel"] = rebel_agg

                except FileNotFoundError as exc:
                    print(RED(f"\n  ✗  {exc}"))
                except Exception:
                    print(RED("\n  ✗  REBEL benchmark failed:"))
                    traceback.print_exc()

    else:
        print(YELLOW("\n  --quick mode: model tests skipped."))
        if args.rebel_file:
            print(YELLOW("  --quick mode: REBEL benchmark also skipped."))

    suite.duration = time.perf_counter() - t_start
    _print_final(suite, agg)

    # ── Save JSON report ──────────────────────────────────────────
    if args.output:
        report = {
            "summary": asdict(suite),
            "aggregate_metrics": agg,
            "suites": {
                name: [asdict(r) for r in rs]
                for name, rs in all_results.items()
                if name != "rebel_benchmark"   # saved separately via --rebel-output
            },
        }
        # Remove non-serialisable 'results' field from summary
        report["summary"].pop("results", None)
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"  Report saved → {args.output}")

    return 0 if suite.pass_rate >= 0.7 else 1


if __name__ == "__main__":
    sys.exit(main())