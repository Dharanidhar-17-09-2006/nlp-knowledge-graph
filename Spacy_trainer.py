"""
train_spacy_ner.py  ── Local REBEL Dataset Edition (Fixed)
===========================================================
Trains a high-accuracy spaCy NER model from your locally downloaded
REBEL dataset at ./rebel_dataset/en_train.jsonl

Install (the ONLY packages you need):
  pip install spacy torch
  python -m spacy download en_core_web_lg   ← recommended
  python -m spacy download en_core_web_sm   ← fallback

DO NOT run: pip install spacy-lookups-data
This script patches the E955 lookup error internally — you don't
need that package. If you already installed it, that's fine too.

Usage:
  python train_spacy_ner.py                         # 8000 samples, 40 epochs
  python train_spacy_ner.py --inspect               # verify data parsing first
  python train_spacy_ner.py --samples 15000 --epochs 60 --balance
  python train_spacy_ner.py --save-mined out.json   # inspect mined annotations

REBEL JSONL schema auto-detected (Schema A or B — see below).
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════
# CRITICAL: Apply the E955 lookup patch BEFORE any other spaCy import.
# This fixes the "Can't find table lexeme_norm" error that occurs when
# spacy-lookups-data is not installed. Do NOT move these lines down.
# ══════════════════════════════════════════════════════════════════
import spacy.lookups as _spacy_lookups
import spacy.language as _spacy_language
from spacy.lookups import Lookups as _Lookups

_orig_load_lookups = _spacy_lookups.load_lookups

def _patched_load_lookups(lang: str, tables: list, strict: bool = True):
    """
    Drop-in replacement for spacy.lookups.load_lookups that silently
    returns empty lookup tables when spacy-lookups-data is not installed
    (E955 error). Has zero effect when the data IS installed.
    """
    try:
        return _orig_load_lookups(lang, tables, strict=strict)
    except ValueError as exc:
        if "E955" in str(exc):
            # Return empty tables — the NER trainer does not need lexeme data
            lk = _Lookups()
            for table in (tables or []):
                lk.add_table(table, {})
            return lk
        raise  # re-raise any other ValueError unchanged

_spacy_lookups.load_lookups = _patched_load_lookups
_spacy_language.load_lookups = _patched_load_lookups
# ── End of patch ──────────────────────────────────────────────────

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import spacy
from spacy.training import Example as SpacyExample
from spacy.util import minibatch, compounding


# ══════════════════════════════════════════════════════════════════
# 1. RELATION → NER LABEL TABLE
# Maps every Wikidata / REBEL predicate string to
# (subject_spaCy_label, object_spaCy_label).
# ══════════════════════════════════════════════════════════════════

REL_TO_NER: Dict[str, Tuple[str, str]] = {
    # ── Person-centric ────────────────────────────────────────────
    "place of birth":           ("PERSON",     "GPE"),
    "place of death":           ("PERSON",     "GPE"),
    "date of birth":            ("PERSON",     "DATE"),
    "date of death":            ("PERSON",     "DATE"),
    "country of citizenship":   ("PERSON",     "GPE"),
    "educated at":              ("PERSON",     "ORG"),
    "student at":               ("PERSON",     "ORG"),
    "employer":                 ("PERSON",     "ORG"),
    "works at":                 ("PERSON",     "ORG"),
    "occupation":               ("PERSON",     "NORP"),
    "position held":            ("PERSON",     "NORP"),
    "award received":           ("PERSON",     "WORK_OF_ART"),
    "member of":                ("PERSON",     "ORG"),
    "spouse":                   ("PERSON",     "PERSON"),
    "child":                    ("PERSON",     "PERSON"),
    "parent":                   ("PERSON",     "PERSON"),
    "sibling":                  ("PERSON",     "PERSON"),
    "field of work":            ("PERSON",     "NORP"),
    "student of":               ("PERSON",     "PERSON"),
    "doctoral advisor":         ("PERSON",     "PERSON"),
    "doctoral student":         ("PERSON",     "PERSON"),
    "notable work":             ("PERSON",     "WORK_OF_ART"),
    "sex or gender":            ("PERSON",     "NORP"),
    "religion":                 ("PERSON",     "NORP"),
    "nationality":              ("PERSON",     "NORP"),
    "ethnic group":             ("PERSON",     "NORP"),
    "languages spoken":         ("PERSON",     "LANGUAGE"),
    "resides in":               ("PERSON",     "FAC"),
    "stays in":                 ("PERSON",     "FAC"),
    "enrolled in":              ("PERSON",     "ORG"),
    # ── Organization-centric ──────────────────────────────────────
    "founded by":               ("ORG",        "PERSON"),
    "inception":                ("ORG",        "DATE"),
    "dissolved":                ("ORG",        "DATE"),
    "headquarters location":    ("ORG",        "GPE"),
    "country":                  ("ORG",        "GPE"),
    "owned by":                 ("ORG",        "ORG"),
    "subsidiary":               ("ORG",        "ORG"),
    "parent organization":      ("ORG",        "ORG"),
    "operator":                 ("ORG",        "ORG"),
    "located in the admin terr":("ORG",        "GPE"),
    # ── Product / Technology ──────────────────────────────────────
    "manufacturer":             ("PRODUCT",    "ORG"),
    "developer":                ("PRODUCT",    "ORG"),
    "instance of":              ("PRODUCT",    "NORP"),
    "subclass of":              ("PRODUCT",    "PRODUCT"),
    "part of":                  ("PRODUCT",    "PRODUCT"),
    "has part":                 ("PRODUCT",    "PRODUCT"),
    # ── Creative Works ────────────────────────────────────────────
    "author":                   ("WORK_OF_ART","PERSON"),
    "director":                 ("WORK_OF_ART","PERSON"),
    "cast member":              ("WORK_OF_ART","PERSON"),
    "screenwriter":             ("WORK_OF_ART","PERSON"),
    "composer":                 ("WORK_OF_ART","PERSON"),
    "publisher":                ("WORK_OF_ART","ORG"),
    "genre":                    ("WORK_OF_ART","NORP"),
    "publication date":         ("WORK_OF_ART","DATE"),
    # ── Location ──────────────────────────────────────────────────
    "located in":               ("LOC",        "GPE"),
    "capital":                  ("GPE",        "GPE"),
    "official language":        ("GPE",        "LANGUAGE"),
    "head of government":       ("GPE",        "PERSON"),
    "head of state":            ("GPE",        "PERSON"),
    "continent":                ("GPE",        "LOC"),
    "administrative division":  ("GPE",        "GPE"),
    # ── Event ─────────────────────────────────────────────────────
    "point in time":            ("EVENT",      "DATE"),
    "location":                 ("EVENT",      "GPE"),
    "participant":              ("EVENT",      "PERSON"),
}

# Priority order: lower index = higher priority when two labels conflict
LABEL_PRIORITY: List[str] = [
    "PERSON", "ORG", "GPE", "LOC", "FAC",
    "WORK_OF_ART", "PRODUCT", "EVENT",
    "DATE", "NORP", "LANGUAGE", "QUANTITY", "MONEY",
]


# ══════════════════════════════════════════════════════════════════
# 2. JSONL READER — auto-detects Schema A or B
# ══════════════════════════════════════════════════════════════════

def _parse_line(raw: dict) -> Optional[Tuple[str, List[Tuple[str, str, str]]]]:
    """Parse one JSONL dict → (sentence, [(subj, pred, obj), ...])"""
    sentence = (
        raw.get("text") or
        raw.get("context") or
        raw.get("title") or
        ""
    ).strip()
    if not sentence or len(sentence) < 10:
        return None

    triplets: List[Tuple[str, str, str]] = []

    # ── Schema A ──────────────────────────────────────────────────
    for t in raw.get("triples", []):
        try:
            s = (t.get("subject",   {}).get("surfaceform") or
                 t.get("subject",   {}).get("uri") or "").strip()
            p = (t.get("predicate", {}).get("surfaceform") or
                 t.get("predicate", {}).get("uri") or "").strip()
            o = (t.get("object",    {}).get("surfaceform") or
                 t.get("object",    {}).get("uri") or "").strip()
            # Skip Wikidata Q-IDs (not useful as surface forms)
            if s and p and o and not re.match(r"^Q\d+$", s):
                triplets.append((s, p.lower(), o))
        except (AttributeError, TypeError):
            continue

    # ── Schema B ──────────────────────────────────────────────────
    for t in raw.get("triplets", []):
        try:
            s = (t.get("head")     or t.get("subject")  or "").strip()
            p = (t.get("type")     or t.get("relation") or "").strip()
            o = (t.get("tail")     or t.get("object")   or "").strip()
            if s and p and o:
                triplets.append((s, p.lower(), o))
        except (AttributeError, TypeError):
            continue

    return (sentence, triplets) if triplets else None


# ══════════════════════════════════════════════════════════════════
# 3. SPAN FINDER — token-boundary aware
# ══════════════════════════════════════════════════════════════════

def _snap_to_tokens(
    doc,
    char_start: int,
    char_end: int,
) -> Optional[Tuple[int, int]]:
    """
    Snap a character span to spaCy token boundaries.
    Tries "contract" (exact) first, falls back to "expand".
    Returns (start_char, end_char) or None.
    """
    # Exact token match
    span = doc.char_span(char_start, char_end, alignment_mode="contract")
    if span and len(span) > 0:
        return span.start_char, span.end_char
    # Expand to nearest token boundaries
    span = doc.char_span(char_start, char_end, alignment_mode="expand")
    if span and len(span) > 0:
        return span.start_char, span.end_char
    return None


def find_span(
    text: str,
    entity: str,
    tok_doc,
) -> Optional[Tuple[int, int]]:
    """
    Find the character span of `entity` in `text`, snapping to token
    boundaries so spaCy never raises alignment errors.

    Search order:
      1. Exact match
      2. Case-insensitive match
      3. Longest word-prefix match (handles truncated surfaceforms)
    """
    # 1. Exact
    idx = text.find(entity)
    if idx != -1:
        result = _snap_to_tokens(tok_doc, idx, idx + len(entity))
        if result:
            return result

    # 2. Case-insensitive
    lower_text = text.lower()
    lower_ent  = entity.lower()
    idx = lower_text.find(lower_ent)
    if idx != -1:
        result = _snap_to_tokens(tok_doc, idx, idx + len(entity))
        if result:
            return result

    # 3. Longest substring match (for surfaceforms slightly longer than the span)
    words = entity.split()
    for length in range(len(words), max(0, len(words) - 2), -1):
        for start in range(len(words) - length + 1):
            sub = " ".join(words[start: start + length])
            if len(sub) < 3:
                continue
            idx = lower_text.find(sub.lower())
            if idx != -1:
                result = _snap_to_tokens(tok_doc, idx, idx + len(sub))
                if result:
                    return result

    return None


# ══════════════════════════════════════════════════════════════════
# 4. OVERLAP RESOLVER
# ══════════════════════════════════════════════════════════════════

def resolve_overlaps(
    spans: List[Tuple[int, int, str]],
) -> List[Tuple[int, int, str]]:
    """
    Remove overlapping spans.
    Resolution order: longer span wins → tie-break by LABEL_PRIORITY.
    """
    if not spans:
        return []

    def _priority(lbl: str) -> int:
        try:
            return LABEL_PRIORITY.index(lbl)
        except ValueError:
            return len(LABEL_PRIORITY)

    # Sort: longest first, then by label priority
    spans = sorted(spans, key=lambda x: (-(x[1] - x[0]), _priority(x[2])))
    kept: List[Tuple[int, int, str]] = []
    for s, e, lbl in spans:
        if not any(s < ke and e > ks for ks, ke, _ in kept):
            kept.append((s, e, lbl))
    return sorted(kept, key=lambda x: x[0])


# ══════════════════════════════════════════════════════════════════
# 5. DATA MINER
# ══════════════════════════════════════════════════════════════════

def mine_local(
    jsonl_path:  str,
    max_samples: int  = 10_000,
    inspect:     bool = False,
) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
    """
    Read local REBEL JSONL and mine NER (char_start, char_end, label) annotations.
    Returns: [(sentence, [(start, end, spacy_label), ...]), ...]
    """
    path = Path(jsonl_path)
    if not path.exists():
        print(f"\n[Miner] ✗ File not found: {jsonl_path}")
        print(f"[Miner]   Check the --data path and try again.")
        sys.exit(1)

    print(f"[Miner] Reading: {path.resolve()}")

    # Blank NLP — only used for tokenisation, no NER needed yet
    nlp_tok = spacy.blank("en")

    ner_data:     List[Tuple[str, List[Tuple[int, int, str]]]] = []
    label_stats:  Dict[str, int] = defaultdict(int)
    n_lines       = 0
    n_skip        = 0
    n_span_miss   = 0

    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            if len(ner_data) >= max_samples:
                break
            n_lines += 1
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                item = json.loads(raw_line)
            except json.JSONDecodeError:
                n_skip += 1
                continue

            parsed = _parse_line(item)
            if not parsed:
                n_skip += 1
                continue

            sentence, triplets = parsed

            # Pre-tokenise once per sentence (reused for all spans)
            tok_doc = nlp_tok.make_doc(sentence)

            # Collect annotations; resolve label conflicts by priority
            raw_spans: Dict[Tuple[int, int], str] = {}

            for subj_surf, pred_str, obj_surf in triplets:
                # Look up relation in table (with partial-match fallback)
                ner_pair = REL_TO_NER.get(pred_str)
                if not ner_pair:
                    for alias, pair in REL_TO_NER.items():
                        if alias in pred_str or pred_str in alias:
                            ner_pair = pair
                            break
                if not ner_pair:
                    continue

                subj_lbl, obj_lbl = ner_pair

                for ent_text, lbl in [(subj_surf, subj_lbl),
                                      (obj_surf,  obj_lbl)]:
                    if not ent_text or len(ent_text) < 2:
                        continue

                    span = find_span(sentence, ent_text, tok_doc)
                    if span:
                        s, e = span
                        existing = raw_spans.get((s, e))
                        if existing is None:
                            raw_spans[(s, e)] = lbl
                        else:
                            # Keep higher-priority label on conflict
                            def _p(l):
                                try: return LABEL_PRIORITY.index(l)
                                except ValueError: return 999
                            if _p(lbl) < _p(existing):
                                raw_spans[(s, e)] = lbl
                        label_stats[lbl] += 1
                    else:
                        n_span_miss += 1

            if not raw_spans:
                continue

            spans = resolve_overlaps(
                [(s, e, lbl) for (s, e), lbl in raw_spans.items()]
            )
            if not spans:
                continue

            ner_data.append((sentence, spans))

            # Show first 5 examples when --inspect is set
            if inspect and len(ner_data) <= 5:
                print(f"\n  ─── Sample {len(ner_data)} ───")
                print(f"  Text : {sentence[:90]}")
                for s, e, lbl in spans:
                    print(f"  [{lbl:<12}] '{sentence[s:e]}'")

    # ── Mining summary ────────────────────────────────────────────
    max_cnt = max(label_stats.values(), default=1)
    print(f"\n[Miner] ── Mining complete ──────────────────────────────")
    print(f"  Lines read       : {n_lines:>8,}")
    print(f"  Lines skipped    : {n_skip:>8,}")
    print(f"  Span misses      : {n_span_miss:>8,}")
    print(f"  Annotated sents  : {len(ner_data):>8,}")
    print(f"\n  Label distribution:")
    for lbl, cnt in sorted(label_stats.items(), key=lambda x: -x[1]):
        bar = "█" * min(40, cnt * 40 // max_cnt)
        print(f"    {lbl:<15} {cnt:>8,}  {bar}")

    return ner_data


# ══════════════════════════════════════════════════════════════════
# 6. CLASS BALANCER
# ══════════════════════════════════════════════════════════════════

def balance_classes(
    data: List[Tuple[str, List[Tuple[int, int, str]]]],
    target_ratio: float = 3.0,
) -> List[Tuple[str, List[Tuple[int, int, str]]]]:
    """
    Oversample sentences containing rare labels so every class has
    at least (max_class_count / target_ratio) examples.
    """
    label_to_idx: Dict[str, List[int]] = defaultdict(list)
    for i, (_, spans) in enumerate(data):
        for _, _, lbl in spans:
            label_to_idx[lbl].append(i)

    if not label_to_idx:
        return data

    max_cnt    = max(len(v) for v in label_to_idx.values())
    min_target = int(max_cnt / target_ratio)
    extras: List[int] = []

    for lbl, idxs in label_to_idx.items():
        if len(idxs) < min_target:
            extras.extend(random.choices(idxs, k=min_target - len(idxs)))

    if extras:
        result = data + [data[i] for i in extras]
        print(f"[Balancer] Added {len(extras)} oversampled examples → {len(result):,} total")
        return result

    return data


# ══════════════════════════════════════════════════════════════════
# 7. EXAMPLE BUILDER
# ══════════════════════════════════════════════════════════════════

def build_examples(
    data: List[Tuple[str, List[Tuple[int, int, str]]]],
    nlp,
) -> List[SpacyExample]:
    """Convert (text, spans) list into spaCy Example objects for training."""
    examples: List[SpacyExample] = []
    n_skip = 0

    for text, spans in data:
        ref_doc = nlp.make_doc(text)
        ents    = []

        for start, end, lbl in spans:
            # Try exact alignment first, expand as fallback
            span = ref_doc.char_span(start, end, label=lbl,
                                     alignment_mode="contract")
            if span is None or len(span) == 0:
                span = ref_doc.char_span(start, end, label=lbl,
                                         alignment_mode="expand")
            if span is not None and len(span) > 0:
                ents.append(span)

        if not ents:
            n_skip += 1
            continue

        try:
            ref_doc.ents = spacy.util.filter_spans(ents)
        except Exception:
            n_skip += 1
            continue

        examples.append(SpacyExample(nlp.make_doc(text), ref_doc))

    if n_skip:
        print(f"  [Builder] Skipped {n_skip} unaligned examples")

    return examples


# ══════════════════════════════════════════════════════════════════
# 8. EVALUATOR
# ══════════════════════════════════════════════════════════════════

def evaluate(
    nlp,
    val_data: List[Tuple[str, List[Tuple[int, int, str]]]],
) -> Dict:
    """Compute span-level Precision / Recall / F1, overall and per-label."""
    per_tp: Dict[str, int] = defaultdict(int)
    per_fp: Dict[str, int] = defaultdict(int)
    per_fn: Dict[str, int] = defaultdict(int)

    for text, gold_spans in val_data:
        doc  = nlp(text)
        pred = {(e.start_char, e.end_char, e.label_) for e in doc.ents}
        gold = {(s, e, lbl) for s, e, lbl in gold_spans}
        for span in pred:
            (per_tp if span in gold else per_fp)[span[2]] += 1
        for span in gold:
            if span not in pred:
                per_fn[span[2]] += 1

    all_labels = sorted(set(list(per_tp) + list(per_fp) + list(per_fn)))

    def _metrics(tp, fp, fn):
        P  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        R  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        F1 = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0
        return P, R, F1

    per_label: Dict[str, Dict] = {}
    for lbl in all_labels:
        P, R, F1 = _metrics(per_tp[lbl], per_fp[lbl], per_fn[lbl])
        per_label[lbl] = {"P": P, "R": R, "F1": F1,
                          "TP": per_tp[lbl], "FP": per_fp[lbl],
                          "FN": per_fn[lbl]}

    otp = sum(per_tp.values())
    ofp = sum(per_fp.values())
    ofn = sum(per_fn.values())
    oP, oR, oF1 = _metrics(otp, ofp, ofn)

    return {
        "overall":   {"P": oP, "R": oR, "F1": oF1,
                      "TP": otp, "FP": ofp, "FN": ofn},
        "per_label": per_label,
    }


def print_eval(scores: Dict, title: str = ""):
    o = scores["overall"]
    if title:
        print(f"\n  ── {title} ──────────────────────────────────────────")
    print(f"  {'Label':<18} {'TP':>5} {'FP':>5} {'FN':>5}  {'P':>6}  {'R':>6}  {'F1':>6}")
    print(f"  {'─'*62}")
    for lbl, m in sorted(scores["per_label"].items()):
        print(f"  {lbl:<18} {m['TP']:>5} {m['FP']:>5} {m['FN']:>5}  "
              f"{m['P']:>6.3f}  {m['R']:>6.3f}  {m['F1']:>6.3f}")
    print(f"  {'─'*62}")
    print(f"  {'OVERALL':<18} {o['TP']:>5} {o['FP']:>5} {o['FN']:>5}  "
          f"{o['P']:>6.3f}  {o['R']:>6.3f}  {o['F1']:>6.3f}")


# ══════════════════════════════════════════════════════════════════
# 9. TRAINER
# ══════════════════════════════════════════════════════════════════

class SpacyNERTrainer:
    """
    Fine-tunes spaCy NER on REBEL-mined annotations, starting from
    the best available pre-trained model (lg → md → sm → blank).

    Training features:
      - Compounding batch size (4 → batch_size)
      - Dropout curriculum: 0.40 → 0.20 over training
      - Early stopping on validation F1 with configurable patience
      - Best-model snapshot saved separately from final checkpoint
      - Per-label F1 printed every 5 epochs
    """

    def __init__(
        self,
        base_model: str = "en_core_web_lg",
        output_dir: str = "./spacy-ner-model",
    ):
        self.output_dir  = output_dir
        self.best_dir    = output_dir + "-best"
        self.best_f1     = 0.0

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.best_dir, exist_ok=True)

        # Load the best available pre-trained model
        self.nlp = None
        for candidate in [base_model, "en_core_web_md", "en_core_web_sm"]:
            try:
                self.nlp = spacy.load(candidate)
                print(f"[Trainer] Base model  : {candidate}")
                break
            except OSError:
                continue

        if self.nlp is None:
            print("[Trainer] WARNING: No pre-trained model found. "
                  "Training from blank English (lower accuracy).")
            print("[Trainer] Run: python -m spacy download en_core_web_lg")
            self.nlp = spacy.blank("en")
            self.nlp.add_pipe("ner", last=True)

        # Ensure NER pipe is present
        if "ner" not in self.nlp.pipe_names:
            self.nlp.add_pipe("ner", last=True)

        # Register all labels
        ner = self.nlp.get_pipe("ner")
        all_labels = set(lbl for pair in REL_TO_NER.values() for lbl in pair)
        all_labels.add("FAC")   # hall of residence / facility
        for lbl in sorted(all_labels):
            ner.add_label(lbl)

        print(f"[Trainer] NER labels   : {sorted(all_labels)}")
        print(f"[Trainer] Output dir   : {output_dir}")
        print(f"[Trainer] Best dir     : {self.best_dir}")

    def train(
        self,
        train_data: List[Tuple[str, List[Tuple[int, int, str]]]],
        val_data:   List[Tuple[str, List[Tuple[int, int, str]]]],
        epochs:     int   = 40,
        batch_size: int   = 32,
        patience:   int   = 6,
    ) -> List[Dict]:
        """Run the training loop. Returns per-epoch history."""

        # Freeze all components except NER
        other_pipes = [p for p in self.nlp.pipe_names if p != "ner"]
        optimizer = self.nlp.initialize()
        try:
            optimizer.learn_rate = 5e-4  # lower LR for fine-tuning
        except AttributeError:
            pass  # some spaCy versions expose alpha instead

        # Compounding batch: starts at 4, grows toward batch_size each update
        batch_sizes = compounding(4.0, float(batch_size), 1.001)

        history:    List[Dict] = []
        no_improve: int        = 0

        print(f"\n[Trainer] ╔══════════════════════════════════════════════╗")
        print(f"[Trainer] ║  spaCy NER Training                          ║")
        print(f"[Trainer] ╠══════════════════════════════════════════════╣")
        print(f"[Trainer] ║  Train samples : {len(train_data):<6,}                    ║")
        print(f"[Trainer] ║  Val samples   : {len(val_data):<6,}                    ║")
        print(f"[Trainer] ║  Max epochs    : {epochs:<6}                    ║")
        print(f"[Trainer] ║  Max batch     : {batch_size:<6} (compounding)      ║")
        print(f"[Trainer] ║  Patience      : {patience:<6}                    ║")
        print(f"[Trainer] ╚══════════════════════════════════════════════╝\n")

        with self.nlp.disable_pipes(*other_pipes):
            for epoch in range(1, epochs + 1):

                # Linear dropout curriculum: 0.40 → 0.20
                dropout = max(0.20, 0.40 - (0.20 * epoch / epochs))

                random.shuffle(train_data)
                losses: Dict[str, float] = {}

                # Build fresh Example objects each epoch
                examples = build_examples(train_data, self.nlp)
                if not examples:
                    print(f"  [WARN] Epoch {epoch}: no valid examples — check data")
                    break

                for batch in minibatch(examples, size=batch_sizes):
                    self.nlp.update(
                        batch,
                        drop=dropout,
                        sgd=optimizer,
                        losses=losses,
                    )

                ner_loss = losses.get("ner", 0.0)
                scores   = evaluate(self.nlp, val_data)
                ov       = scores["overall"]
                f1       = ov["F1"]
                improved = "★" if f1 > self.best_f1 else " "

                print(f"  {improved} Epoch {epoch:>3}/{epochs}"
                      f"  loss={ner_loss:>9.2f}"
                      f"  drop={dropout:.2f}"
                      f"  P={ov['P']:.4f}"
                      f"  R={ov['R']:.4f}"
                      f"  F1={f1:.4f}"
                      f"  (TP={ov['TP']} FP={ov['FP']} FN={ov['FN']})")

                # Per-label snapshot every 5 epochs
                if epoch % 5 == 0 and scores["per_label"]:
                    parts = [
                        f"{lbl}={m['F1']:.2f}"
                        for lbl, m in sorted(scores["per_label"].items())
                        if m["TP"] + m["FN"] > 0
                    ]
                    print(f"    └─ {' '.join(parts)}")

                # CRITICAL FIX: Cast to int/float for JSON serialization
                history.append({
                    "epoch":     int(epoch),
                    "loss":      float(ner_loss),
                    "val_P":     float(ov["P"]),
                    "val_R":     float(ov["R"]),
                    "val_F1":    float(f1),
                    "dropout":   float(dropout),
                    "per_label": {str(lbl): float(m["F1"])
                                  for lbl, m in scores["per_label"].items()},
                })

                # Save best snapshot
                if f1 > self.best_f1:
                    self.best_f1 = f1
                    self.nlp.to_disk(self.best_dir)
                    no_improve = 0
                else:
                    no_improve += 1

                # Early stopping
                if no_improve >= patience:
                    print(f"\n[Trainer] Early stopping: "
                          f"no improvement for {patience} consecutive epochs.")
                    break

        # Save final model (may be slightly worse than best)
        self.nlp.to_disk(self.output_dir)

        # ── Final report ──────────────────────────────────────────
        print(f"\n[Trainer] ╔══════════════════════════════════════════════╗")
        print(f"[Trainer] ║  TRAINING COMPLETE                           ║")
        print(f"[Trainer] ╠══════════════════════════════════════════════╣")
        print(f"[Trainer] ║  Best Val F1  : {self.best_f1:.4f}                     ║")
        print(f"[Trainer] ║  Best model   : {self.best_dir:<29} ║")
        print(f"[Trainer] ║  Final model  : {self.output_dir:<29} ║")
        print(f"[Trainer] ╚══════════════════════════════════════════════╝")

        # Evaluate best checkpoint
        if os.path.exists(self.best_dir):
            best_nlp    = spacy.load(self.best_dir)
            best_scores = evaluate(best_nlp, val_data)
            print_eval(best_scores, "BEST CHECKPOINT — per-label scores")

        # Save training stats
        stats_path = os.path.join(self.output_dir, "training_stats.json")
        with open(stats_path, "w") as fh:
            json.dump(history, fh, indent=2)
        print(f"\n[Trainer] Training stats → {stats_path}")

        return history


# ══════════════════════════════════════════════════════════════════
# 10. CLI
# ══════════════════════════════════════════════════════════════════

def _args():
    ap = argparse.ArgumentParser(
        description="Train spaCy NER on local REBEL JSONL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data",        default="./rebel_dataset/en_train.jsonl",
                    help="Path to REBEL JSONL file")
    ap.add_argument("--samples",     type=int, default=8_000,
                    help="Max sentences to mine from the JSONL")
    ap.add_argument("--epochs",      type=int, default=40,
                    help="Max training epochs (early stopping may end sooner)")
    ap.add_argument("--batch",       type=int, default=32,
                    help="Max batch size (compounding schedule grows up to this)")
    ap.add_argument("--val-ratio",   type=float, default=0.10,
                    help="Fraction of data held out for validation")
    ap.add_argument("--patience",    type=int, default=6,
                    help="Early stopping patience (epochs without F1 improvement)")
    ap.add_argument("--output",      default="./spacy-ner-model",
                    help="Output directory for trained model")
    ap.add_argument("--base-model",  default="en_core_web_lg",
                    help="spaCy pre-trained model to fine-tune from")
    ap.add_argument("--balance",     action="store_true",
                    help="Oversample rare entity labels for class balance")
    ap.add_argument("--inspect",     action="store_true",
                    help="Print first 5 mined annotations to verify schema parsing")
    ap.add_argument("--save-mined",  default=None,
                    help="Save first 200 mined examples as JSON (for inspection)")
    return ap.parse_args()


if __name__ == "__main__":
    args = _args()

    # ── STEP 1: Mine ──────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  STEP 1 — Mining NER annotations from REBEL JSONL")
    print(f"{'═'*62}")

    ner_data = mine_local(
        jsonl_path=args.data,
        max_samples=args.samples,
        inspect=args.inspect,
    )

    if not ner_data:
        print(f"\n[ERROR] No data mined.")
        print(f"  • Check that --data points to your JSONL file: {args.data}")
        print(f"  • Run with --inspect to see what the parser finds.")
        sys.exit(1)

    # Optional: save mined sample for manual inspection
    if args.save_mined:
        sample = [
            {"text":  text,
             "spans": [{"start": s, "end": e, "label": lbl, "text": text[s:e]}
                       for s, e, lbl in spans]}
            for text, spans in ner_data[:200]
        ]
        with open(args.save_mined, "w", encoding="utf-8") as fh:
            json.dump(sample, fh, indent=2, ensure_ascii=False)
        print(f"[Main] Mined sample saved → {args.save_mined}")

    # ── STEP 2: Split ─────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  STEP 2 — Train / Validation split")
    print(f"{'═'*62}")

    random.shuffle(ner_data)
    n_val      = max(200, int(len(ner_data) * args.val_ratio))
    val_data   = ner_data[-n_val:]
    train_data = ner_data[:-n_val]
    print(f"  Train : {len(train_data):,}")
    print(f"  Val   : {len(val_data):,}")

    # ── STEP 3: Balance (optional) ────────────────────────────────
    if args.balance:
        print(f"\n{'═'*62}")
        print(f"  STEP 3 — Class balancing")
        print(f"{'═'*62}")
        train_data = balance_classes(train_data)

    # ── STEP 4: Train ─────────────────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  STEP 4 — Training")
    print(f"{'═'*62}")

    trainer = SpacyNERTrainer(
        base_model=args.base_model,
        output_dir=args.output,
    )
    trainer.train(
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch,
        patience=args.patience,
    )

    # ── STEP 5: Final evaluation ──────────────────────────────────
    print(f"\n{'═'*62}")
    print(f"  STEP 5 — Final evaluation on validation set")
    print(f"{'═'*62}")

    best_path = args.output + "-best"
    eval_nlp  = spacy.load(best_path) if os.path.exists(best_path) else trainer.nlp
    src_label = best_path if os.path.exists(best_path) else args.output

    print(f"  (Evaluating: {src_label})")
    final_scores = evaluate(eval_nlp, val_data)
    print_eval(final_scores, "FINAL RESULTS")

    print(f"\n  ✓ Done. To use in nlp_pipeline.py:")
    print(f"    NLPPipeline(spacy_model_path='{best_path}')\n")