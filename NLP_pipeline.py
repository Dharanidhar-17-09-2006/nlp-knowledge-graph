"""
nlp_pipeline.py  ── v4 (Production)
=====================================
Inference pipeline that loads the models trained by:
  train_rebel.py     → ./rebel-finetuned/
  train_spacy_ner.py → ./spacy-ner-model/
"""
from __future__ import annotations


import os
import re
import unicodedata
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import spacy
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Silence warnings
warnings.filterwarnings("ignore", module="spacy")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

try:
    from rapidfuzz import fuzz as _fuzz
    _FUZZY = True
except ImportError:
    _FUZZY = False

# ══════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════

@dataclass
class Entity:
    text:       str
    label:      str = "Entity"
    properties: dict = field(default_factory=dict)

    @property
    def canonical(self) -> str:
        return self.text.lower().strip()

    def __hash__(self):   return hash(self.canonical)
    def __eq__(self, o):  return self.canonical == o.canonical
    def __repr__(self):   return f"{self.text}:{self.label}"

@dataclass
class Triplet:
    subject:         Entity
    relation:        str
    obj:             Entity
    confidence:      float = 1.0
    source_sentence: str   = ""

    def __repr__(self):
        return (f"({self.subject})"
                f" -[{self.relation}|{self.confidence:.0%}]->"
                f" ({self.obj})")

# ══════════════════════════════════════════════════════════════════
# LABEL MAPS
# ══════════════════════════════════════════════════════════════════

SPACY_TO_NEO4J: Dict[str, str] = {
    "PERSON":      "Person", "ORG":         "Organization",
    "GPE":         "Location", "LOC":         "Location",
    "FAC":         "Facility", "PRODUCT":     "Product",
    "EVENT":       "Event", "WORK_OF_ART": "CreativeWork",
    "LAW":         "Law", "LANGUAGE":    "Language",
    "DATE":        "Date", "TIME":        "Time",
    "MONEY":       "Money", "NORP":        "Group",
    "CARDINAL":    "Number", "ORDINAL":     "Number",
    "PERCENT":     "Percentage", "QUANTITY":    "Quantity",
}

REL_TYPE_HINTS: Dict[str, Tuple[str, str]] = {
    "place of birth":          ("Person",       "Location"),
    "place of death":          ("Person",       "Location"),
    "date of birth":           ("Person",       "Date"),
    "date of death":           ("Person",       "Date"),
    "country of citizenship":  ("Person",       "Location"),
    "educated at":             ("Person",       "Organization"),
    "employer":                ("Person",       "Organization"),
    "occupation":              ("Person",       "Entity"),
    "position held":           ("Person",       "Entity"),
    "award received":          ("Person",       "CreativeWork"),
    "member of":               ("Person",       "Organization"),
    "spouse":                  ("Person",       "Person"),
    "child":                   ("Person",       "Person"),
    "parent":                  ("Person",       "Person"),
    "sibling":                 ("Person",       "Person"),
    "field of work":           ("Person",       "Entity"),
    "student of":              ("Person",       "Person"),
    "notable work":            ("Person",       "CreativeWork"),
    "sex or gender":           ("Person",       "Entity"),
    "religion":                ("Person",       "Group"),
    "nationality":             ("Person",       "Group"),
    "doctoral advisor":        ("Person",       "Person"),
    "doctoral student":        ("Person",       "Person"),
    "ethnic group":            ("Person",       "Group"),
    "languages spoken":        ("Person",       "Language"),
    "founded by":              ("Organization", "Person"),
    "inception":               ("Organization", "Date"),
    "dissolved":               ("Organization", "Date"),
    "headquarters location":   ("Organization", "Location"),
    "country":                 ("Organization", "Location"),
    "developer":               ("Product",      "Organization"),
    "manufacturer":            ("Product",      "Organization"),
    "publisher":               ("CreativeWork", "Organization"),
    "owned by":                ("Organization", "Entity"),
    "subsidiary":              ("Organization", "Organization"),
    "parent organization":     ("Organization", "Organization"),
    "located in":              ("Entity",       "Location"),
    "capital":                 ("Location",     "Location"),
    "official language":       ("Location",     "Language"),
    "head of government":      ("Location",     "Person"),
    "head of state":           ("Location",     "Person"),
    "author":                  ("CreativeWork", "Person"),
    "director":                ("CreativeWork", "Person"),
    "cast member":             ("CreativeWork", "Person"),
    "genre":                   ("CreativeWork", "Entity"),
    "instance of":             ("Entity",       "Entity"),
    "subclass of":             ("Entity",       "Entity"),
    "part of":                 ("Entity",       "Entity"),
    "has part":                ("Entity",       "Entity"),
    "operator":                ("Entity",       "Organization"),
    "screenwriter":            ("CreativeWork", "Person"),
    "composer":                ("CreativeWork", "Person"),
}

# Expand RELATION_MAP with domain-specific aliases
RELATION_MAP: Dict[str, str] = {
    "instance of":              "IS_A",
    "subclass of":              "SUBCLASS_OF",
    "part of":                  "PART_OF",
    "has part":                 "HAS_PART",
    "member of":                "MEMBER_OF",
    "employer":                 "WORKS_FOR",
    "occupation":               "HAS_OCCUPATION",
    "position held":            "HOLDS_POSITION",
    "educated at":              "STUDIED_AT",
    "student at":               "STUDIED_AT",
    "studies in":               "STUDIED_AT",
    "enrolled in":              "STUDIED_AT",
    "studies":                  "STUDIED_AT",
    "student of":               "STUDENT_OF",
    "place of birth":           "BORN_IN",
    "place of death":           "DIED_IN",
    "date of birth":            "BORN_ON",
    "date of death":            "DIED_ON",
    "country of citizenship":   "CITIZEN_OF",
    "nationality":              "NATIONALITY",
    "religion":                 "FOLLOWS_RELIGION",
    "sex or gender":            "HAS_GENDER",
    "located in":               "LOCATED_IN",
    "headquarters location":    "HEADQUARTERED_IN",
    "capital":                  "HAS_CAPITAL",
    "official language":        "HAS_OFFICIAL_LANGUAGE",
    "country":                  "IN_COUNTRY",
    "founded by":               "FOUNDED_BY",
    "inception":                "FOUNDED_ON",
    "dissolved":                "DISSOLVED_ON",
    "developer":                "DEVELOPED_BY",
    "manufacturer":             "MANUFACTURED_BY",
    "publisher":                "PUBLISHED_BY",
    "owned by":                 "OWNED_BY",
    "subsidiary":               "HAS_SUBSIDIARY",
    "parent organization":      "PARENT_ORG",
    "creator":                  "CREATED_BY",
    "author":                   "AUTHORED_BY",
    "director":                 "DIRECTED_BY",
    "cast member":              "ACTED_IN",
    "genre":                    "HAS_GENRE",
    "award received":           "RECEIVED_AWARD",
    "notable work":             "KNOWN_FOR",
    "field of work":            "WORKS_IN_FIELD",
    "spouse":                   "MARRIED_TO",
    "child":                    "PARENT_OF",
    "parent":                   "CHILD_OF",
    "sibling":                  "SIBLING_OF",
    "head of government":       "GOVERNED_BY",
    "head of state":            "HEAD_OF_STATE",
    "follows":                  "FOLLOWS",
    "followed by":              "FOLLOWED_BY",
    "doctoral advisor":         "ADVISED_BY",
    "doctoral student":         "ADVISED",
    "ethnic group":             "ETHNICITY",
    "languages spoken":         "SPEAKS",
    "operator":                 "OPERATED_BY",
    "screenwriter":             "WRITTEN_BY",
    "composer":                 "COMPOSED_BY",
    "works for":                "WORKS_FOR",
    "works at":                 "WORKS_FOR",
    "stays in":                 "LIVES_IN",
    "resides in":               "LIVES_IN",
    "lives in":                 "LIVES_IN",
    "born in":                  "BORN_IN",
    "is a":                     "IS_A",
    "affiliated with":          "AFFILIATED_WITH",
    "associated with":          "ASSOCIATED_WITH",
    "belongs to":               "BELONGS_TO",
    "based in":                 "BASED_IN",
    "language":                 "USES_LANGUAGE",
    "language used":            "USES_LANGUAGE",
}

# Reverse lookup to allow Labeler to handle Normalized variants
for k, v in list(REL_TYPE_HINTS.items()):
    norm_key = RELATION_MAP.get(k)
    if norm_key and norm_key.lower() not in REL_TYPE_HINTS:
        REL_TYPE_HINTS[norm_key.lower()] = v

def normalize_relation(rel: str) -> str:
    lower = rel.lower().strip()
    # 1. Exact Match
    if lower in RELATION_MAP:
        return RELATION_MAP[lower]
    
    # 2. Longest-match first: only match if the alias IS the entire lower string,
    #    or lower IS the entire alias (word-boundary safe via whole-word check)
    for alias in sorted(RELATION_MAP.keys(), key=len, reverse=True):
        # Require that the alias matches as a complete phrase, not a fragment
        if alias == lower:
            return RELATION_MAP[alias]
        # Accept alias as a prefix/suffix only when separated by whitespace
        if (lower.startswith(alias + " ") or lower.endswith(" " + alias)
                or (" " + alias + " ") in (" " + lower + " ")):
            return RELATION_MAP[alias]
            
    # 3. Fallback regex formatting
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", rel)
    return "_".join(cleaned.upper().split()) or "RELATED_TO"

# ══════════════════════════════════════════════════════════════════
# TEXT PREPROCESSOR & ENTITY LABELER
# ══════════════════════════════════════════════════════════════════

SUBJ_PRONOUNS = {"he", "she", "they", "it", "we", "i", "who"}
OBJ_PRONOUNS  = {"him", "her", "them", "it", "us", "me", "whom"}

class TextPreprocessor:
    def __init__(self, nlp):
        self.nlp = nlp

    def clean(self, text: str) -> str:
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii")
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\.{2,}", ".", text)
        text = re.sub(r"\s([,\.;:!?])", r"\1", text)
        text = re.sub(r"([,\.;:!?])([A-Za-z])", r"\1 \2", text)
        return text

    def split_sentences(self, text: str) -> List[str]:
        return [s.text.strip() for s in self.nlp(text).sents if len(s.text.strip()) > 5]

    def resolve_pronouns(self, sentences: List[str]) -> List[str]:
        """Smarter rule-based coreference using spaCy's native tagging."""
        resolved = []
        last_male = None
        last_female = None
        last_org_loc = None

        for sent in sentences:
            doc = self.nlp(sent)
            
            # 1. Update memory with entities from this sentence
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    # Simple heuristic for gendered pronouns
                    first_name = ent.text.split()[0].lower()
                    if first_name in ["marie", "alice", "sarah"]: 
                        last_female = ent.text
                    else:
                        last_male = ent.text
                elif ent.label_ in ["ORG", "GPE", "LOC", "PRODUCT", "FAC"]:
                    last_org_loc = ent.text

            # 2. Replace pronouns based on memory
            out_tokens = []
            for tok in doc:
                word = tok.text.lower()
                ws = tok.whitespace_
                
                if word in ["he", "him", "his"] and last_male:
                    out_tokens.append(last_male + ws)
                elif word in ["she", "her", "hers"] and last_female:
                    out_tokens.append(last_female + ws)
                elif word in ["it", "its"] and last_org_loc:
                    out_tokens.append(last_org_loc + ws)
                else:
                    out_tokens.append(tok.text_with_ws)
                    
            resolved.append("".join(out_tokens).strip())
            
        return resolved


class EntityLabeler:
    FALLBACK_MODELS = ["en_core_web_lg", "en_core_web_trf", "en_core_web_md", "en_core_web_sm"]

    def __init__(self, spacy_model_path: Optional[str] = None):
        self.nlp = None

        if spacy_model_path and os.path.exists(spacy_model_path):
            try:
                self.nlp = spacy.load(spacy_model_path)
                print(f"[Labeler] Loaded fine-tuned NER: {spacy_model_path}")
            except Exception as e:
                print(f"[Labeler] Could not load {spacy_model_path}: {e}")

        if not self.nlp:
            for m in self.FALLBACK_MODELS:
                try:
                    self.nlp = spacy.load(m)
                    print(f"[Labeler] Loaded spaCy: {m}")
                    break
                except OSError:
                    continue

        if not self.nlp:
            raise RuntimeError("No spaCy model found.\nRun: python -m spacy download en_core_web_lg")
        if not self.nlp.has_pipe("sentencizer") and not self.nlp.has_pipe("parser"):
            self.nlp.add_pipe("sentencizer")
        self._cache: Dict[Tuple, str] = {}

    def label(self, text: str, context: str = "", relation: str = "", role: str = "subject") -> str:
        key = (text.lower().strip(), relation.lower(), role)
        if key in self._cache:
            return self._cache[key]
        result = self._label_impl(text, context, relation, role)
        self._cache[key] = result
        return result

    def _label_impl(self, text, context, relation, role) -> str:
        rel_l = relation.lower().strip()
        if rel_l in REL_TYPE_HINTS:
            hint_s, hint_o = REL_TYPE_HINTS[rel_l]
            hint = hint_s if role == "subject" else hint_o
            if hint != "Entity":
                return hint

        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.text.lower().strip() == text.lower().strip():
                return SPACY_TO_NEO4J.get(ent.label_, "Entity")

        if context:
            doc_ctx = self.nlp(context)
            key_l   = text.lower()
            key_tok = set(key_l.split())
            for ent in doc_ctx.ents:
                el = ent.text.lower()
                if key_l in el or el in key_l or (key_tok & set(el.split())):
                    return SPACY_TO_NEO4J.get(ent.label_, "Entity")

        for alias, (hs, ho) in REL_TYPE_HINTS.items():
            if alias in rel_l or rel_l in alias:
                hint = hs if role == "subject" else ho
                if hint != "Entity":
                    return hint

        for tok in doc:
            if tok.pos_ == "PROPN":
                return "Entity"
        if text and text[0].isupper():
            return "Entity"

        return "Entity"


class EntityDeduplicator:
    THRESHOLD = 80

    def __init__(self):
        self._canon: Dict[str, str] = {}

    def get_canonical(self, text: str) -> str:
        if not text: 
            return text
        
        clean_text = " ".join(text.split())
        key = clean_text.lower()

        if key in self._canon:
            return self._canon[key]

        for existing, canon in list(self._canon.items()):
            if self._similar(key, existing):
                better = clean_text if len(clean_text) >= len(canon) else canon
                self._canon[key] = self._canon[existing] = self._canon[canon] = better
                return better
        
        self._canon[key] = clean_text
        return clean_text

    def _similar(self, a: str, b: str) -> bool:
        al, bl = a.lower(), b.lower()
        # Only allow substring match if both strings are meaningful length
        # This prevents "AI" matching "Hairbrush" or "Cat" matching "Caterpillar"
        if min(len(al), len(bl)) >= 4:
            # Check substring only if the shorter one is a whole word inside the longer
            shorter, longer = (al, bl) if len(al) <= len(bl) else (bl, al)
            # Must appear as a whole word boundary, not just any substring
            import re as _re
            if _re.search(r'\b' + _re.escape(shorter) + r'\b', longer):
                return True
        # Exact match always valid
        if al == bl:
            return True
        # Only apply fuzzy threshold on strings long enough to be meaningful
        if min(len(al), len(bl)) < 4:
            return False
        if _FUZZY:
            return _fuzz.token_sort_ratio(al, bl) >= self.THRESHOLD
        return False

# ══════════════════════════════════════════════════════════════════
# REBEL EXTRACTOR
# ══════════════════════════════════════════════════════════════════

class REBELExtractor:
    MAX_INPUT_TOKENS = 256   # increased from 128 — REBEL-large supports up to 1024 tokens;
    OVERLAP_TOKENS   = 32    # most sentences fit in 256, eliminating entity boundary splits

    def __init__(
        self,
        model_path:    str = "Babelscape/rebel-large",
        device:        str = "cpu",
        num_beams:     int = 3,
        num_sequences: int = 1,
    ):
        if os.path.exists(model_path):
            print(f"[REBEL] Loading fine-tuned model: {model_path}")
        else:
            print(f"[REBEL] Loading base model: {model_path}")

        self.tok   = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
        self.device       = device
        self.num_beams    = num_beams
        self.num_sequences = min(num_sequences, num_beams)
        self.dedup = EntityDeduplicator()
        print("[REBEL] Ready.")

    def _parse(self, text: str) -> List[Tuple[str, str, str]]:
        out  = []
        text = re.sub(r"<s>|</s>|<pad>", "", text).strip()
        for part in text.split("<triplet>"):
            part = part.strip()
            if not part or "<subj>" not in part or "<obj>" not in part:
                continue
            try:
                s_raw, rest  = part.split("<subj>", 1)
                o_raw, r_raw = rest.split("<obj>",  1)
                clean = lambda x: re.sub(r"<[^>]+>", "", x).strip()
                s, o, r = clean(s_raw), clean(o_raw), clean(r_raw)
                if s and o and r and len(s) > 1 and len(o) > 1 and s.lower() != o.lower():
                    # Normalize relation string immediately upon extraction
                    out.append((s, normalize_relation(r), o))
            except (ValueError, IndexError):
                continue
        return out

    def _chunks(self, text: str) -> List[str]:
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) <= self.MAX_INPUT_TOKENS:
            return [text]
        step   = self.MAX_INPUT_TOKENS - self.OVERLAP_TOKENS
        chunks = []
        for start in range(0, len(ids), step):
            chunk = self.tok.decode(ids[start:start + self.MAX_INPUT_TOKENS], skip_special_tokens=True)
            chunks.append(chunk)
            if start + self.MAX_INPUT_TOKENS >= len(ids):
                break
        return chunks

    def _generate(self, chunk: str) -> List[Tuple[str, str, str, float]]:
        """
        Run beam search and return (subj, rel, obj, beam_score) tuples.

        """
        inp = self.tok(
            chunk, return_tensors="pt", padding=True,
            truncation=True, max_length=self.MAX_INPUT_TOKENS + 10,
        ).to(self.device)

        input_len = inp["input_ids"].shape[1]
        min_len   = min(10, input_len + 2)

        out = self.model.generate(
            inp["input_ids"],
            attention_mask        = inp["attention_mask"],
            max_length            = 256,
            min_length            = min_len,
            num_beams             = self.num_beams,
            num_return_sequences  = self.num_sequences,
            early_stopping        = True,
            return_dict_in_generate = True,
            output_scores         = True,
        )

        # sequences_scores: log-prob of each returned sequence (shape: num_sequences)
        seq_scores = out.sequences_scores          # tensor of shape [num_sequences]
        # Convert log-probs to relative probabilities via softmax
        probs = torch.softmax(seq_scores, dim=0)  # sums to 1 across returned sequences

        raw: List[Tuple[str, str, str, float]] = []
        for seq, prob in zip(out.sequences, probs):
            decoded  = self.tok.decode(seq, skip_special_tokens=False)
            triplets = self._parse(decoded)
            score    = float(prob)
            for t in triplets:
                raw.append((*t, score))
        return raw

    def extract_with_confidence(self, sentence: str) -> List[Tuple[str, str, str, float]]:
        """
        Return (subj, rel, obj, confidence) tuples sorted by confidence desc.

        Confidence is now the SUM of softmax-normalised beam probabilities for
        all sequences that produced this triplet.  This means:
          - A triplet appearing in only 1 of 4 sequences at low beam score → low conf
          - A triplet appearing in 3 of 4 sequences at high beam score  → high conf
        This is a real model signal, not just a count ratio.
        """
        # _generate now returns (s, r, o, score) 4-tuples
        raw: List[Tuple[str, str, str, float]] = []
        for chunk in self._chunks(sentence):
            raw.extend(self._generate(chunk))
        if not raw:
            return []

        score_acc: Dict[tuple, float] = {}
        originals: Dict[tuple, Tuple[str, str, str]] = {}
        for s, r_norm, o, score in raw:
            # Canonicalize entities before keying — merges typo variants
            s_canon = self.dedup.get_canonical(s) if hasattr(self, 'dedup') else s
            o_canon = self.dedup.get_canonical(o) if hasattr(self, 'dedup') else o
            key = (s_canon.lower(), r_norm.lower(), o_canon.lower())
            score_acc[key] = score_acc.get(key, 0.0) + score
            if key not in originals:
                originals[key] = (s_canon, r_norm, o_canon)

        results = [
            (originals[k][0], originals[k][1], originals[k][2], min(conf, 1.0))
            for k, conf in score_acc.items()
        ]
        results.sort(key=lambda x: x[3], reverse=True)

        # Keep only best triple per (subject, object) pair
        best: Dict[tuple, Tuple[str, str, str, float]] = {}
        for s, r, o, conf in results:
            key = (s.lower(), o.lower())
            if key not in best or conf > best[key][3]:
                best[key] = (s, r, o, conf)
        results = sorted(best.values(), key=lambda x: x[3], reverse=True)
        return results

# ══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

class NLPPipeline:
    REBEL_FINETUNED = "./rebel-finetuned"
    SPACY_FINETUNED = "./spacy-ner-model-best"

    def __init__(
        self,
        rebel_model_path: str           = REBEL_FINETUNED,
        spacy_model_path: Optional[str] = SPACY_FINETUNED,
        device:           str           = "cpu",
        num_beams:        int           = 3,
        num_sequences:    int           = 1,
        min_confidence:   float         = 0.25,  # raised from 0.1 — real beam scores now used
    ):
        if not os.path.exists(rebel_model_path):
            rebel_model_path = "Babelscape/rebel-large"

        self.labeler  = EntityLabeler(spacy_model_path)
        self.rebel    = REBELExtractor(rebel_model_path, device, num_beams, num_sequences)
        self.preproc  = TextPreprocessor(self.labeler.nlp)
        self.dedup    = EntityDeduplicator()
        self.min_conf = min_confidence

    def process(self, text: str, verbose: bool = True) -> List[Triplet]:
        text      = self.preproc.clean(text)
        sentences = self.preproc.split_sentences(text)
        sentences = self.preproc.resolve_pronouns(sentences)

        if verbose:
            print(f"\n[Pipeline] {len(sentences)} sentences.\n{'─'*60}")

        all_triplets: List[Triplet] = []

        for sent in sentences:
            if verbose: print(f"▶ {sent}")

            raw = self.rebel.extract_with_confidence(sent)
            if not raw:
                if verbose: print("  → (nothing extracted)\n")
                continue

            for s_text, r_norm, o_text, conf in raw:
                if conf < self.min_conf:
                    continue

                s_canon = self.dedup.get_canonical(s_text)
                o_canon = self.dedup.get_canonical(o_text)

                s_label = self.labeler.label(s_canon, sent, r_norm, "subject")
                o_label = self.labeler.label(o_canon, sent, r_norm, "object")

                t = Triplet(
                    subject         = Entity(s_canon, s_label),
                    relation        = r_norm,
                    obj             = Entity(o_canon, o_label),
                    confidence      = round(conf, 3),
                    source_sentence = sent,
                )
                all_triplets.append(t)

                if verbose: print(f"  [{conf:.0%}] {t}")

            if verbose: print()

        best: Dict[Tuple, Triplet] = {}
        for t in all_triplets:
            key = (t.subject.canonical, t.relation, t.obj.canonical)
            if key not in best or t.confidence > best[key].confidence:
                best[key] = t

        result = sorted(best.values(), key=lambda x: x.confidence, reverse=True)

        if verbose:
            print(f"{'─'*60}\n[Pipeline] ✓ {len(result)} unique triplets.\n")

        return result

# ══════════════════════════════════════════════════════════════════
# CYPHER GENERATOR
# ══════════════════════════════════════════════════════════════════

def _escape_cypher(text: str) -> str:
    return text.replace("\\", "\\\\").replace("'", "\\'")

def triplets_to_cypher(
    triplets,
    include_confidence: bool = True,
    min_confidence: float = 0.0,
) -> str:
    if not triplets:
        return "// No triplets extracted."

    lines = ["// Auto-generated Cypher — "]
    seen: set = set()

    for t in triplets:
        if hasattr(t, "subject"):
            s_text, s_label = t.subject.text, t.subject.label
            o_text, o_label = t.obj.text, t.obj.label
            rel, conf = t.relation, getattr(t, "confidence", 1.0)
        else:
            s_text, s_label = t.get("subject", "Unknown"), t.get("subject_label", "Entity")
            o_text, o_label = t.get("object", "Unknown"), t.get("object_label", "Entity")
            rel, conf = t.get("relation", "RELATED_TO"), t.get("confidence", 1.0)

        if conf < min_confidence:
            continue

        key = (s_text.lower(), rel, o_text.lower())
        if key in seen:
            continue
            
        idx = len(seen)
        seen.add(key)

        se = _escape_cypher(s_text)
        oe = _escape_cypher(o_text)

        sl = re.sub(r"[^A-Za-z0-9]", "", s_label) or "Entity"
        ol = re.sub(r"[^A-Za-z0-9]", "", o_label) or "Entity"

        rel_props = f" {{confidence: {conf:.3f}}}" if include_confidence else ""

        lines.append("")
        lines.append(f"MERGE (s_{idx}:{sl} {{name: '{se}'}})")
        lines.append(f"MERGE (o_{idx}:{ol} {{name: '{oe}'}})")
        lines.append(f"MERGE (s_{idx})-[:{rel}{rel_props}]->(o_{idx})")

    return "\n".join(lines)



# ══════════════════════════════════════════════════════════════════
# CLI EXECUTION (Run this file directly to test)
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # A quick test block to verify the pipeline works standalone
    TEXT = """
    Srishwan is a student at IIT Kharagpur. He studies Computer Science.
    Google was founded by Larry Page and Sergey Brin in Menlo Park.
    Albert Einstein was born in Ulm, Germany. He developed the theory of relativity.
    """

    print("Initializing Pipeline...")
    # Using small beam numbers for a fast local test
    pipeline = NLPPipeline(num_beams=3, num_sequences=1, min_confidence=0.1)
    
    print("Processing Text...")
    triplets = pipeline.process(TEXT.strip(), verbose=True)

    print("\n--- GENERATED CYPHER ---")
    print(triplets_to_cypher(triplets))