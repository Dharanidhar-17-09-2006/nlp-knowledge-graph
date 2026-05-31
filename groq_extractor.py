"""
Groq-backed triple extraction and hybrid fusion.

This module keeps the external LLM extractor small, then combines Groq and
REBEL outputs with lexical grounding, confidence scoring, and provenance.
"""
from __future__ import annotations

import json
import re
import time
from typing import Dict, List, Tuple

import requests

import os
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are a precise knowledge graph extractor.
Extract Subject-Relation-Object triples from the given sentence.
Rules:
- Only extract facts explicitly stated in the text. Never invent facts.
- Do NOT extract triples where subject and object are the same.
- Do NOT use pronouns (he, she, it, they) as subjects or objects.
- Relations must be concise (2-4 words max).
- Return ONLY valid JSON, no explanation, no markdown.
- Format: [{"subject": "...", "relation": "...", "object": "..."}]
- If no clear triple exists, return []

Examples:
Input: "Elon Musk founded SpaceX in California."
Output: [{"subject": "Elon Musk", "relation": "founded", "object": "SpaceX"}, {"subject": "SpaceX", "relation": "located in", "object": "California"}]

Input: "He studies Computer Science."
Output: []

Input: "Srishwan is a student at IIT Kharagpur."
Output: [{"subject": "Srishwan", "relation": "studies at", "object": "IIT Kharagpur"}]
"""


class GroqExtractor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def extract(self, sentence: str) -> List[Tuple[str, str, str]]:
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f'Extract triples from: "{sentence}"'},
            ],
            "temperature": 0.0,
            "max_tokens": 512,
        }

        for attempt in range(5):
            try:
                resp = requests.post(
                    GROQ_API_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=15,
                    proxies={"http": None, "https": None},
                )

                if resp.status_code == 429:
                    wait_time = 2 ** attempt
                    print(f"[Groq] Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
                content = re.sub(r"```json|```", "", content).strip()
                triples_raw = json.loads(content)

                pronouns = {"he", "she", "it", "they", "we", "i", "him", "her", "them"}
                results = []
                for triple in triples_raw:
                    s = triple.get("subject", "").strip()
                    r = triple.get("relation", "").strip()
                    o = triple.get("object", "").strip()
                    if not s or not r or not o:
                        continue
                    if s.lower() in pronouns or o.lower() in pronouns:
                        continue
                    if s.lower() == o.lower():
                        continue
                    results.append((s, r, o))

                time.sleep(1)
                return results

            except Exception as e:
                print(f"[Groq] Error: {e}")
                return []

        return []


def lexical_overlap_score(triple: Tuple[str, str, str], source: str) -> float:
    source_words = set(re.findall(r"\b\w+\b", source.lower()))
    triple_words = set(re.findall(r"\b\w+\b", f"{triple[0]} {triple[2]}".lower()))
    stopwords = {"the", "a", "an", "is", "was", "are", "were", "of", "in", "at", "by", "for"}
    triple_words -= stopwords
    if not triple_words:
        return 0.0
    return len(triple_words & source_words) / len(triple_words)


def normalize_for_fusion(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _fallback_normalize_relation(rel: str) -> str:
    aliases = {
        "founded by": "FOUNDED_BY",
        "founded": "FOUNDED_BY",
        "co-founded": "FOUNDED_BY",
        "studied at": "STUDIED_AT",
        "studies at": "STUDIED_AT",
        "student at": "STUDIED_AT",
        "located in": "LOCATED_IN",
        "headquartered in": "HEADQUARTERED_IN",
        "born in": "BORN_IN",
        "member of": "MEMBER_OF",
        "part of": "PART_OF",
        "created by": "CREATED_BY",
        "married to": "MARRIED_TO",
    }
    key = rel.lower().strip().replace("_", " ")
    return aliases.get(key, "_".join(re.sub(r"[^A-Za-z0-9\s]", " ", rel).upper().split()) or "RELATED_TO")


def _upsert_fused(
    fused: Dict[Tuple[str, str, str], dict],
    subject: str,
    relation: str,
    obj: str,
    confidence: float,
    extractor: str,
    lexical_score: float,
    subject_label: str = "Entity",
    object_label: str = "Entity",
) -> None:
    key = (normalize_for_fusion(subject), relation, normalize_for_fusion(obj))
    candidate = {
        "subject": subject,
        "relation": relation,
        "object": obj,
        "confidence": round(min(max(confidence, 0.0), 1.0), 3),
        "source": extractor,
        "extractors": [extractor],
        "lexical_score": round(lexical_score, 3),
        "subject_label": subject_label,
        "object_label": object_label,
    }

    if key not in fused:
        fused[key] = candidate
        return

    current = fused[key]
    if extractor not in current["extractors"]:
        current["extractors"].append(extractor)
    current["confidence"] = round(min(max(current["confidence"], confidence) + 0.15, 1.0), 3)
    current["lexical_score"] = round(max(current.get("lexical_score", 0.0), lexical_score), 3)
    current["source"] = "hybrid_corroborated"
    if current.get("subject_label", "Entity") == "Entity" and subject_label != "Entity":
        current["subject_label"] = subject_label
    if current.get("object_label", "Entity") == "Entity" and object_label != "Entity":
        current["object_label"] = object_label


def fuse_triples(
    rebel_triples,
    groq_triples,
    source_sentence,
    lexical_threshold=0.7,
    rebel_conf_threshold=0.35,
):
    """
    Fuse Groq and REBEL triples with provenance.

    Groq contributes grounded triples. REBEL also contributes high-confidence
    grounded triples when Groq misses them, so Hybrid is a real union instead
    of only Groq with a confidence boost.
    """
    try:
        from NLP_pipeline import normalize_relation
    except Exception:
        normalize_relation = _fallback_normalize_relation

    fused = {}

    for s, r, o in groq_triples:
        overlap = lexical_overlap_score((s, r, o), source_sentence)
        if overlap < lexical_threshold:
            continue
        _upsert_fused(
            fused,
            s,
            normalize_relation(r),
            o,
            confidence=min(overlap + 0.3, 0.88),
            extractor="groq",
            lexical_score=overlap,
        )

    for item in rebel_triples:
        if len(item) == 4:
            s, r, o, conf = item
            s_label = "Entity"
            o_label = "Entity"
        else:
            s, r, o, conf, s_label, o_label = item
        r_norm = normalize_relation(r)
        overlap = lexical_overlap_score((s, r_norm, o), source_sentence)
        if overlap < lexical_threshold or conf < rebel_conf_threshold:
            continue
        _upsert_fused(
            fused,
            s,
            r_norm,
            o,
            confidence=min((conf * 0.75) + (overlap * 0.25), 0.9),
            extractor="rebel",
            lexical_score=overlap,
            subject_label=s_label,
            object_label=o_label,
        )

    return sorted(fused.values(), key=lambda t: t["confidence"], reverse=True)
