"""
conflict_detector.py
====================
Detects contradicting facts in extracted triples.

Example:
  "Elon Musk founded Tesla"  →  (Elon Musk, FOUNDED_BY, Tesla)
  "Martin Eberhard founded Tesla" → (Martin Eberhard, FOUNDED_BY, Tesla)
  → CONFLICT: Two different subjects for same relation+object

Non-conflicts (multiple subjects OK):
  Larry Page WORKS_FOR Google
  Sergey Brin WORKS_FOR Google  → NOT a conflict
"""
from __future__ import annotations

import re
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Relations where multiple subjects are perfectly normal
NON_CONFLICT_RELATIONS = {
    "WORKS_FOR", "STUDIED_AT", "BORN_IN", "MEMBER_OF",
    "WORKS_IN_FIELD", "AFFILIATED_WITH", "FOUNDED_BY",
    "ACTED_IN", "AUTHORED_BY", "DIRECTED_BY", "COMPOSED_BY",
    "RECEIVED_AWARD", "SPEAKS", "NATIONALITY", "CITIZEN_OF",
    "HEADQUARTERED_IN", "LOCATED_IN", "BASED_IN", "PART_OF",
}

@dataclass
class Conflict:
    relation:     str
    object:       str
    subject_a:    str
    subject_b:    str
    source_a:     str
    source_b:     str
    confidence_a: float
    confidence_b: float

    def __repr__(self):
        return (
            f"CONFLICT on [{self.relation}] → [{self.object}]:\n"
            f"  Version A: ({self.subject_a}) conf={self.confidence_a:.0%}\n"
            f"  Version B: ({self.subject_b}) conf={self.confidence_b:.0%}"
        )


def _normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text.lower().strip())


def detect_conflicts(triples: List[dict]) -> Tuple[List[dict], List[Conflict]]:
    """
    Scan triples for real contradictions.
    Ignores relations where multiple subjects are normal (WORKS_FOR, STUDIED_AT etc.)
    """
    conflicts: List[Conflict] = []
    rel_obj_index: Dict[Tuple[str, str], List[dict]] = {}

    for t in triples:
        key = (_normalize(t["relation"]), _normalize(t["object"]))
        rel_obj_index.setdefault(key, []).append(t)

    conflict_keys = set()
    for key, group in rel_obj_index.items():
        if len(group) < 2:
            continue

        relation_upper = key[0].upper()

        # Skip non-conflict relations
        if relation_upper in NON_CONFLICT_RELATIONS:
            continue

        subjects = [_normalize(t["subject"]) for t in group]
        if len(set(subjects)) < 2:
            continue

        conflict_keys.add(key)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if _normalize(group[i]["subject"]) != _normalize(group[j]["subject"]):
                    conflicts.append(Conflict(
                        relation     = group[i]["relation"],
                        object       = group[i]["object"],
                        subject_a    = group[i]["subject"],
                        subject_b    = group[j]["subject"],
                        source_a     = group[i].get("source", "unknown"),
                        source_b     = group[j].get("source", "unknown"),
                        confidence_a = group[i].get("confidence", 1.0),
                        confidence_b = group[j].get("confidence", 1.0),
                    ))

    clean_triples = []
    for t in triples:
        key = (_normalize(t["relation"]), _normalize(t["object"]))
        t_copy = dict(t)
        t_copy["conflict"] = key in conflict_keys
        clean_triples.append(t_copy)

    return clean_triples, conflicts


def push_conflicts_to_neo4j(conflicts: List[Conflict], session) -> None:
    for c in conflicts:
        try:
            session.run("""
                MERGE (a:Entity {name: $subj_a})
                MERGE (b:Entity {name: $subj_b})
                MERGE (a)-[r:CONFLICT_WITH {
                    relation: $rel, object: $obj,
                    confidence_a: $conf_a, confidence_b: $conf_b,
                    source_a: $src_a, source_b: $src_b
                }]->(b)
            """, {
                "subj_a": c.subject_a, "subj_b": c.subject_b,
                "obj":    c.object,    "rel":    c.relation,
                "conf_a": c.confidence_a, "conf_b": c.confidence_b,
                "src_a":  c.source_a,     "src_b":  c.source_b,
            })
        except Exception as e:
            print(f"[Conflict] Error: {e}")