"""
Validation helpers for natural-language-to-Cypher query execution.

The app lets an LLM draft Cypher, but only read-only graph inspection queries
should ever reach Neo4j. This guard rejects writes, admin commands, multi
statement output, and unsupported query starts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


WRITE_KEYWORDS = {
    "CREATE",
    "MERGE",
    "SET",
    "DELETE",
    "DETACH",
    "REMOVE",
    "DROP",
    "LOAD",
    "CALL",
    "CREATE INDEX",
    "CREATE CONSTRAINT",
    "ALTER",
    "GRANT",
    "DENY",
    "REVOKE",
    "START",
    "STOP",
}

ALLOWED_STARTS = ("MATCH", "OPTIONAL MATCH", "WITH")


@dataclass
class CypherValidation:
    ok: bool
    query: str
    reason: str = ""


def strip_markdown_fences(text: str) -> str:
    cleaned = text.replace("```cypher", "").replace("```", "").strip()
    return cleaned


def validate_readonly_cypher(query: str, max_limit: int = 50) -> CypherValidation:
    cleaned = strip_markdown_fences(query)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return CypherValidation(False, cleaned, "Empty query.")

    if ";" in cleaned:
        return CypherValidation(False, cleaned, "Multiple statements are not allowed.")

    upper = cleaned.upper()
    if not upper.startswith(ALLOWED_STARTS):
        return CypherValidation(False, cleaned, "Only MATCH/OPTIONAL MATCH/WITH read queries are allowed.")

    for keyword in WRITE_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}\b", upper):
            return CypherValidation(False, cleaned, f"Blocked keyword: {keyword}.")

    if " RETURN " not in f" {upper} ":
        return CypherValidation(False, cleaned, "Query must return results.")

    if " LIMIT " not in f" {upper} ":
        cleaned = f"{cleaned} LIMIT {max_limit}"

    return CypherValidation(True, cleaned)
