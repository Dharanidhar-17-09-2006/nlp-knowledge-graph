"""
main_neo4j.py  ── v2 (Fully Automatic)
=======================================
End-to-end pipeline — zero manual steps:
  1. Accept input text (CLI arg, file, or built-in demo)
  2. Run NLPPipeline  →  List[Triplet]
  3. Auto-connect to Neo4j and push all nodes + relationships
  4. Immediately query Neo4j back and print the full graph in the terminal
  5. Print per-node neighbourhood summaries

Dependencies (install once):
    pip install neo4j spacy transformers torch rapidfuzz
    python -m spacy download en_core_web_lg

Usage examples:
    # Quickstart — uses demo text, default localhost Neo4j
    python main_neo4j.py --password <your-neo4j-password>

    # Your own text file
    python main_neo4j.py --input sentences.txt --password secret

    # AuraDB or remote instance
    python main_neo4j.py --uri neo4j+s://xxxx.databases.neo4j.io --user neo4j --password secret

    # Wipe DB first, then rebuild (useful during testing)
    python main_neo4j.py --clear-db --password secret

    # Just print Cypher, skip Neo4j entirely
    python main_neo4j.py --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ── Local pipeline import ────────────────────────────────────────────────────
try:
    from NLP_pipeline import NLPPipeline, Triplet, triplets_to_cypher
except ModuleNotFoundError:
    try:
        from nlp_pipeline import NLPPipeline, Triplet, triplets_to_cypher
    except ModuleNotFoundError:
        sys.exit(
            "[ERROR] Cannot find NLP_pipeline.py / nlp_pipeline.py.\n"
            "Place this script in the same directory as your pipeline file."
        )

# ── Neo4j driver ─────────────────────────────────────────────────────────────
try:
    from neo4j import GraphDatabase, exceptions as neo4j_exc
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False


# ═════════════════════════════════════════════════════════════════════════════
# TERMINAL DISPLAY HELPERS
# ═════════════════════════════════════════════════════════════════════════════

W = 72  # box width

def _safe_label(label: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9]", "", label)
    return clean if clean else "Entity"

def _box(title: str):
    inner = W - 4
    pad   = max(0, inner - len(title))
    print(f"\n╔══ {title} {'═' * pad}╗")

def _box_end():
    print("╚" + "═" * (W - 1) + "╝")

def _row(text: str = ""):
    inner = W - 4
    # Hard-wrap long lines
    while len(text) > inner:
        print(f"║  {text[:inner]}  ║")
        text = "    " + text[inner:]
    pad = max(0, inner - len(text))
    print(f"║  {text}{' ' * pad}  ║")

def _divider():
    _row("─" * (W - 6))


# ═════════════════════════════════════════════════════════════════════════════
# NEO4J HANDLER
# ═════════════════════════════════════════════════════════════════════════════

class Neo4jHandler:
    """
    Connects to Neo4j, pushes triplets as MERGE statements,
    then immediately queries and prints the full graph — no browser needed.
    """

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        if not NEO4J_AVAILABLE:
            raise ImportError("neo4j package not installed. Run: pip install neo4j")
        self.driver   = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.driver.verify_connectivity()
        print(f"[Neo4j] ✓ Connected  →  {uri}  (db: {database})")

    def close(self):
        self.driver.close()

    # ── constraints ──────────────────────────────────────────────────────────

    def _ensure_constraints(self, labels: Set[str]):
        with self.driver.session(database=self.database) as s:
            for lbl in sorted(labels):
                try:
                    s.run(
                        f"CREATE CONSTRAINT IF NOT EXISTS "
                        f"FOR (n:{lbl}) REQUIRE n.name IS UNIQUE"
                    )
                except Exception:
                    pass  # older Neo4j versions — safe to ignore

    # ── push ─────────────────────────────────────────────────────────────────

    def push(self, triplets: List[Triplet]) -> Dict:
        """MERGE all triplets into Neo4j. Returns push statistics."""
        stats: Dict = {"pushed": 0, "errors": 0}

        labels: Set[str] = set()
        for t in triplets:
            labels.add(_safe_label(t.subject.label))
            labels.add(_safe_label(t.obj.label))
        self._ensure_constraints(labels)

        seen: Set[Tuple] = set()
        with self.driver.session(database=self.database) as session:
            for t in triplets:
                key = (t.subject.canonical, t.relation, t.obj.canonical)
                if key in seen:
                    continue
                seen.add(key)

                s_lbl = _safe_label(t.subject.label)
                o_lbl = _safe_label(t.obj.label)

                cypher = (
                    f"MERGE (s:{s_lbl} {{name: $sn}}) "
                    f"MERGE (o:{o_lbl} {{name: $on}}) "
                    f"MERGE (s)-[r:{t.relation}]->(o) "
                    f"  SET r.confidence = $conf, r.source = $src"
                )
                try:
                    session.run(cypher, {
                        "sn":   t.subject.text,
                        "on":   t.obj.text,
                        "conf": round(t.confidence, 3),
                        "src":  t.source_sentence[:200],
                    })
                    stats["pushed"] += 1
                    print(f"  ✓  {t.subject.text}  ─[{t.relation}]─▶  {t.obj.text}")
                except Exception as exc:
                    print(f"  ✗  {t}  →  {exc}")
                    stats["errors"] += 1

        return stats

    # ── auto-query and display ────────────────────────────────────────────────

    def query_and_display(self):
        """
        Runs a series of MATCH queries against the live database
        and prints the full graph directly in the terminal.
        No Neo4j Browser or manual steps required.
        """
        with self.driver.session(database=self.database) as session:

            # ── 1. Graph overview ────────────────────────────────────────────
            nc = session.run("MATCH (n)    RETURN count(n) AS c").single()["c"]
            rc = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

            _box("GRAPH OVERVIEW")
            _row(f"Nodes         : {nc}")
            _row(f"Relationships : {rc}")
            _box_end()

            # ── 2. Node labels ───────────────────────────────────────────────
            label_rows = session.run(
                "MATCH (n) UNWIND labels(n) AS lbl "
                "RETURN lbl, count(*) AS cnt ORDER BY cnt DESC"
            ).data()

            _box("NODE LABELS")
            _row(f"  {'LABEL':<20}  COUNT   DISTRIBUTION")
            _divider()
            for r in label_rows:
                bar = "▓" * min(r["cnt"], 25)
                _row(f"  {r['lbl']:<20}  {r['cnt']:<6}  {bar}")
            _box_end()

            # ── 3. Relationship types ────────────────────────────────────────
            rel_rows = session.run(
                "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS cnt "
                "ORDER BY cnt DESC"
            ).data()

            _box("RELATIONSHIP TYPES")
            _row(f"  {'TYPE':<32}  COUNT   DIST")
            _divider()
            for r in rel_rows:
                bar = "▓" * min(r["cnt"], 20)
                _row(f"  {r['rel']:<32}  {r['cnt']:<6}  {bar}")
            _box_end()

            # ── 4. Full edge list ────────────────────────────────────────────
            edges = session.run(
                "MATCH (s)-[r]->(o) "
                "RETURN s.name          AS subject, "
                "       labels(s)[0]    AS s_lbl, "
                "       type(r)         AS relation, "
                "       r.confidence    AS conf, "
                "       o.name          AS object, "
                "       labels(o)[0]    AS o_lbl "
                "ORDER BY subject, relation"
            ).data()

            _box(f"COMPLETE EDGE LIST  ({len(edges)} edges)")
            _row(f"  {'SUBJECT (label)':<26} {'RELATION':<28} {'OBJECT (label)':<26} CONF")
            _divider()
            for e in edges:
                subj   = f"{e['subject']} ({e['s_lbl']})"
                obj    = f"{e['object']} ({e['o_lbl']})"
                conf   = f"{e['conf']:.0%}" if e["conf"] is not None else "n/a"
                arrow  = f"─[{e['relation']}]─▶"
                _row(f"  {subj:<26} {arrow:<30} {obj:<26} {conf}")
            _box_end()

            # ── 5. Per-node neighbourhood ────────────────────────────────────
            nodes = session.run(
                "MATCH (n) RETURN n.name AS name, labels(n)[0] AS lbl "
                "ORDER BY lbl, name"
            ).data()

            _box(f"PER-NODE NEIGHBOURHOOD  ({len(nodes)} nodes)")
            for node in nodes:
                name, lbl = node["name"], node["lbl"]

                out_edges = session.run(
                    "MATCH (n {name:$nm})-[r]->(o) "
                    "RETURN type(r) AS rel, o.name AS tgt, labels(o)[0] AS t_lbl "
                    "ORDER BY rel",
                    {"nm": name}
                ).data()

                in_edges = session.run(
                    "MATCH (o)-[r]->(n {name:$nm}) "
                    "RETURN type(r) AS rel, o.name AS src, labels(o)[0] AS s_lbl "
                    "ORDER BY rel",
                    {"nm": name}
                ).data()

                _row(f"◉  {name}  [{lbl}]")
                for e in out_edges:
                    _row(f"     ──[{e['rel']}]──▶  {e['tgt']}  [{e['t_lbl']}]")
                for e in in_edges:
                    _row(f"     ◀──[{e['rel']}]──  {e['src']}  [{e['s_lbl']}]")
                if not out_edges and not in_edges:
                    _row("     (isolated node — no relationships)")
                _row()
            _box_end()


# ═════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ═════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NLP → Triplets → Neo4j  (fully automatic)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input", "-i", metavar="FILE",
                   help="Plain-text input file. Omit to use built-in demo text.")
    p.add_argument("--uri",       default="bolt://localhost:7687",
                   help="Neo4j Bolt URI           (default: bolt://localhost:7687)")
    p.add_argument("--user",      default="neo4j",
                   help="Neo4j username            (default: neo4j)")
    p.add_argument("--password",  default="password",
                   help="Neo4j password            (default: password)")
    p.add_argument("--database",  default="neo4j",
                   help="Neo4j database name       (default: neo4j)")
    p.add_argument("--min-confidence", type=float, default=0.15, metavar="FLOAT",
                   help="Min confidence threshold  (default: 0.15)")
    p.add_argument("--num-beams",     type=int, default=8,
                   help="REBEL beam width          (default: 8)")
    p.add_argument("--num-sequences", type=int, default=4,
                   help="REBEL return sequences    (default: 4)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                   help="Torch device              (default: cpu)")
    p.add_argument("--save-cypher", metavar="FILE",
                   help="Also save Cypher to a .cypher file")
    p.add_argument("--dry-run", action="store_true",
                   help="Print Cypher only — skip Neo4j entirely")
    p.add_argument("--clear-db", action="store_true",
                   help="⚠  DELETE all nodes/edges in the DB before pushing")
    return p


# ═════════════════════════════════════════════════════════════════════════════
# BUILT-IN DEMO TEXT  (edit or replace with --input)
# ═════════════════════════════════════════════════════════════════════════════

DEMO_TEXT = """
Srishwan is a student at IIT Kharagpur.Aaron Jason Baptist studies in the AI Department at IIT Kharagpur.Rahul is enrolled in Computer Science at IIT Kharagpur.Albert Einstein was born in Ulm, Germany.
Elon Musk is the founder of SpaceX, which is headquartered in California.Christopher Nolan directed The Dark Knight, which won the Academy Award.
Marie Curie was educated at the University of Paris and was born in Warsaw.
Virat Kohli is an Indian cricketer who plays for Royal Challengers Bangalore and is married to Anushka Sharma.
"""


# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    args = build_parser().parse_args()

    # ── 1. Input ─────────────────────────────────────────────────────────────
    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
        print(f"[Input] {args.input}  ({len(text)} chars)")
    else:
        text = DEMO_TEXT.strip()
        print("[Input] Using built-in demo text  (pass --input <file> for your own)")

    # ── 2. NLP extraction ────────────────────────────────────────────────────
    print("\n[Pipeline] Loading models …")
    pipeline = NLPPipeline(
        num_beams      = args.num_beams,
        num_sequences  = args.num_sequences,
        device         = args.device,
        min_confidence = args.min_confidence,
    )
    print("[Pipeline] Extracting triplets …\n")
    triplets = pipeline.process(text, verbose=True)

    if not triplets:
        print("[Pipeline] No triplets extracted — try different text or lower --min-confidence.")
        sys.exit(0)

    # ── 3. Print extracted triplets ──────────────────────────────────────────
    _box(f"EXTRACTED TRIPLETS  ({len(triplets)} unique)")
    for t in triplets:
        bar = "█" * int(t.confidence * 10) + "░" * (10 - int(t.confidence * 10))
        _row(f"[{bar}] {t.confidence:.0%}  {t}")
    _box_end()

    # ── 4. Generate and display Cypher ───────────────────────────────────────
    cypher_block = triplets_to_cypher(
        triplets,
        include_confidence = True,
        min_confidence     = args.min_confidence,
    )
    print("\n─── GENERATED CYPHER " + "─" * 50)
    print(cypher_block)
    print("─" * 71 + "\n")

    if args.save_cypher:
        Path(args.save_cypher).write_text(cypher_block, encoding="utf-8")
        print(f"[Cypher] Saved → {args.save_cypher}\n")

    # ── 5. Dry-run exit ──────────────────────────────────────────────────────
    if args.dry_run:
        print("[Dry-run] Neo4j step skipped.")
        return

    # ── 6. Neo4j driver check ────────────────────────────────────────────────
    if not NEO4J_AVAILABLE:
        sys.exit("[ERROR] neo4j package missing.\n  Run:  pip install neo4j")

    # ── 7. Connect ───────────────────────────────────────────────────────────
    print(f"[Neo4j] Connecting to {args.uri} …")
    try:
        handler = Neo4jHandler(args.uri, args.user, args.password, args.database)
    except neo4j_exc.ServiceUnavailable:
        sys.exit(
            f"\n[ERROR] Neo4j not reachable at {args.uri}\n"
            f"  • Local:   neo4j start   OR   docker run --rm -p7687:7687 -p7474:7474 "
            f"-e NEO4J_AUTH=neo4j/password neo4j\n"
            f"  • AuraDB:  --uri neo4j+s://<id>.databases.neo4j.io\n"
            f"  • Use --dry-run to skip Neo4j."
        )
    except neo4j_exc.AuthError:
        sys.exit(
            "\n[ERROR] Authentication failed.\n"
            "  • Use --password <your-password>\n"
            "  • Default first-login credentials are neo4j / neo4j."
        )

    # ── 8. Optional DB wipe ──────────────────────────────────────────────────
    if args.clear_db:
        print("[Neo4j] ⚠  Clearing all existing nodes and relationships …")
        with handler.driver.session(database=args.database) as s:
            s.run("MATCH (n) DETACH DELETE n")
        print("[Neo4j] ✓  Database cleared.\n")

    # ── 9. Push all triplets ─────────────────────────────────────────────────
    print(f"[Neo4j] Pushing {len(triplets)} triplets …\n")
    stats = handler.push(triplets)
    print(f"\n[Neo4j] ✓  Done — pushed: {stats['pushed']}  |  errors: {stats['errors']}")

    # ── 10. Automatically query and display the graph ─────────────────────────
    print("\n[Neo4j] Fetching graph from database …")
    handler.query_and_display()

    handler.close()
    print("\n[Done] ✓  Graph is live in Neo4j.\n")


if __name__ == "__main__":
    main()