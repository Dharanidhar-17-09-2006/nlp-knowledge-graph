"""
Streamlit frontend for the hybrid NLP to Neo4j knowledge graph system.
"""
from __future__ import annotations
import os
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
import json

import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st
from dotenv import load_dotenv

from cypher_guard import validate_readonly_cypher

load_dotenv()

st.set_page_config(page_title="NLP Knowledge Graph Builder", page_icon="KG", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; }
    .metric-row [data-testid="stMetricValue"] { font-size: 1.6rem; }
    .kg-header {
        border-bottom: 1px solid #e6e8ef;
        padding-bottom: 0.9rem;
        margin-bottom: 1rem;
    }
    .small-muted { color: #5f6b7a; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="kg-header">
      <h1>Modular NLP Knowledge Graph (REBEL + Groq LLM)</h1>
      <p class="small-muted">
      Extract grounded subject–relation–object triples from text, PDFs, and Wikipedia. Fuse outputs from REBEL and Groq (or use either model independently via a toggle), detect and resolve conflicts, and store the resulting knowledge graph in Neo4j.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)


def text_contains_entity(entity_text: str, source_text: str) -> bool:
    entity_lower = entity_text.lower().strip()
    source_lower = source_text.lower()
    if entity_lower in source_lower:
        return True
    words = [w for w in entity_lower.split() if len(w) > 3]
    return any(w in source_lower for w in words)


def safe_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", label or "Entity") or "Entity"


def safe_relation(relation: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", relation.upper()) or "RELATED_TO"


def is_wikipedia_url(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    return parsed.scheme in {"http", "https"} and (host == "wikipedia.org" or host.endswith(".wikipedia.org"))


@st.cache_resource
def load_pipeline(min_confidence: float):
    from NLP_pipeline import NLPPipeline

    return NLPPipeline(min_confidence=min_confidence)


def fetch_wikipedia_text(url: str) -> str:
    if not is_wikipedia_url(url):
        raise ValueError("Only wikipedia.org URLs are accepted.")

    import requests
    from bs4 import BeautifulSoup

    resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    paragraphs = soup.select("#mw-content-text p")
    return " ".join(p.get_text(" ", strip=True) for p in paragraphs[:15])[:4000]


def extract_rebel_triples(raw_text: str, min_conf: float, enabled: bool) -> list[dict]:
    if not enabled:
        return []

    pipeline = load_pipeline(min_conf)
    rebel_triplets = pipeline.process(raw_text.strip(), verbose=False)
    rebel_dicts = []

    for t in rebel_triplets:
        if not text_contains_entity(t.subject.text, raw_text):
            continue
        if not text_contains_entity(t.obj.text, raw_text):
            continue
        if t.subject.text.lower() == t.obj.text.lower():
            continue
        rebel_dicts.append(
            {
                "subject": t.subject.text,
                "relation": t.relation,
                "object": t.obj.text,
                "confidence": t.confidence,
                "source": "rebel",
                "extractors": ["rebel"],
                "lexical_score": 1.0,
                "subject_label": t.subject.label,
                "object_label": t.obj.label,
            }
        )

    return rebel_dicts


def extract_groq_triples(raw_text: str, groq_key: str) -> list[tuple[str, str, str]]:
    from groq_extractor import GroqExtractor

    pipeline = load_pipeline(0.15)
    nlp = pipeline.labeler.nlp
    sentences = [s.text.strip() for s in nlp(raw_text).sents if len(s.text.strip()) > 5]
    groq = GroqExtractor(groq_key)
    groq_all = []
    for sent in sentences:
        groq_all.extend(groq.extract(sent))
    return groq_all


def build_pyvis_graph(triples: list[dict]) -> str:
    from pyvis.network import Network

    net = Network(height="560px", width="100%", bgcolor="#ffffff", font_color="#222222")
    net.barnes_hut()

    color_map = {
        "Person": "#2563eb",
        "Organization": "#ea580c",
        "Location": "#16a34a",
        "Date": "#9333ea",
        "Entity": "#64748b",
        "CreativeWork": "#db2777",
        "Product": "#0891b2",
    }

    added_nodes = set()
    for triple in triples:
        s_node = triple["subject"]
        o_node = triple["object"]
        s_label = triple.get("subject_label", "Entity")
        o_label = triple.get("object_label", "Entity")
        conf = triple.get("confidence", 1.0)
        conflict = triple.get("conflict", False)
        source = ", ".join(triple.get("extractors", [triple.get("source", "unknown")]))

        if s_node not in added_nodes:
            net.add_node(s_node, label=s_node, color=color_map.get(s_label, "#64748b"), title=s_label, size=22)
            added_nodes.add(s_node)
        if o_node not in added_nodes:
            net.add_node(o_node, label=o_node, color=color_map.get(o_label, "#64748b"), title=o_label, size=22)
            added_nodes.add(o_node)

        net.add_edge(
            s_node,
            o_node,
            label=triple["relation"],
            title=f"confidence: {conf:.0%} | source: {source}",
            color="#dc2626" if conflict else "#475569",
            width=1 + conf * 3,
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
        net.save_graph(f.name)
        return Path(f.name).read_text(encoding="utf-8")


def push_to_neo4j(triples: list[dict], conflicts: list, uri: str, user: str, password: str) -> int:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    pushed = 0
    with driver.session() as session:
        for triple in triples:
            rel = safe_relation(triple["relation"])
            s_lbl = safe_label(triple.get("subject_label", "Entity"))
            o_lbl = safe_label(triple.get("object_label", "Entity"))
            session.run(
                f"MERGE (s:{s_lbl} {{name: $sn}}) "
                f"MERGE (o:{o_lbl} {{name: $on}}) "
                f"MERGE (s)-[r:{rel}]->(o) "
                "SET r.confidence=$conf, r.source=$src, r.extractors=$extractors, "
                "r.lexical_score=$lexical, r.conflict=$conflict",
                {
                    "sn": triple["subject"],
                    "on": triple["object"],
                    "conf": triple.get("confidence", 1.0),
                    "src": triple.get("source", "unknown"),
                    "extractors": triple.get("extractors", [triple.get("source", "unknown")]),
                    "lexical": triple.get("lexical_score", 0.0),
                    "conflict": triple.get("conflict", False),
                },
            )
            pushed += 1

        if conflicts:
            from conflict_detector import push_conflicts_to_neo4j

            push_conflicts_to_neo4j(conflicts, session)
    driver.close()
    return pushed


with st.sidebar:
    st.header("Configuration")
    groq_key = os.getenv("GROQ_API_KEY", "")
    st.subheader("Neo4j")
    neo4j_uri = st.text_input("URI", value=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user = st.text_input("Username", value=os.getenv("NEO4J_USER", "neo4j"))
    neo4j_pass = st.text_input("Password", type="password", value=os.getenv("NEO4J_PASSWORD", ""))
    st.subheader("Pipeline")
    min_conf = st.slider("Minimum REBEL confidence", 0.0, 1.0, 0.15, 0.05)
    use_groq = st.toggle("Use Groq extractor", value=True)
    use_rebel = st.toggle("Use REBEL extractor", value=True)
    detect_conf = st.toggle("Detect conflicts", value=True)
    dry_run = st.toggle("Dry run: skip Neo4j write", value=False)

input_tab, results_tab, query_tab = st.tabs(["Input", "Graph Builder", "Ask Neo4j"])

with input_tab:
    input_mode = st.radio("Input type", ["Paste text", "Upload PDF", "Wikipedia URL"], horizontal=True)
    raw_text = ""

    if input_mode == "Paste text":
        raw_text = st.text_area(
            "Source text",
            height=220,
            placeholder="Srishwan is a student at IIT Kharagpur. Elon Musk founded SpaceX...",
        )
    elif input_mode == "Upload PDF":
        uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded:
            try:
                import pdfplumber

                with pdfplumber.open(uploaded) as pdf:
                    raw_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
                st.success(f"Extracted {len(raw_text)} characters from PDF.")
                st.text_area("Preview", raw_text[:2000], height=180)
            except ImportError:
                st.error("Install pdfplumber to use PDF input.")
    else:
        url = st.text_input("Wikipedia URL", placeholder="https://en.wikipedia.org/wiki/IIT_Kharagpur")
        if url:
            try:
                raw_text = fetch_wikipedia_text(url)
                st.success(f"Fetched {len(raw_text)} characters.")
                st.text_area("Preview", raw_text[:2000], height=180)
            except Exception as e:
                st.error(f"Failed to fetch article: {e}")

with results_tab:
    if st.button("Extract and Build Graph", type="primary", disabled=not raw_text.strip()):
        progress = st.progress(0, text="Starting pipeline...")
        try:
            progress.progress(15, "Running REBEL extractor...")
            rebel_dicts = extract_rebel_triples(raw_text, min_conf, use_rebel)
            fused_triples = rebel_dicts.copy()

            progress.progress(45, "Running Groq extractor and hybrid fusion...")
            if use_groq and groq_key:
                from groq_extractor import fuse_triples

                groq_all = extract_groq_triples(raw_text, groq_key)
                rebel_for_fusion = [
                    (
                        t["subject"],
                        t["relation"],
                        t["object"],
                        t["confidence"],
                        t.get("subject_label", "Entity"),
                        t.get("object_label", "Entity"),
                    )
                    for t in rebel_dicts
                ]
                fused_triples = fuse_triples(rebel_for_fusion, groq_all, raw_text)
            elif use_groq and not groq_key:
                st.warning("Groq is enabled, but no API key was provided. Using REBEL output only.")

            conflicts = []
            if detect_conf:
                progress.progress(65, "Detecting conflicts...")
                from conflict_detector import detect_conflicts

                fused_triples, conflicts = detect_conflicts(fused_triples)

            progress.progress(80, "Rendering graph...")
            st.session_state["last_triples"] = fused_triples
            st.session_state["last_conflicts"] = conflicts

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Triples", len(fused_triples))
            c2.metric("Conflicts", len(conflicts))
            c3.metric("Hybrid triples", sum(1 for t in fused_triples if t.get("source") == "hybrid_corroborated"))
            c4.metric("Avg confidence", f"{(sum(t.get('confidence', 0) for t in fused_triples) / max(len(fused_triples), 1)):.0%}")

            table_rows = [
                {
                    "Subject": t["subject"],
                    "Relation": t["relation"],
                    "Object": t["object"],
                    "Confidence": t.get("confidence", 0),
                    "Source": t.get("source", "unknown"),
                    "Extractors": ", ".join(t.get("extractors", [])),
                    "Conflict": t.get("conflict", False),
                }
                for t in fused_triples
            ]
            st.dataframe(table_rows, use_container_width=True, hide_index=True)

            try:
                import streamlit.components.v1 as components

                components.html(build_pyvis_graph(fused_triples), height=585, scrolling=True)
            except ImportError:
                st.info("Install pyvis to render the interactive graph.")

            st.download_button(
                "Download triples JSON",
                data=json.dumps(fused_triples, indent=2),
                file_name="extracted_triples.json",
                mime="application/json",
            )

            if not dry_run:
                progress.progress(92, "Pushing to Neo4j...")
                pushed = push_to_neo4j(fused_triples, conflicts, neo4j_uri, neo4j_user, neo4j_pass)
                st.success(f"Pushed {pushed} triples to Neo4j.")
            else:
                st.info("Dry run enabled. Neo4j write skipped.")

            progress.progress(100, "Done.")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            import traceback

            st.code(traceback.format_exc())
    elif "last_triples" in st.session_state:
        st.info("Showing the latest extracted triples from this session.")
        st.dataframe(st.session_state["last_triples"], use_container_width=True)
    else:
        st.info("Add input text, then run extraction.")

with query_tab:
    st.write("Ask a natural-language question. The generated Cypher is validated as read-only before execution.")
    query_input = st.text_input("Question", placeholder="Who studied at IIT Kharagpur?")

    if st.button("Generate and Run Query", disabled=not query_input.strip()):
        if not groq_key:
            st.error("Groq API key required for natural-language query generation.")
        else:
            try:
                import requests as req
                from neo4j import GraphDatabase, READ_ACCESS

                try:
                    schema_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
                    with schema_driver.session() as s:
                        labels = [r["label"] for r in s.run("CALL db.labels() YIELD label RETURN label").data()]
                        rel_types = [r["relationshipType"] for r in s.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType").data()]
                        sample_nodes = [r["name"] for r in s.run("MATCH (n) RETURN n.name AS name LIMIT 20").data() if r["name"]]
                        # Getting node counts per label for better context
                        label_counts = [r for r in s.run("CALL db.labels() YIELD label CALL apoc.cypher.run('MATCH (n:' + label + ') RETURN count(n) as count', {}) YIELD value RETURN label, value.count as count").data()]
                    schema_driver.close()
                    schema_str = (
                        f"Node labels: {', '.join(labels)}. "
                        f"Relationship types: {', '.join(rel_types)}. "
                        f"Sample node names (up to 20): {', '.join(sample_nodes)}."
                    )
                except Exception:
                    schema_str = "Nodes: Person, Organization, Location, Entity. Relationships: STUDIED_AT, WORKS_FOR, FOUNDED_BY, BORN_IN, LOCATED_IN."

                prompt = f"""Convert this question to a read-only Neo4j Cypher query.
                this is the schema string of graph database on which query is needed : {schema_str}
                All nodes have a name property.
                Only return one MATCH/OPTIONAL MATCH query with RETURN. Never write, delete, merge, create, call procedures, or use semicolons.
                Only give valid neo4j query of the question, follow all neo4j query rules, any natrual language or invalid query language
                Question: {query_input}"""

                resp = req.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                    json={
                        "model": "llama-3.1-8b-instant",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "max_tokens": 256,
                    },
                    timeout=10,
                    proxies={"http": None, "https": None},
                )
                resp.raise_for_status()
                cypher_query = resp.json()["choices"][0]["message"]["content"].strip()
                validation = validate_readonly_cypher(cypher_query)

                st.code(validation.query, language="cypher")
                if not validation.ok:
                    st.error(f"Blocked query: {validation.reason}")
                else:
                    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))
                    with driver.session(default_access_mode=READ_ACCESS) as session:
                        results = session.run(validation.query).data()
                    driver.close()
                    if results:
                        st.dataframe(results, use_container_width=True)
                    else:
                        st.info("No results found.")
            except Exception as e:
                st.error(f"Query error: {e}")
