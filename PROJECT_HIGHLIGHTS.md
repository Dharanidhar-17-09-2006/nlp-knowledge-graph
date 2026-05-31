# CDC Technical Highlights

## Resume Pitch

Built a hybrid NLP-to-Knowledge-Graph system using REBEL, Groq Llama, spaCy, and Neo4j to extract structured triples from text, PDF, and Wikipedia inputs with lexical grounding, confidence scoring, conflict detection, graph visualization, and benchmark evaluation.

## Strong Technical Talking Points

- Hybrid extraction: combines symbolic/model-based REBEL triples with Groq LLM triples.
- Lexical grounding: filters triples whose subject/object evidence is missing from source text.
- Provenance: stores extractor source, confidence, lexical score, and conflict flags per relation.
- Knowledge graph persistence: maps entities and relations into Neo4j with labels and typed edges.
- Safe graph querying: validates LLM-generated Cypher and only allows read-only MATCH/RETURN queries.
- Evaluation: compares REBEL, Groq, and tuned Hybrid using precision, recall, Exact F1, and Partial F1.
- Frontend: Streamlit interface for text, PDF, Wikipedia ingestion, graph rendering, JSON export, and NL querying.

## Suggested Resume Bullet

Built a hybrid NLP-to-Knowledge-Graph pipeline using REBEL, Groq Llama, spaCy, and Neo4j, extracting grounded triples from text/PDF/Wikipedia sources with confidence scoring, provenance, conflict detection, graph visualization, and benchmark evaluation achieving 0.88 recall and 0.93 Partial F1 in the tuned hybrid setting.

## Interview Discussion Flow

1. Explain the pipeline: input loader -> preprocessing -> extractors -> fusion -> validation -> Neo4j -> query UI.
2. Explain why Hybrid is useful: REBEL gives structured relation extraction; Groq improves recall and handles natural language variation.
3. Explain safety: LLM output is not trusted; generated Cypher passes a read-only validator before execution.
4. Explain evaluation: Exact F1 is strict, Partial F1 gives credit for near-correct entity/relation matches.
5. Explain scaling: batch extraction, caching, async LLM calls, graph indexes, and read-only query users.
