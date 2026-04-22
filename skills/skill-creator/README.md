python skill.py validate  path/to/my-skill
python skill.py loop      --skill-path path/to/my-skill --eval-set evals.json --model claude-opus-4-5
python skill.py review    workspaces/my-workspace/

Yes — no special privileges needed. Here's why:

MCP's Streamable HTTP transport (the current spec) is just JSON-RPC 2.0 over POST /mcp on a local HTTP server, optionally with SSE streaming. Your bridge already is a local HTTP server on 127.0.0.1:7777. Adding MCP support means adding a /mcp route handler to the existing extension.js — nothing more.

Any MCP client can then point at it:

// .vscode/mcp.json
{
  "servers": {
    "copilot-bridge": {
      "type": "http",
      "url": "http://127.0.0.1:7777/mcp"
    }
  }
}

hat you'd get: other tools (Claude Desktop, Cursor, custom scripts) calling into Copilot through the bridge via standard MCP protocol — no direct API keys, no outbound MCP calls from your network, everything stays on loopback through VS Code's auth.

Caveats:

The bridge only exposes chat/completions — no tool-use or resource endpoints unless you add them
MCP clients need to support HTTP transport (stdio-only clients won't work without a wrapper)
VS Code's own Copilot Chat won't use it (it has its own LM access), but skill.py and any external MCP client would
Want me to add the /mcp JSON-RPC endpoint to extension.js?

Architecture Advice
The core problem: VS Code extensions run in a sandboxed Node.js host — no DOM, but they can spawn child processes and host webview panels (full HTML/CSS/JS iframes). This is exactly the right environment for what you want.

Don't build a second extension. You'll end up maintaining two extension lifecycles, two activation events, two package.json manifests. Add a PipelinePanel command to the existing skill-bridge extension. One install, everything shares the same process.

State machine — where should it live?

Layer	What lives there	Why
Extension host (Node.js)	The FSM, child_process.spawn, file I/O, stdout capture	Has OS access, persists across webview reloads
Webview (HTML/JS)	Rendering, button clicks, terminal display	Sandboxed, talks to host via postMessage
Never put the FSM in the webview — if the user closes and reopens the panel mid-run, the webview is recreated but the host keeps running. The webview is just a dumb display.

State machine design — keep it hand-rolled, no XState:

// A "step" in the pipeline
{ id, label, command, args, workingDir, 
  onSuccess: stepId | null,   // → next step on exit 0
  onFailure: stepId | null,   // → branch on non-zero
  captureOutput: bool          // feed stdout into next step's env
}

Output piping: capture stdout from step N, inject it as $PREV_OUTPUT env var into step N+1. Simple, shell-composable, no magic.

Pipeline definition — a JSON file in the workspace:

{
  "name": "Skill Eval Loop",
  "entry": "validate",
  "steps": {
    "validate": { "command": "python", "args": ["skill.py", "validate", "."], "onSuccess": "eval", "onFailure": null },
    "eval":     { "command": "python", "args": ["skill.py", "eval", "--skill-path", "."], "onSuccess": "loop", "captureOutput": true },
    "loop":     { "command": "python", "args": ["skill.py", "loop", "--skill-path", "."], "onSuccess": "report" },
    "report":   { "command": "python", "args": ["skill.py", "report", "-"], "onSuccess": null }
  }
}

UI approach — a VS Code Webview Panel:

Left sidebar: pipeline graph (SVG, nodes highlight as they run)
Right panel: live streaming terminal output per step (ANSI rendered)
Bottom: run/pause/retry controls
Use a CSS custom properties theme that reads VS Code's color variables (--vscode-editor-background etc.) so it looks native, not a foreign app embedded in your editor
Honest caveats:

child_process in a VS Code extension host works well for short-lived scripts; long-running processes need explicit cleanup in deactivate()
ANSI color rendering in the webview requires a small library (xterm.js ~200KB) or a simple regex stripper if you want zero deps
The webview's postMessage throughput is fine for log lines but don't push megabytes of stdout through it — buffer and tail instead
This will work well for your use case. The skill.py scripts are CLI tools with structured JSON output — ideal for piping. The main risk is someone's pipeline hanging on a script that waits for stdin; add a configurable timeout per step.

Want me to build it? I'd add a PipelinePanel to the existing extension.js, a pipeline.json schema, and the webview HTML with a live terminal + step graph — all in one shot.

Good question to think through carefully before touching code. Here's the honest principal engineer analysis:

Local Embeddings — the real options
Approach	Quality	Deps	Verdict
BM25-lite (current)	★★☆	none	Good baseline, no IDF weighting
TF-IDF + cosine	★★★	none (pure Python)	Best pure-stdlib upgrade
Static word vectors (GloVe/Word2Vec)	★★★☆	numpy + ~50MB binary file	Good offline after one-time download
ONNX sentence-transformers	★★★★	onnxruntime pip dep + ~80MB model	Best quality still-offline; needs one-time download
sentence-transformers (PyTorch)	★★★★★	torch (~2GB)	Overkill
Recommendation: TF-IDF + cosine similarity in pure Python. Proper IDF weighting kills false positives from common terms (e.g. "skill", "the", "use") that BM25-lite only partially handles. ~60 lines, zero deps, measurable quality improvement for technical docs.

If you can allow one pip dep at install time: ONNX + all-MiniLM-L6-v2 (80MB, runs fully offline after download, real semantic similarity). Worth doing if the corpus is large or query phrasing varies a lot from doc phrasing.

Local Vector Search — no DB needed
For your corpus size (skill docs = hundreds to low thousands of chunks), flat cosine scan is the right answer. An ANN index (HNSW, LSH) only pays off above ~100k vectors. Flat scan over 5k TF-IDF vectors is microseconds in Python.

The only thing worth adding is a persistent cache: serialize the TF-IDF matrix + vocabulary to a .pkl or .json file keyed by a hash of the source files' mtimes. Avoids re-indexing on every query.

Graph RAG — the useful 20%
Microsoft's full GraphRAG is complex and expensive (many LM calls for hierarchical summarization). The genuinely useful subset with zero external calls:

Document similarity graph (the most practical):


Index time: build sparse adjacency — for every doc pair with cosine(d_i, d_j) > threshold, store an edge
Query time: find top-k by TF-IDF/cosine → expand: include 1-hop graph neighbors of each top hit

This catches "related but different phrasing" documents that pure retrieval misses. Imagine querying "when should I invoke this" — TF-IDF finds the "trigger detection" doc, graph expansion pulls in the "description writing guide" that doesn't share those terms but is heavily linked to it.

What you get for ~80 lines of pure Python:

TF-IDF + cosine retrieval (replaces BM25-lite)
Lazy-built, mtime-cached index
Sparse doc-similarity graph for 1-hop expansion
Still zero external deps, zero network
What I'd skip:

Concept/entity extraction graphs — needs an NLP library to do well, error-prone with regex
LSH/HNSW — premature optimization for this corpus size
Full hierarchical GraphRAG — too many LM calls, too much complexity for the return
Concrete plan
Replace _retrieve_context() in skill.py with three layered functions

_build_index(context_dir)   # TF-IDF matrix, vocabulary, doc list — cached to .skill-index/
_build_graph(index)         # sparse adjacency dict, threshold=0.25 — cached alongside index  
_retrieve_context(query, context_dir, top_k, graph_hops)  # query → retrieval → graph expansionDuckDB is the right call. Polars adds nothing here that DuckDB doesn't already do better.

Here's why DuckDB is a perfect fit:

pip install duckdb — ~20MB wheel, zero model downloads, zero network after install
Has list_cosine_similarity(a, b) built-in since v0.10 — that's your vector search engine
Single .duckdb file = persistent cache with zero extra infrastructure
Graph edges are just a table — 1-hop expansion is a JOIN
TF-IDF vectors stored as FLOAT[] columns and queried with a single SQL ORDER BY score DESC LIMIT k
Polars — useful for fast dataframe ETL but it has no vector ops and no persistence. With DuckDB you get both. Don't add two deps when one does the job.

-- built once, cached, invalidated by source file mtime hash
CREATE TABLE meta  (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE vocab (term TEXT PRIMARY KEY, idf FLOAT, idx INTEGER);
CREATE TABLE docs  (
    id      TEXT PRIMARY KEY,
    source  TEXT,           -- source file path
    chunk   INTEGER,        -- chunk index within file
    content TEXT,
    vec     FLOAT[]         -- dense TF-IDF vector (len = |vocab|)
);
CREATE TABLE graph (
    src         TEXT,
    dst         TEXT,
    similarity  FLOAT,
    PRIMARY KEY (src, dst)
);


# 1. Tokenize query → sparse TF dict
# 2. Build dense query vector against vocab table (one SELECT)
# 3. Retrieval:
SELECT id, source, content,
       list_cosine_similarity(vec, $query_vec) AS score
FROM   docs
ORDER  BY score DESC
LIMIT  $top_k;

# 4. Graph expansion (1-hop):
SELECT DISTINCT d.content
FROM   graph g
JOIN   docs d ON d.id = g.dst
WHERE  g.src IN ($top_k_ids)
  AND  g.similarity > 0.25
ORDER  BY g.similarity DESC
LIMIT  $expand_k;

# On every retrieve call:
current_hash = hash({f: f.stat().st_mtime for f in context_dir.rglob("*.md")})
stored_hash  = db.execute("SELECT value FROM meta WHERE key='source_hash'").fetchone()
if stored_hash != current_hash:
    rebuild_index(db, context_dir)

    Graph only needs rebuilding when docs change — it's an O(n²) pairwise cosine but for hundreds of chunks that's milliseconds in DuckDB.

Upgrade path if you later get ONNX access
The schema doesn't change at all — you'd just replace the TF-IDF vec builder with an ONNX encoder outputting the same FLOAT[]. Everything downstream (SQL retrieval, graph, cache) stays identical. That's the real architectural win of this approach.

Want me to implement this now — replace _retrieve_context() with _build_index() / _build_graph() / _retrieve_context() backed by DuckDB?

Now add hashlib to the imports — check what's currently imported:

Read skill.py, lines 55 to 75

 all there and complete. The DuckDB RAG implementation was already written in the previous session. The full implementation is in skill.py:

_tokenize() — stop-word filtered tokenizer
_chunk_file() — sliding window chunker with overlap
_source_hash() — mtime-based cache invalidation
_build_rag_index() — TF-IDF vectors + pairwise similarity graph, persisted to .skill-index/rag.duckdb
_get_rag_db() — lazy open + stale-check, silent rebuild on change
_query_vec_from_db() — projects query onto stored vocab
_retrieve_context() — DuckDB cosine retrieval + 1-hop graph expansion, degrades gracefully if duckdb not installed
The only thing needed to use it is pip install duckdb, then pass --context-dir path/to/docs to eval, improve, or loop.

hat maps directly to what we already have
Hindsight component	Our equivalent	Gap
Semantic search	TF-IDF cosine (DuckDB)	Not semantic — no synonym matching
BM25 keyword search	DuckDB FTS extension (one line)	None — this is free
Document graph	Cosine similarity graph (already built)	None
Temporal search	Add created_at TIMESTAMP to docs table	Trivial
RRF fusion	20 lines Python	None
Fact extraction	LM bridge call with structured prompt	None
Observation synthesis	LM bridge + background consolidation	None
Cross-encoder reranking	❌ needs a model	Blocked
Real semantic embeddings	❌ TF-IDF is not semantic	Blocked without model
The single real gap: TF-IDF won't match automobile → car, utilise → use, or conceptual similarity without shared terms. Everything else is implementable today.

The one pip install that fixes the gap

pip install sentence-transformers   # ~80MB, downloads model once, runs fully offline after
all-MiniLM-L6-v2 — 80MB, 384-dim real semantic embeddings, CPU inference in ~5ms per chunk. The schema is identical to what we already have (vec FLOAT[]). You'd swap one function in _build_rag_index. Everything downstream — DuckDB cosine query, graph, RRF — unchanged.

If that model download is blocked by your network, the fallback is pip install onnxruntime + ship the .onnx file through your internal artifact repo (no internet at inference time either way).

Recommended architecture: memory.py as a standalone module
Don't bloat skill.py further. A clean separation:


CREATE TABLE facts (
    id          TEXT PRIMARY KEY,
    content     TEXT,
    fact_type   TEXT,           -- 'world' | 'experience'
    source      TEXT,           -- originating text/conversation id
    created_at  TIMESTAMP,
    vec         FLOAT[]         -- TF-IDF or sentence-transformer embedding
);

CREATE TABLE entities (
    id          TEXT PRIMARY KEY,
    name        TEXT,
    type        TEXT,           -- person/org/concept/place
    last_seen   TIMESTAMP
);

CREATE TABLE fact_entities (
    fact_id     TEXT,
    entity_id   TEXT,
    PRIMARY KEY (fact_id, entity_id)
);

CREATE TABLE observations (
    id          TEXT PRIMARY KEY,
    content     TEXT,
    evidence    TEXT[],         -- list of fact_ids that support this
    created_at  TIMESTAMP,
    updated_at  TIMESTAMP
);

CREATE TABLE fact_graph (
    src         TEXT,
    dst         TEXT,
    link_type   TEXT,           -- 'semantic'|'temporal'|'entity'|'causal'
    weight      FLOAT,
    PRIMARY KEY (src, dst)
);


Public API surface (three functions mirroring Hindsight's three operations):

def retain(text: str, source: str, db_path: Path, bridge_port: int) -> int:
    """Extract facts + entities from text via LM, store in DuckDB, update graph."""

def recall(query: str, db_path: Path, top_k: int = 8) -> str:
    """BM25 + TF-IDF cosine + graph traversal + temporal → RRF → formatted context."""

def reflect(query: str, db_path: Path, bridge_port: int) -> str:
    """recall() → LM synthesis → answer string."""


retain makes one LM call with a structured extraction prompt returning JSON facts + entities. That's the only per-ingestion cost. Everything else is local DuckDB.

What you get at the end
Fast: DuckDB cosine + FTS queries are sub-millisecond on thousands of facts
Cheap: One LM call per retain, zero calls per recall, one per reflect
Safe: Everything in a local .duckdb file, no external service, bridge stays on loopback
Upgradeable: Drop in sentence_transformers later and swap one function — the rest is unchanged
Want me to implement memory.py now? I'll build retain/recall/reflect with DuckDB FTS (proper BM25), TF-IDF cosine, temporal decay, graph traversal, and RRF — with a clean abstraction so you can swap in real embeddings by changing one function.

