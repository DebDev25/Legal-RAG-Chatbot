# Local AI-Powered Legal Case Law RAG

A fully private, offline Retrieval-Augmented Generation (RAG) system for reasoning over Indian Supreme Court case law, the Indian Penal Code (IPC), and the Code of Criminal Procedure (CrPC). Built for law students and practitioners who require strict data privacy — no query or document ever leaves the machine.

**Stack:** Python · LangChain · ChromaDB · Ollama (Qwen2.5-7B) · HuggingFace · RAGAS · Streamlit

---

## Key Features

- **100% local** — generation (Ollama), embeddings, retrieval, and re-ranking all run on-device. No cloud API calls.
- **Hybrid retrieval** — BM25 keyword search + dense vector search, merged with Reciprocal Rank Fusion, plus exact section-number lookup.
- **Cross-encoder re-ranking** with guaranteed source-type diversity (statute + procedure + case law).
- **Relevance gate** — short-circuits to "I don't know" when nothing in the corpus is close enough, instead of hallucinating.
- **Structured output** — every answer is a validated `LegalResponse` (answer, reasoning, punishment & precedent, confidence, found-in-context).
- **Token streaming** in the UI and a one-time pipeline load for responsive interaction.

---

## Architecture

The pipeline has two phases: ingestion (run once) and query (run per user question).

### Ingestion Pipeline (`src/ingest.py`)

1. **PDF extraction** — PyMuPDF extracts text block-by-block from the corpus (1,001+ Supreme Court PDFs, the IPC, and the CrPC), preserving multi-column reading order.
2. **Chunking** — `RecursiveCharacterTextSplitter` splits each page into ~1200-token chunks with 200-token overlap, keeping full statutory clauses and reasoning passages intact.
3. **Section tagging** — a regex extracts every IPC/CrPC section number referenced in a chunk (e.g. `"302|34"`) into metadata for exact-section lookup at query time.
4. **Embedding** — `all-MiniLM-L6-v2` (384-dimensional) embeds each chunk into a local ChromaDB instance using **cosine distance**.
5. **Metadata tagging** — every chunk carries `source_type` (`penal_code`, `crpc`, or `case_law`), source filename, page number, and the section list.

### Query Pipeline (`src/retrieval.py`)

```
User Query (natural language)
        │
        ▼
 Query Rewriting ──── Ollama qwen2.5:7b rewrites colloquial language into legal
        │              keywords. Two guards run after: STRIP cross-query terms
        │              carried from the previous question, and INJECT dropped
        │              content words so the query stays anchored to the topic.
        ▼
 Cosine Similarity Gate ── short-circuits to "I don't know" if the best chunk
        │                   distance > 0.6 (cosine similarity < 0.4)
        ▼
 Intent + Section Detection ── keyword scan → penal_code / case_law / mixed;
        │                       regex pulls any explicit section numbers
        ▼
 Hybrid Retrieval (k=25 per path)
  ├── BM25 keyword search   ──┐   (statutory-only index on penal_code intent)
  ├── ChromaDB vector search ─┤──► Reciprocal Rank Fusion
  └── Exact section lookup   ──┘   (weights 0.4 / 0.6 / 1.0)
        │
        ▼
 Cross-Encoder Re-ranking (ms-marco-MiniLM-L-6-v2)
        │   + intent bonus (+0.5 primary type, +0.25 CrPC on penal queries)
        │   + section-exact bonus (+1.0)
        │   → top 6 chunks (2 penal_code + 2 crpc + 2 case_law, diversity guaranteed)
        ▼
 Ollama qwen2.5:7b ── streamed JSON generation, parsed into LegalResponse;
        │              safety-refusal sentences sanitized out
        ▼
 LegalResponse { answer · reasoning · possible_ruling · confidence · found_in_context }
        │
        ▼
 Streamlit UI (confidence badge · answer · reasoning · punishment & precedent · reference chips)
```

---

## Project Structure

```
Legal RAGchatbot/
├── src/
│   ├── app.py          # Streamlit UI (streaming generation, safety sanitization)
│   ├── ingest.py       # PDF → ChromaDB ingestion pipeline
│   └── retrieval.py    # Query pipeline (rewrite → gate → retrieve → rerank → generate)
├── benchmarks/
│   ├── benchmark.py    # End-to-end latency and confidence benchmark
│   ├── ragas_eval.py   # RAGAS correctness evaluation (answer_relevancy, local judge)
│   ├── generate_plots.py
│   └── results/        # Generated CSVs and plots (git-ignored)
├── tests/
│   └── test_pipeline.py    # Integration smoke tests
├── storage/
│   └── chroma_db/      # Vector database (git-ignored)
├── data/               # PDF corpus (git-ignored)
│   ├── indian_supreme_court_pdfs/
│   ├── penal_code/
│   └── code_of_criminal_procedure/
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- The `qwen2.5:7b` model pulled: `ollama pull qwen2.5:7b`

### Install dependencies

```bash
pip install -r requirements.txt
```

### Ingest your PDF corpus

Place PDFs in the corresponding folders under `data/`, then:

```bash
python src/ingest.py
```

This builds the ChromaDB vector store under `storage/chroma_db/` with cosine distance and section-number metadata.

### Run the app

```bash
streamlit run src/app.py
```

---

## Evaluation

### Latency benchmark

Runs all questions through the full pipeline and records per-query timing, confidence, and top retrieved source:

```bash
python benchmarks/benchmark.py        # → benchmarks/results/benchmark_metrics.csv
```

### RAGAS correctness evaluation

Measures **answer relevancy** entirely locally (embedding-based, no LLM verdict parsing). `faithfulness`, `context_precision`, and `context_recall` are excluded: all three require the judge LLM to emit binary `Verdict: 1/0` per claim/sentence, a format that models under 13B do not follow reliably (ragas hits parse errors on every row).

```bash
python benchmarks/ragas_eval.py       # → benchmarks/results/ragas_scores.csv
```

### Visualise results

```bash
python benchmarks/generate_plots.py   # → benchmarks/results/rag_performance_metrics.png
```

### Tests

```bash
pytest tests/test_pipeline.py -s      # integration smoke tests (needs Ollama + ChromaDB)
```
