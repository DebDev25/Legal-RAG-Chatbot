import re
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = str(Path(__file__).parent.parent / "storage" / "chroma_db")
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
K_CANDIDATES  = 25    # candidates retrieved by each sub-retriever before re-ranking
K_PENAL_FINAL = 2     # guaranteed penal_code slots in final context
K_CRPC_FINAL  = 2     # guaranteed crpc (Code of Criminal Procedure) slots
K_CASE_FINAL  = 2     # guaranteed case_law slots in final context
K_FINAL = K_PENAL_FINAL + K_CRPC_FINAL + K_CASE_FINAL  # 6 total

# Cosine distance threshold (= 1 - cosine_similarity, range 0–1).
# < 0.3 = strong match (cosine_sim > 0.7), 0.3–0.6 = marginal, > 0.6 = no match.
# If the best chunk exceeds this, the query has no relevant law in the DB.
NO_CONTEXT_THRESHOLD = 0.6


# ── Structured Output Schema ──────────────────────────────────────────────────

class LegalResponse(BaseModel):
    """Schema for the model's structured response."""
    answer: str = Field(
        description=(
            "One-sentence verdict: start with 'Yes.' or 'No.' if the legality is explicit, "
            "otherwise a direct statement of the legal position. Never a single word."
        )
    )
    reasoning: str = Field(
        description=(
            "2–4 sentences explaining the legal basis. Cite document name, page, and the "
            "specific section or ruling. If penal_code chunks are present, state the exact "
            "statutory punishment (e.g. 'death or imprisonment for life'). "
            "If case_law chunks are present, name the case and holding."
        )
    )
    possible_ruling: str = Field(
        description=(
            "For criminal queries, populate BOTH sub-parts in order:\n"
            "(a) Statutory punishment — copy verbatim from penal_code chunks: "
            "section number + act + the exact punishment text "
            "(e.g. 'Section 302 IPC: death, or imprisonment for life, and fine').\n"
            "(b) Case law — scan EVERY chunk labelled Type: case_law. "
            "For each one related to the query topic, write one line giving: "
            "the case/document name, source file and page, and what the court decided "
            "(verdict, sentence, or key holding). "
            "Format: '[Name], [file] p.[page]: [holding].' "
            "Example: 'Bachan Singh v. State, bachan_singh.pdf p.12: "
            "Supreme Court upheld death penalty under Section 302 IPC in rarest-of-rare cases.' "
            "If NO case_law chunk relates to this offence after scanning all of them, "
            "write exactly: 'No matching case law in the retrieved context for this offence.' "
            "NEVER leave part (b) blank on a criminal query.\n"
            "Write 'N/A' ONLY if the query is entirely non-criminal."
        )
    )
    confidence: Literal["high", "medium", "low", "none"] = Field(
        description=(
            "high   — context directly and explicitly answers the query. "
            "medium — context partially covers the query. "
            "low    — context is tangentially related. "
            "none   — context contains no relevant information."
        )
    )
    found_in_context: bool = Field(
        description="True if the Document Context contained information relevant to the query."
    )


# ── Query Rewriting ───────────────────────────────────────────────────────────

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a legal search query optimizer for an Indian law database. "
        "Respond in English only — never use any other language.\n\n"
        "Convert the user's question into 5–10 space-separated legal keywords. "
        "Extract keywords ONLY from the CURRENT question — do not include legal terms "
        "from any other topic not mentioned in this question.\n\n"
        "Rules:\n"
        "- Use spaces between words. Never use underscores, hyphens, or camelCase.\n"
        "- Use formal Indian legal terms for concepts that ARE in the question.\n"
        "- Do NOT add any IPC section numbers, act names, or legal citations.\n"
        "- Output ONLY the keywords. No sentence, no punctuation, no explanation.\n\n"
        "Examples:\n"
        "Question: What is the punishment for theft?\n"
        "Keywords: theft punishment dishonest taking movable property\n\n"
        "Question: Can someone enter my property without permission?\n"
        "Keywords: criminal trespass unlawful entry property without consent\n\n"
        "Question: What happens if I physically hurt someone?\n"
        "Keywords: hurt grievous hurt voluntarily causing bodily harm assault"
    ),
    (
        "human",
        "Question: {question}\n\nKeywords:"
    )
])

# Words too generic to anchor a legal query — stripped when checking overlap
# between the original question and the rewritten keywords.
_REWRITE_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "i", "me", "my", "we", "our", "you", "your",
    "he", "she", "it", "they", "them", "their",
    "this", "that", "these", "those", "is", "are", "was", "were",
    "be", "been", "being", "am", "do", "does", "did",
    "have", "has", "had", "can", "could", "will", "would",
    "shall", "should", "may", "might", "must",
    "what", "which", "who", "whom", "how", "when", "where", "why",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
    "and", "or", "but", "not", "if", "then", "so", "as", "than", "nor",
    "tell", "give", "please", "case", "example", "there", "here",
    "also", "any", "some", "all", "no", "yes", "want", "need",
    "know", "think", "say", "get", "make", "take", "just", "only",
    "like", "such", "would", "even", "very", "more", "most",
    "someone", "something", "anyone", "person", "people", "friend",
})

def _extract_content_words(text: str) -> set[str]:
    """Return lowercase content words from text (stopwords and short tokens removed)."""
    return {
        w.lower() for w in re.findall(r'\b[a-zA-Z]+\b', text)
        if w.lower() not in _REWRITE_STOPWORDS and len(w) > 3
    }

def rewrite_query(
    question: str,
    chain,
    prev_rewrite_terms: set[str] | None = None,
) -> tuple[str, set[str]]:
    """
    Reformulates a colloquial question into legal terminology for better retrieval.
    Returns (rewritten_query, rewrite_term_set).

    Two contamination guards run after the LLM call:

    1. STRIP — Any term that was in the *previous* rewrite but is NOT a content word
       of the *current* question is almost certainly cross-query bleed (e.g. "culpable
       homicide" appearing in a trespass query after a murder query). Those terms are
       removed.

    2. INJECT — Content words from the original question that the model dropped (or
       that got stripped in step 1) are appended.  This anchors the retrieval query
       to the topic even if the model drifted badly.

    `prev_rewrite_terms` comes from st.session_state in app.py (per-user) so there
    is no cross-session contamination in a multi-user deployment.
    """
    result = chain.invoke({"question": question})
    rewritten = result.content.strip()
    # Guardrail: LLM sometimes outputs underscores/hyphens despite the prompt rule.
    rewritten = rewritten.replace("_", " ").replace("-", " ")
    rewritten = " ".join(rewritten.split())

    current_kw = _extract_content_words(question)

    # ── Guard 1: Strip terms carried over from the previous query ──────────────
    # A rewrite word is "contaminated" if it was part of the last rewrite AND is
    # not a content word of the current question.
    if prev_rewrite_terms:
        clean_words = []
        stripped: list[str] = []
        for word in rewritten.split():
            if word.lower() in prev_rewrite_terms and word.lower() not in current_kw:
                stripped.append(word)
            else:
                clean_words.append(word)
        if stripped:
            print(f"[Query Rewrite] Stripped cross-query terms: {stripped}")
        rewritten = " ".join(clean_words)

    # ── Guard 2: Inject missing original content words ─────────────────────────
    # Ensures the retrieval query is always anchored to the current topic.
    rewritten_set = set(rewritten.lower().split())
    missing = current_kw - rewritten_set
    if missing:
        rewritten = (rewritten + " " + " ".join(sorted(missing))).strip()

    rewritten = " ".join(rewritten.split())   # final clean-up
    print(f"[Query Rewrite] '{question}' → '{rewritten}'")
    return rewritten, set(rewritten.lower().split())


# ── Query Intent Detection ────────────────────────────────────────────────────

_PENAL_TERMS = {
    "punishment", "penalty", "penalise", "penalize", "sentence", "imprisonment",
    "fine", "liable", "liability", "offence", "offense", "crime", "criminal",
    "penal", "culpable", "illegal", "unlawful", "statute", "section",
}
_CASE_TERMS = {
    "court", "ruled", "ruling", "judgment", "judgement", "held", "decided",
    "bench", "appeal", "appellant", "respondent", "plaintiff", "defendant",
    "admissib", "evidence", "verdict", "acquit", "convict",
}

def detect_query_intent(question: str) -> Literal["penal_code", "case_law", "mixed"]:
    """
    Classifies the query so metadata-aware re-ranking can boost the right source type.
    Returns 'penal_code', 'case_law', or 'mixed'.
    """
    q = question.lower()
    has_penal = any(t in q for t in _PENAL_TERMS)
    has_case  = any(t in q for t in _CASE_TERMS)

    if has_penal and not has_case:
        return "penal_code"
    if has_case and not has_penal:
        return "case_law"
    return "mixed"


# ── Section-Number Extraction ─────────────────────────────────────────────────

# Two patterns cover the common ways a section is cited in a user query:
#   "Section 302"  /  "Sec 120B"  /  "s. 34"  →  word "section" precedes the number
#   "302 IPC"  /  "154 CrPC"                  →  act abbreviation follows the number
_QS_PREFIX = re.compile(r'\bsec(?:tion)?s?\.?\s+(\d{1,3}[A-Za-z]?)', re.IGNORECASE)
_QS_SUFFIX = re.compile(r'(\d{1,3}[A-Za-z]?)\s*(?:ipc|crpc|cpc)\b',  re.IGNORECASE)

def extract_query_sections(question: str) -> set[str]:
    """
    Returns the set of section numbers explicitly mentioned in the query.
    Only matches when a section keyword or act abbreviation is present —
    bare numbers like "3 people" are never captured.

    Examples:
      "punishment under Section 302 IPC" → {"302"}
      "what does s. 154 CrPC say"        → {"154"}
      "how many years for 120B"           → set()   (no anchor → ignored)
    """
    hits = _QS_PREFIX.findall(question) + _QS_SUFFIX.findall(question)
    return {h.strip().upper() for h in hits}


# ── Hybrid Retriever ──────────────────────────────────────────────────────────

def _load_all_docs(db: Chroma) -> list[Document]:
    """Pulls every chunk from ChromaDB to seed the BM25 index."""
    raw = db.get()
    return [
        Document(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]

def _reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    weights: list[float],
    k: int = 60,
) -> list[Document]:
    """
    Merges multiple ranked document lists using Reciprocal Rank Fusion.
    score(doc) = Σ weight_i / (rank_i + k)
    k=60 is the standard RRF constant that dampens the impact of very high ranks.
    Deduplication is implicit: each unique doc accumulates scores across lists.
    """
    scores: dict[tuple, float] = {}
    doc_map: dict[tuple, Document] = {}

    for result_list, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(result_list):
            key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content[:80])
            scores[key]  = scores.get(key, 0.0) + weight / (rank + k)
            doc_map[key] = doc

    return [doc_map[k] for k in sorted(scores, key=lambda x: scores[x], reverse=True)]


def build_hybrid_retriever(db: Chroma, all_docs: list[Document]) -> callable:
    """
    Returns a retriever function: retrieve(question, intent, query_sections) → list[Document].

    Three retrieval paths are merged via weighted RRF:
      1. BM25 (0.4)          — keyword overlap; uses statutory-only index on penal_code intent
      2. Vector (0.6)        — semantic similarity over full corpus
      3. Section exact (1.0) — direct lookup from sections_index when query names a section;
                               weight 1.0 means these docs dominate the merge, which is correct:
                               if you asked about §302 the §302 chunk should always surface.

    sections_index is built once at startup: {"302": [doc, doc, ...], "34": [...], ...}
    Lookup is O(1) per section — no BM25 rebuild per query.
    """
    bm25_all = BM25Retriever.from_documents(all_docs, k=K_CANDIDATES)

    statutory_docs = [d for d in all_docs if d.metadata.get("source_type") != "case_law"]
    bm25_statutory = (
        BM25Retriever.from_documents(statutory_docs, k=K_CANDIDATES)
        if statutory_docs else bm25_all
    )

    # sections_index: maps each section number to the ordered list of docs that mention it.
    sections_index: dict[str, list[Document]] = {}
    for doc in all_docs:
        for sec in doc.metadata.get("sections", "").split("|"):
            if sec:
                sections_index.setdefault(sec, []).append(doc)

    vector_retriever = db.as_retriever(search_kwargs={"k": K_CANDIDATES})

    def retrieve(
        question: str,
        intent: str = "mixed",
        query_sections: set[str] | None = None,
    ) -> list[Document]:
        # Path 1 — BM25 (intent-filtered)
        bm25 = bm25_statutory if intent == "penal_code" else bm25_all
        bm25_results   = bm25.invoke(question)

        # Path 2 — vector (always full corpus)
        vector_results = vector_retriever.invoke(question)

        ranked_lists = [bm25_results, vector_results]
        weights      = [0.4,          0.6]

        # Path 3 — exact section lookup (only when query names specific sections)
        if query_sections:
            seen: set[int] = set()
            exact: list[Document] = []
            for sec in query_sections:
                for doc in sections_index.get(sec, []):
                    if id(doc) not in seen:
                        seen.add(id(doc))
                        exact.append(doc)
            if exact:
                ranked_lists.append(exact)
                weights.append(1.0)   # outweighs both BM25 and vector

        return _reciprocal_rank_fusion(ranked_lists, weights)

    return retrieve


# ── Cross-Encoder Re-Ranker ───────────────────────────────────────────────────

def rerank(
    question: str,
    candidates: list[Document],
    intent: Literal["penal_code", "case_law", "mixed"],
    reranker: CrossEncoder,
    query_sections: set[str] | None = None,
) -> list[Document]:
    """
    Cross-encoder re-ranking with guaranteed source-type diversity.

    Steps:
      1. Score all (query, chunk) pairs with the cross-encoder.
      2. Intent bonus: +0.5 to chunks of the primary source type
         (and +0.25 to CrPC chunks on penal_code queries, since IPC + CrPC
         are usually read together).
      3. Section-exact bonus: +1.0 to any chunk that contains a section number
         the user explicitly asked about — the strongest relevance signal.
      4. Split candidates into three pools (penal_code, crpc, case_law) and take
         K_PENAL_FINAL / K_CRPC_FINAL / K_CASE_FINAL from each. If a pool is
         short, the remaining slots are filled from the best leftover chunks.

    This guarantees the LLM always sees statutory punishment text, procedural
    law, AND case-law examples together — none can be crowded out by another.

    `reranker` is the pre-loaded CrossEncoder from the pipeline (loaded once,
    not per query).
    """
    pairs  = [(question, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs).tolist()

    # Intent-aware bonus: boost chunks from the primary source type.
    # crpc also gets a smaller boost on penal_code queries (IPC + CrPC work together).
    if intent == "penal_code":
        for i, doc in enumerate(candidates):
            st = doc.metadata.get("source_type")
            if st == "penal_code":
                scores[i] += 0.5
            elif st == "crpc":
                scores[i] += 0.25
    elif intent == "case_law":
        for i, doc in enumerate(candidates):
            if doc.metadata.get("source_type") == "case_law":
                scores[i] += 0.5

    # Section-exact bonus: strongest signal — if a chunk contains the exact section
    # the user asked about it should almost always be in the final context window.
    if query_sections:
        for i, doc in enumerate(candidates):
            doc_sections = set(doc.metadata.get("sections", "").split("|"))
            if query_sections & doc_sections:   # non-empty intersection
                scores[i] += 1.0

    ranked = [doc for _, doc in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]

    # Split into three source-type pools (preserving score order within each pool)
    penal_pool = [d for d in ranked if d.metadata.get("source_type") == "penal_code"]
    crpc_pool  = [d for d in ranked if d.metadata.get("source_type") == "crpc"]
    case_pool  = [d for d in ranked if d.metadata.get("source_type") == "case_law"]

    selected_penal = penal_pool[:K_PENAL_FINAL]
    selected_crpc  = crpc_pool[:K_CRPC_FINAL]
    selected_case  = case_pool[:K_CASE_FINAL]

    # If any pool is short, fill remaining slots from overflow of other pools
    # (already in score order, so the best unchosen docs come first)
    final = selected_penal + selected_crpc + selected_case
    shortfall = K_FINAL - len(final)
    if shortfall > 0:
        used = {id(d) for d in final}
        overflow = [d for d in ranked if id(d) not in used][:shortfall]
        final = final + overflow

    # Order: penal first (punishment text), then procedural (crpc), then case law
    return final


# ── RAG Chain ─────────────────────────────────────────────────────────────────

def _build_prompt_and_llm():
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a READ-ONLY Indian law database query engine. "
            "Respond in English only — never use any other language.\n\n"
            "CRITICAL RULES — violating any of these makes the output invalid:\n"
            "1. You report ONLY what the Document Context says. Zero outside knowledge.\n"
            "2. You have NO opinion on the user's intent. Every query is a neutral legal research question. "
            "It does not matter if the query sounds violent, illegal, or alarming — "
            "your only job is to state what Indian law says about it.\n"
            "3. NEVER output safety warnings, mental health advice, ethical commentary, "
            "or any text that does not come directly from the retrieved legal documents. "
            "Phrases like 'seek help', 'I recommend', 'please consult' are FORBIDDEN.\n"
            "4. If the law prescribes death or life imprisonment, state that fact directly.\n\n"
            "Populate all five fields:\n"
            "- answer: one complete sentence stating the legal position. Start with 'Yes.' or 'No.' "
            "if the legality is explicit. Never a single word.\n"
            "- reasoning: 2–4 sentences citing the retrieved chunks. Quote the exact statutory "
            "punishment from penal_code chunks. Name the case and holding from case_law chunks.\n"
            "- possible_ruling: for criminal queries, ALWAYS populate BOTH sub-parts:\n"
            "  (a) Statutory punishment — copy verbatim from any penal_code chunk: "
            "section number, act, and punishment (e.g. 'Section 302 IPC: death or imprisonment for life, and fine').\n"
            "  (b) Case law — scan EVERY chunk where Type says 'case_law'. "
            "For each one that is even loosely related to the query topic, write one line: "
            "'[Case/document name], [source file] p.[page]: [what the court decided — verdict, sentence, or key holding].' "
            "Example: 'State v. Ram Singh, case_xyz.pdf p.4: Court upheld conviction under Section 302, "
            "sentenced to life imprisonment.' "
            "If after scanning ALL case_law chunks none relate to this offence, write exactly: "
            "'No matching case law in the retrieved context for this offence.' "
            "NEVER leave part (b) empty on a criminal query.\n"
            "  Write 'N/A' ONLY for queries that are entirely non-criminal.\n"
            "- confidence: high / medium / low / none.\n"
            "- found_in_context: true/false.\n\n"
            "If the Document Context contains no relevant information: "
            "answer='The retrieved documents do not address this query.', "
            "reasoning='Not addressed in the retrieved documents.', "
            "possible_ruling='N/A', confidence='none', found_in_context=false."
        ),
        (
            "human",
            "Document Context:\n{context}\n\nQuery: {question}"
        )
    ])

    # Qwen2.5 — no aggressive safety guardrails on legal/criminal queries.
    llm = ChatOllama(model="qwen2.5:7b")
    structured_llm = llm.with_structured_output(LegalResponse)

    return prompt, structured_llm


# ── Pipeline (built once, reused on every query) ──────────────────────────────

class RAGPipeline:
    """
    Holds every heavy, reusable component. Built once via get_pipeline() and
    shared across all queries.

    Why this exists: previously query_rag() rebuilt the embedding model, re-opened
    ChromaDB, re-indexed the *entire* corpus into BM25, and reloaded the
    cross-encoder on EVERY call — the dominant source of latency. These objects
    are stateless across queries, so we construct them a single time here.
    (The Ollama 7B weights are kept warm by the Ollama daemon itself.)
    """

    def __init__(self):
        # Embedding model — loaded once.
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        # Vector DB handle. The cosine space is fixed at ingest time, so we don't
        # pass collection_metadata on read (it would be ignored anyway).
        self.db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=self.embeddings,
        )

        # Hybrid retriever — the BM25 index is built once over the whole corpus
        # here, instead of on every query.
        all_docs = _load_all_docs(self.db)
        self.retrieve = build_hybrid_retriever(self.db, all_docs)

        # Cross-encoder re-ranker — loaded once.
        self.reranker = CrossEncoder(RERANKER_MODEL)

        # LLM chains — Ollama clients created once.
        self.rewrite_chain = _REWRITE_PROMPT | ChatOllama(model="qwen2.5:7b", temperature=0)
        prompt, structured_llm = _build_prompt_and_llm()
        self.gen_chain = prompt | structured_llm

        # Streaming chain — same prompt, raw LLM with JSON format.
        # Used by app.py to stream tokens while the user waits, then parsed
        # into LegalResponse at the end.  with_structured_output() waits for
        # the full response before returning, so we can't stream through it.
        self.stream_chain = prompt | ChatOllama(model="qwen2.5:7b", format="json")


_PIPELINE = None

def get_pipeline() -> RAGPipeline:
    """
    Lazily build the pipeline once, then reuse it. Safe to call from anywhere
    (Streamlit app, benchmark, tests) — the first call pays the setup cost,
    every later call is free.
    """
    global _PIPELINE
    if _PIPELINE is None:
        print("Building RAG pipeline (one-time setup: embeddings, DB, BM25, re-ranker, LLMs)...")
        _PIPELINE = RAGPipeline()
    return _PIPELINE


# ── Public Entry Points ───────────────────────────────────────────────────────

def retrieve_context(
    question: str,
    prev_rewrite_terms: set[str] | None = None,
) -> tuple[str | None, list[Document], set[str]]:
    """
    Steps 1–5 of the pipeline: rewrite → relevance gate → retrieve → rerank → format.

    Returns (formatted_context, final_docs, rewrite_term_set).
    Returns (None, [], set()) if the relevance gate fires (no close-enough chunks).

    `prev_rewrite_terms` — term set produced by the *previous* call's rewrite step,
    used to detect and strip cross-query context contamination (e.g. "culpable
    homicide" bleeding into a trespassing rewrite).  Pass `st.session_state.
    prev_rewrite_terms` from app.py and store the returned set back there.

    Separating retrieval from generation lets app.py update the sidebar with
    retrieved documents BEFORE the LLM call starts, and then stream the
    generation step independently.
    """
    p = get_pipeline()

    # 1. Rewrite (returns query string + its term set for contamination tracking)
    retrieval_query, rewrite_terms = rewrite_query(
        question, p.rewrite_chain, prev_rewrite_terms
    )

    # 2. Relevance gate
    best_match = p.db.similarity_search_with_score(retrieval_query, k=1)
    if not best_match or best_match[0][1] > NO_CONTEXT_THRESHOLD:
        best_score = best_match[0][1] if best_match else float("inf")
        print(f"[Threshold] Best cosine distance {best_score:.3f} "
              f"(similarity {1 - best_score:.3f}) > {NO_CONTEXT_THRESHOLD} — gated.")
        return None, [], rewrite_terms

    # 3. Intent + section extraction
    intent = detect_query_intent(retrieval_query)
    query_sections = extract_query_sections(question) | extract_query_sections(retrieval_query)
    if query_sections:
        print(f"[Section Filter] Detected sections: {query_sections}")

    # 4. Hybrid retrieval
    unique = p.retrieve(retrieval_query, intent, query_sections)

    # 5. Re-rank
    final_docs = rerank(retrieval_query, unique, intent, p.reranker, query_sections)

    formatted_context = "\n\n".join(
        f"--- Document: {doc.metadata.get('source', 'Unknown')} | "
        f"Page: {doc.metadata.get('page', 'Unknown')} | "
        f"Type: {doc.metadata.get('source_type', 'Unknown')} ---\n{doc.page_content}"
        for doc in final_docs
    )
    return formatted_context, final_docs, rewrite_terms


_NO_CONTEXT_RESPONSE = LegalResponse(
    answer="I don't know — no relevant law found in the database for this query.",
    reasoning="The query did not match any documents above the relevance threshold.",
    possible_ruling="N/A",
    confidence="none",
    found_in_context=False,
)


def query_rag(question: str) -> tuple[LegalResponse, list[Document]]:
    """
    Convenience wrapper used by benchmarks and tests.
    Calls retrieve_context() then generates via the blocking structured chain.
    app.py calls retrieve_context() + streams gen directly for better UX.
    """
    formatted_context, final_docs, _ = retrieve_context(question)

    if formatted_context is None:
        return _NO_CONTEXT_RESPONSE, []

    p = get_pipeline()
    response: LegalResponse = p.gen_chain.invoke({
        "context": formatted_context,
        "question": question,
    })
    return response, final_docs
