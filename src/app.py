import json
import re
import streamlit as st
from pydantic import ValidationError
from retrieval import retrieve_context, get_pipeline, LegalResponse
from dotenv import load_dotenv

# ── Safety-refusal detector ───────────────────────────────────────────────────
# Small models occasionally leak safety-guardrail language into structured
# output despite the system prompt.  We filter at the sentence level so that
# valid legal sentences before/after the advice sentence are kept.
#
# Phrases are matched as substrings (case-insensitive) against individual
# sentences split on [.!?].  Keep this list narrow — false positives would
# suppress real legal content (e.g. "professional misconduct").
_SAFETY_PHRASES = (
    # Generic advice / referral language
    "seek help", "seeking help",
    "mental health",
    "highly recommended",
    "considering harm", "negative impact",
    "consult a",
    "i recommend", "i strongly",
    "please note",
    "it is important to",
    "i must advise",
    "this is not something i",
    "i cannot assist",
    # "always consider / before resorting" pattern from self-defence queries
    "always consider",
    "before resorting",
    # "contact local law enforcement for guidance" pattern
    "contact local",
    "law enforcement for guidance",
)

def _sanitize_field(text: str) -> str:
    """
    Strip individual sentences that contain safety-refusal language, keeping
    the rest of the field intact.

    Previously we blanked the whole field on a single hit, which could discard
    valid legal content that happened to appear in the same field as one bad
    sentence.  Sentence-level filtering is more surgical.
    """
    if not text:
        return text
    # Split on sentence boundaries; keep the delimiter attached so punctuation
    # is preserved when we rejoin.
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    clean = [s for s in sentences if not any(p in s.lower() for p in _SAFETY_PHRASES)]
    return " ".join(clean)

load_dotenv()

st.set_page_config(page_title="Legal Case Law RAG", layout="wide", page_icon="⚖️")


# ── One-time pipeline load ────────────────────────────────────────────────────
# @st.cache_resource builds the pipeline once per process and shares that single
# instance across every rerun and user session. Without this, Streamlit's
# top-to-bottom re-execution would rebuild embeddings, the BM25 index, and the
# cross-encoder on every message.
@st.cache_resource(show_spinner="Loading RAG pipeline (one-time setup)...")
def load_pipeline():
    return get_pipeline()

load_pipeline()

st.markdown("""
<style>
    .source-box {
        background-color: #f1f3f4;
        padding: 15px; border-radius: 8px; margin-bottom: 15px;
        font-family: monospace; font-size: 13px; color: #202124;
        white-space: pre-wrap; word-wrap: break-word;
    }
    .source-header { font-weight: bold; color: #1a73e8; margin-bottom: 5px; }
    .ref-chip {
        display: inline-block; background: #e8f0fe; color: #1a73e8;
        border-radius: 12px; padding: 2px 10px; margin: 2px 4px 2px 0;
        font-size: 12px; font-family: monospace;
    }
    @media (prefers-color-scheme: dark) {
        .source-box  { background-color: #2b2b2b; color: #e8eaed; }
        .source-header { color: #8ab4f8; }
        .ref-chip    { background: #1e3a5f; color: #8ab4f8; }
    }
</style>
""", unsafe_allow_html=True)

st.title("⚖️ Legal RAG Assistant")
st.caption("Answers are grounded in retrieved case law and statutory text — powered locally via Ollama.")

# ── Confidence badge helpers ──────────────────────────────────────────────────
CONFIDENCE_STYLE = {
    "high":   ("🟢", "#1e8e3e", "High confidence"),
    "medium": ("🟡", "#f9a825", "Medium confidence"),
    "low":    ("🟠", "#e65100", "Low confidence"),
    "none":   ("🔴", "#c62828", "Not found in context"),
}

def confidence_badge(level: str) -> str:
    icon, color, label = CONFIDENCE_STYLE.get(level, ("⚪", "#888", level))
    return (
        f'<span style="background:{color};color:#fff;border-radius:10px;'
        f'padding:2px 10px;font-size:12px;font-weight:bold;">'
        f'{icon} {label}</span>'
    )

def reference_chips(docs) -> str:
    seen = set()
    chips = ""
    for doc in docs:
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        key = f"{src}:p{page}"
        if key not in seen:
            seen.add(key)
            chips += f'<span class="ref-chip">📄 {src} p.{page}</span>'
    return chips or "<em style='color:grey'>No references</em>"

def render_context(docs):
    if not docs:
        return "<p style='color:grey'>No context retrieved yet. Send a query!</p>"
    html = ""
    for doc in docs:
        src  = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "?")
        safe = doc.page_content.replace("<", "&lt;").replace(">", "&gt;")
        html += f'<div class="source-header">📄 {src} | Page {page}</div>'
        html += f'<div class="source-box">{safe}</div>'
    return html


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # {role, content, meta?}
if "last_docs" not in st.session_state:
    st.session_state.last_docs = None
if "prev_rewrite_terms" not in st.session_state:
    # Stores the keyword set produced by the previous rewrite call.
    # Passed into retrieve_context() so it can strip cross-query contamination
    # (e.g. "culpable homicide" bleeding from a murder query into a trespass query).
    # Stored per-session so multiple users on the same server never cross-contaminate.
    st.session_state.prev_rewrite_terms = set()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📄 Retrieved Context")
    st.caption("Raw PDF chunks pulled from ChromaDB for the last query.")
    context_container = st.empty()

context_container.markdown(
    render_context(st.session_state.last_docs), unsafe_allow_html=True
)

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# ── Query input ───────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask a legal question (e.g., 'What is the punishment for theft?')"):

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            p = get_pipeline()   # already warm — instant return from @st.cache_resource

            # ── Phase 1: Retrieval (fast ~2-5s) ──────────────────────────────
            # Runs rewrite → relevance gate → BM25+vector → cross-encoder.
            # Heavy objects (embeddings, BM25 index, cross-encoder) are loaded
            # once at startup via RAGPipeline + @st.cache_resource, NOT here.
            with st.spinner("🔍 Retrieving legal context..."):
                formatted_context, docs, new_rewrite_terms = retrieve_context(
                    user_input,
                    st.session_state.prev_rewrite_terms,
                )
            # Update the per-session rewrite term set AFTER retrieval completes.
            # The next query will use this to strip cross-query contamination.
            st.session_state.prev_rewrite_terms = new_rewrite_terms

            display_html = ""

            if formatted_context is None:
                # Relevance gate fired — no LLM call needed.
                display_html = (
                    confidence_badge("none") + "<br><br>"
                    "<strong>I don't know — no relevant law found in the "
                    "database for this query.</strong><br><br>"
                    "<em>The query did not match any documents above the "
                    "relevance threshold.</em>"
                )
                st.markdown(display_html, unsafe_allow_html=True)

            else:
                # Update sidebar NOW — user sees retrieved chunks while LLM runs.
                st.session_state.last_docs = docs
                context_container.markdown(render_context(docs), unsafe_allow_html=True)

                # ── Phase 2: Streaming generation (slow ~10-30s) ─────────────
                # stream_chain uses the raw Ollama JSON endpoint so tokens arrive
                # incrementally.  with_structured_output() blocks until the full
                # JSON is ready — that is why we stream manually and parse at end.
                gen_placeholder = st.empty()
                accumulated    = ""
                token_count    = 0

                for chunk in p.stream_chain.stream({
                    "context":  formatted_context,
                    "question": user_input,
                }):
                    accumulated += chunk.content
                    token_count += 1
                    # Refresh every 20 tokens to avoid hammering Streamlit re-render.
                    if token_count % 20 == 0:
                        gen_placeholder.markdown(
                            f"⚖️ *Generating legal analysis... ({token_count} tokens)*"
                        )

                # Parse accumulated JSON → LegalResponse. If the streamed text
                # isn't valid JSON / doesn't satisfy the schema, fall back to the
                # blocking structured chain (rare — only on malformed output).
                try:
                    response = LegalResponse(**json.loads(accumulated))
                except (json.JSONDecodeError, ValidationError):
                    response = p.gen_chain.invoke({
                        "context":  formatted_context,
                        "question": user_input,
                    })

                # ── Build display HTML ────────────────────────────────────────
                badge = confidence_badge(response.confidence)
                chips = reference_chips(docs)

                # Sanitize all three text fields — models can leak safety
                # commentary into any of them.  _sanitize_field() strips
                # individual offending sentences rather than the whole field.
                answer    = _sanitize_field(response.answer or "")
                reasoning = _sanitize_field(response.reasoning or "")
                ruling    = _sanitize_field(
                    response.possible_ruling.strip() if response.possible_ruling else ""
                )

                ruling_section = (
                    f"<br><br><strong>⚖️ Punishment &amp; Precedent:</strong><br>{ruling}"
                    if ruling and ruling != "N/A"
                    else ""
                )

                display_html = (
                    f"{badge}<br><br>"
                    f"<strong>{answer}</strong><br><br>"
                    f"{reasoning}"
                    f"{ruling_section}<br><br>"
                    f"<strong>References:</strong><br>{chips}"
                )

                gen_placeholder.markdown(display_html, unsafe_allow_html=True)

            # ── Persist to session ────────────────────────────────────────────
            st.session_state.messages.append({
                "role": "assistant",
                "content": display_html,
            })
            if docs:
                st.session_state.last_docs = docs
                context_container.markdown(render_context(docs), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")
