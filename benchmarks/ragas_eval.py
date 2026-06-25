"""
ragas_eval.py — correctness evaluation of the Legal RAG pipeline using RAGAS.

Metrics measured:
  - answer_relevancy  : are the answers on-topic for the questions?
                        Computed as cosine similarity between embeddings of the
                        original question and LLM-generated paraphrase questions
                        from the answer — no LLM verdict parsing required.

Excluded metrics and why:
  - faithfulness      : requires the judge to emit "Verdict: 1/0" per NLI claim.
                        Models under 13B (including qwen2.5:7b) don't follow that
                        exact template, so ragas hits parse errors on every row.
  - context_precision : requires per-sentence binary judgments — same issue.
  - context_recall    : same.

All evaluation is done locally — Ollama qwen2.5:7b acts as the judge LLM
and all-MiniLM-L6-v2 handles embeddings. No OpenAI key required.

Run from the project root:
    python benchmarks/ragas_eval.py
"""

import sys
import types
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ragas imports langchain_community.chat_models.vertexai at module load time,
# but newer langchain_community moved it to langchain_google_vertexai.
# Inject a dummy module so the import doesn't crash — we never use VertexAI.
if "langchain_community.chat_models.vertexai" not in sys.modules:
    _dummy = types.ModuleType("langchain_community.chat_models.vertexai")
    _dummy.ChatVertexAI = None
    sys.modules["langchain_community.chat_models.vertexai"] = _dummy

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings as LCHFEmbeddings

from retrieval import query_rag

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Ground truth reference answers ───────────────────────────────────────────
# Legally correct reference answers for each evaluation question. answer_relevancy
# (the only metric run locally) does not consume these, but they are kept so the
# reference-based metrics (context_recall / faithfulness) can be enabled later
# with a stronger judge model. Update them if the corpus jurisdiction changes.

EVAL_SET = [
    {
        "question":   "What is the punishment for murder under the penal code?",
        "reference":  "Under Section 302 of the Indian Penal Code, whoever commits murder "
                      "shall be punished with death or imprisonment for life, and shall also "
                      "be liable to fine.",
    },
    {
        "question":   "What are the exceptions to culpable homicide amounting to murder?",
        "reference":  "Section 300 IPC lists five exceptions: grave and sudden provocation, "
                      "private defence exceeding the right, act of a public servant in good faith, "
                      "sudden fight without premeditation, and consent of the victim who is above 18.",
    },
    {
        "question":   "If a police officer arrests a person without a warrant, what rights do they have?",
        "reference":  "An arrested person has the right to be informed of the grounds of arrest, "
                      "the right to bail in bailable offences, the right to legal representation, "
                      "and must be produced before a magistrate within 24 hours under Article 22 "
                      "of the Constitution and Section 57 CrPC.",
    },
    {
        "question":   "Can a confession made directly to a police officer be used as definitive evidence in court?",
        "reference":  "Under Sections 25 and 26 of the Indian Evidence Act, a confession made "
                      "to a police officer or while in police custody is inadmissible in court "
                      "as evidence against the accused.",
    },
    {
        "question":   "What is the penalty for kidnapping a minor for ransom under the penal code?",
        "reference":  "Section 364A IPC prescribes death penalty or imprisonment for life with "
                      "fine for kidnapping or abducting a person in order to compel the government "
                      "or any person to pay ransom.",
    },
    {
        "question":   "What is the punishment for theft?",
        "reference":  "Under Section 379 IPC, whoever commits theft shall be punished with "
                      "imprisonment of either description for a term which may extend to three years, "
                      "or with fine, or with both.",
    },
]


# ── Local judge configuration ─────────────────────────────────────────────────

def build_evaluator():
    """Configures the local Ollama judge and HuggingFace embeddings for RAGAS.

    Embeddings note: RagasHFEmbeddings wraps sentence-transformers directly but
    does NOT implement embed_query(), which ragas calls internally for answer_relevancy.
    LangchainEmbeddingsWrapper delegates to LangChain's HuggingFaceEmbeddings, which
    does implement embed_query() / embed_documents() — so this is the correct wrapper.
    """
    llm = LangchainLLMWrapper(ChatOllama(model="qwen2.5:7b", temperature=0))
    embeddings = LangchainEmbeddingsWrapper(
        LCHFEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    )
    return llm, embeddings


# ── Evaluation runner ─────────────────────────────────────────────────────────

def run_ragas_eval():
    print("=" * 60)
    print("RAGAS Correctness Evaluation — Legal RAG Pipeline")
    print("Judge: qwen2.5:7b (local Ollama) | Embeddings: all-MiniLM-L6-v2")
    print("=" * 60)

    records = []

    for i, item in enumerate(EVAL_SET):
        question  = item["question"]
        reference = item["reference"]
        print(f"\n[{i + 1}/{len(EVAL_SET)}] {question}")

        try:
            response, docs = query_rag(question)
            answer   = f"{response.answer} {response.reasoning}".strip()
            contexts = [doc.page_content for doc in docs] if docs else ["[No documents retrieved]"]
            print(f"    Confidence: {response.confidence} | Chunks: {len(docs)}")
        except Exception as e:
            print(f"    ERROR: {e}")
            answer   = f"ERROR: {e}"
            contexts = ["[Pipeline error — no context]"]

        records.append({
            "user_input":          question,
            "response":            answer,
            "retrieved_contexts":  contexts,
            "reference":           reference,
        })

    # Build HuggingFace Dataset expected by RAGAS
    dataset = Dataset.from_list(records)

    print("\nRunning RAGAS evaluation (sequential — one job at a time for local Ollama)...")
    llm, embeddings = build_evaluator()

    # max_workers=1 forces sequential evaluation — local Ollama can't handle
    # parallel LLM calls and will time out under concurrent load.
    # timeout=300 gives a 7B model plenty of headroom for multi-step metric reasoning.
    run_config = RunConfig(timeout=300, max_retries=3, max_workers=1)

    # faithfulness is excluded: it uses an NLI step that requires the judge LLM
    # to output "Verdict: 1" or "Verdict: 0" per claim. Models under 13B
    # (including qwen2.5:7b) consistently fail to follow that exact format,
    # causing RagasOutputParserException on every row. answer_relevancy uses
    # embedding cosine similarity instead of LLM verdicts — reliable locally.
    results = evaluate(
        dataset,
        metrics=[answer_relevancy],
        llm=llm,
        embeddings=embeddings,
        run_config=run_config,
        raise_exceptions=False,
    )

    df = results.to_pandas()

    # Save full results
    output_path = RESULTS_DIR / "ragas_scores.csv"
    df.to_csv(str(output_path), index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("RAGAS Score Summary")
    print("=" * 60)
    score_cols = ["answer_relevancy"]
    for col in score_cols:
        if col in df.columns:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                print(f"  {col:<22}: {non_null.mean():.3f}  (min {non_null.min():.3f} / max {non_null.max():.3f})  [{len(non_null)}/{len(df)} rows scored]")
            else:
                print(f"  {col:<22}: all NaN — judge LLM may not have responded in time")

    print(f"\nFull per-question scores saved to '{output_path}'.")


if __name__ == "__main__":
    run_ragas_eval()
