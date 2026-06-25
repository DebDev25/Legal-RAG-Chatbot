"""
benchmark.py — end-to-end latency and quality benchmark for the Legal RAG pipeline.

Runs a set of representative queries through the full query_rag() pipeline
(query rewriting → cosine threshold gate → hybrid BM25+vector retrieval →
cross-encoder re-ranking → Ollama generation) and records per-query metrics.

Run from the project root:
    python benchmarks/benchmark.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from retrieval import query_rag

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

QUESTIONS = [
    "What is the punishment for murder under the penal code?",
    "What are the exceptions to culpable homicide amounting to murder?",
    "If a police officer arrests a person without a warrant, what rights do they have?",
    "Can a confession made directly to a police officer be used as definitive evidence in court?",
    "What is the penalty for kidnapping a minor for ransom under the penal code?",
    "What is the punishment for theft?",
    "What constitutes culpable homicide not amounting to murder?",
]


def run_benchmark():
    print("=" * 60)
    print("Legal RAG Pipeline Benchmark")
    print("Pipeline: query rewrite → cosine gate → BM25+vector → cross-encoder → Ollama qwen2.5:7b")
    print("=" * 60)

    results = []

    for i, question in enumerate(QUESTIONS):
        print(f"\n[{i + 1}/{len(QUESTIONS)}] {question}")

        start = time.time()
        try:
            response, docs = query_rag(question)
            elapsed = time.time() - start

            # Source file of the top-ranked chunk (docs are already re-ranked,
            # so docs[0] is the most relevant chunk for this query).
            best_source = docs[0].metadata.get("source", "N/A") if docs else "N/A"

            result = {
                "Question":            question,
                "Total_Time_sec":      round(elapsed, 2),
                "Confidence":          response.confidence,
                "Found_In_Context":    response.found_in_context,
                "Chunks_Retrieved":    len(docs),
                "Top_Source":          best_source,
                "Answer":              response.answer.strip(),
                "Reasoning":           response.reasoning.strip(),
                "Possible_Ruling":     response.possible_ruling.strip(),
            }
            print(f"    Confidence: {response.confidence} | Found: {response.found_in_context} "
                  f"| Time: {elapsed:.2f}s | Chunks: {len(docs)}")

        except Exception as e:
            elapsed = time.time() - start
            print(f"    ERROR after {elapsed:.2f}s: {e}")
            result = {
                "Question":         question,
                "Total_Time_sec":   round(elapsed, 2),
                "Confidence":       "error",
                "Found_In_Context": False,
                "Chunks_Retrieved": 0,
                "Top_Source":       "N/A",
                "Answer":           f"ERROR: {e}",
                "Reasoning":        "",
                "Possible_Ruling":  "",
            }

        results.append(result)

    df = pd.DataFrame(results)
    output_path = RESULTS_DIR / "benchmark_metrics.csv"
    df.to_csv(str(output_path), index=False)
    print(f"\n{'=' * 60}")
    print(f"Benchmark complete. Results saved to '{output_path}'.")
    print(f"Avg total time : {df['Total_Time_sec'].mean():.2f}s")
    print(f"Found in context: {df['Found_In_Context'].sum()}/{len(df)} queries")
    print(f"Confidence breakdown:\n{df['Confidence'].value_counts().to_string()}")


if __name__ == "__main__":
    run_benchmark()
