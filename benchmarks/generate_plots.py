"""
generate_plots.py — visualise benchmark_metrics.csv results.

Produces two charts:
  1. End-to-end latency per query (bar chart)
  2. Confidence level distribution (count bar chart)

Run from the project root after benchmark.py:
    python benchmarks/generate_plots.py
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS_DIR = Path(__file__).parent / "results"

df = pd.read_csv(str(RESULTS_DIR / "benchmark_metrics.csv"))

short_labels = [f"Q{i + 1}" for i in range(len(df))]

CONFIDENCE_COLORS = {
    "high":   "#1e8e3e",
    "medium": "#f9a825",
    "low":    "#e65100",
    "none":   "#c62828",
    "error":  "#888888",
}

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Legal RAG Pipeline — Benchmark results", fontsize=15, fontweight="bold")

# ── Plot 1: End-to-end latency ────────────────────────────────────────────────
bar_colors = [CONFIDENCE_COLORS.get(c, "#888") for c in df["Confidence"]]
bars = axs[0].bar(short_labels, df["Total_Time_sec"], color=bar_colors, edgecolor="white")
axs[0].set_title("End-to-End Latency per Query (seconds)", fontsize=13)
axs[0].set_ylabel("Time (s)", fontsize=11)
axs[0].set_xlabel("Query", fontsize=11)
for bar, val in zip(bars, df["Total_Time_sec"]):
    axs[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{val:.1f}s", ha="center", fontsize=9, fontweight="bold")

# Legend for confidence colours
legend_patches = [
    mpatches.Patch(color=color, label=label.capitalize())
    for label, color in CONFIDENCE_COLORS.items() if label != "error"
]
axs[0].legend(handles=legend_patches, title="Confidence", fontsize=9, title_fontsize=9)

# ── Plot 2: Confidence distribution ──────────────────────────────────────────
confidence_order = ["high", "medium", "low", "none", "error"]
conf_counts = df["Confidence"].value_counts().reindex(confidence_order, fill_value=0)
palette = [CONFIDENCE_COLORS[c] for c in confidence_order]

axs[1].bar(conf_counts.index, conf_counts.values, color=palette, edgecolor="white")
axs[1].set_title("Confidence Level Distribution", fontsize=13)
axs[1].set_ylabel("Number of Queries", fontsize=11)
axs[1].set_xlabel("Confidence", fontsize=11)
for i, (label, val) in enumerate(conf_counts.items()):
    if val > 0:
        axs[1].text(i, val + 0.05, str(val), ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
output_path = RESULTS_DIR / "rag_performance_metrics.png"
plt.savefig(str(output_path), dpi=300)
print(f"Plot saved to '{output_path}'.")
