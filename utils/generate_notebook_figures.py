"""
Generate flowchart figures for UNISTRA NLP 2026 notebooks.

Design language:
  - Background: #2C2421 (dark warm)
  - Text: #FAF7F2 (cream)
  - Box fill: #3A302B (slightly lighter bg)
  - Box borders: #E07850 (coral)
  - Arrows: #D4A855 (gold)
  - Highlight boxes: #E07850 fill (for key steps)
  - 150 DPI, sans-serif font

Output: notebooks/figures/ directory (5 PNG flowcharts)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Colors ──────────────────────────────────────────────────────────
BG = "#2C2421"
TEXT = "#FAF7F2"
BOX_FILL = "#3A302B"
BORDER = "#E07850"
ARROW = "#D4A855"
HIGHLIGHT = "#E07850"
HIGHLIGHT_TEXT = "#FAF7F2"

OUT_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150
FONT = {"family": "sans-serif", "color": TEXT, "fontsize": 11, "ha": "center", "va": "center"}
TITLE_FONT = {"family": "sans-serif", "color": TEXT, "fontsize": 14, "fontweight": "bold",
              "ha": "center", "va": "center"}


def draw_box(ax, x, y, w, h, label, highlight=False, fontsize=11):
    """Draw a rounded box with centered text."""
    fill = HIGHLIGHT if highlight else BOX_FILL
    txt_color = HIGHLIGHT_TEXT if highlight else TEXT
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.15",
        facecolor=fill, edgecolor=BORDER, linewidth=2
    )
    ax.add_patch(rect)
    ax.text(x, y, label, fontsize=fontsize, color=txt_color,
            ha="center", va="center", fontfamily="sans-serif", fontweight="bold" if highlight else "normal")


def draw_arrow(ax, x1, y1, x2, y2, horizontal=False):
    """Draw a gold arrow between two points."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=ARROW, lw=2.5, mutation_scale=18)
    )


def setup_ax(fig, ax, title=None):
    """Style axis with dark background."""
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, color=TEXT, fontfamily="sans-serif",
                      fontweight="bold", pad=15)


# ════════════════════════════════════════════════════════════════════
# Figure 1: BERTopic Pipeline (vertical, 6 boxes) — NB04
# ════════════════════════════════════════════════════════════════════
def make_bertopic_pipeline():
    fig, ax = plt.subplots(figsize=(5, 8))
    setup_ax(fig, ax, "BERTopic Pipeline")

    steps = [
        ("Documents", False),
        ("Sentence Embeddings", False),
        ("UMAP\n(Dimensionality Reduction)", False),
        ("HDBSCAN\n(Clustering)", True),
        ("c-TF-IDF\n(Topic Representation)", False),
        ("LLM Topic Naming\n(via Groq)", True),
    ]

    w, h = 3.2, 0.7
    x = 2.5
    y_start = 7.0
    gap = 1.1

    for i, (label, hl) in enumerate(steps):
        y = y_start - i * gap
        draw_box(ax, x, y, w, h, label, highlight=hl, fontsize=10)
        if i < len(steps) - 1:
            draw_arrow(ax, x, y - h / 2, x, y - gap + h / 2)

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 8.5)
    fig.savefig(OUT_DIR / "bertopic_pipeline.png", dpi=DPI, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  ✓ bertopic_pipeline.png")


# ════════════════════════════════════════════════════════════════════
# Figure 2: SetFit Pipeline (vertical, 2 phases) — NB05
# ════════════════════════════════════════════════════════════════════
def make_setfit_pipeline():
    fig, ax = plt.subplots(figsize=(5.5, 8.5))
    setup_ax(fig, ax, "SetFit: Two-Phase Training")

    x = 2.75
    w, h = 3.6, 0.65

    # Phase 1 label
    ax.text(x, 7.8, "Phase 1: Contrastive Fine-tuning", fontsize=11, color=ARROW,
            ha="center", va="center", fontfamily="sans-serif", fontweight="bold")

    phase1 = [
        ("Few-shot Examples (8 texts)", False),
        ("Generate Contrastive Pairs", False),
        ("Fine-tune Sentence\nTransformer", True),
    ]

    y = 7.2
    for i, (label, hl) in enumerate(phase1):
        draw_box(ax, x, y, w, h, label, highlight=hl, fontsize=10)
        if i < len(phase1) - 1:
            draw_arrow(ax, x, y - h / 2, x, y - 1.0 + h / 2)
        y -= 1.0

    # Divider
    y -= 0.3
    ax.plot([0.5, 5.0], [y, y], color=ARROW, linewidth=1, linestyle="--", alpha=0.5)

    # Phase 2 label
    y -= 0.4
    ax.text(x, y, "Phase 2: Classification Head", fontsize=11, color=ARROW,
            ha="center", va="center", fontfamily="sans-serif", fontweight="bold")

    phase2 = [
        ("Encode Examples with\nFine-tuned Model", False),
        ("Train Logistic Regression", False),
        ("Ready for Inference!", True),
    ]

    y -= 0.6
    for i, (label, hl) in enumerate(phase2):
        draw_box(ax, x, y, w, h, label, highlight=hl, fontsize=10)
        if i < len(phase2) - 1:
            draw_arrow(ax, x, y - h / 2, x, y - 1.0 + h / 2)
        y -= 1.0

    ax.set_xlim(0, 5.5)
    ax.set_ylim(0.5, 8.5)
    fig.savefig(OUT_DIR / "setfit_pipeline.png", dpi=DPI, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  ✓ setfit_pipeline.png")


# ════════════════════════════════════════════════════════════════════
# Figure 3: Reranking Pipeline (vertical, 2 stages) — NB07
# ════════════════════════════════════════════════════════════════════
def make_reranking_pipeline():
    fig, ax = plt.subplots(figsize=(5.5, 8))
    setup_ax(fig, ax, "Two-Stage Retrieval Pipeline")

    x = 2.75
    w, h = 3.6, 0.65

    # Top
    draw_box(ax, x, 7.2, w, h, "Full Corpus\n(thousands of docs)", fontsize=10)
    draw_arrow(ax, x, 7.2 - h / 2, x, 6.2 + h / 2)

    # Stage 1
    ax.text(x, 6.55, "Stage 1", fontsize=9, color=ARROW, ha="center",
            fontfamily="sans-serif", fontstyle="italic")
    draw_box(ax, x, 6.2, w, h, "Bi-encoder Retrieval\n(fast, top-K)", highlight=False, fontsize=10)

    # Arrow with label
    draw_arrow(ax, x, 6.2 - h / 2, x, 5.0 + h / 2)
    ax.text(x + 1.9, 5.6, "top-K\ncandidates", fontsize=8, color=TEXT,
            ha="center", va="center", fontfamily="sans-serif", fontstyle="italic", alpha=0.7)

    # Stage 2
    ax.text(x, 5.35, "Stage 2", fontsize=9, color=ARROW, ha="center",
            fontfamily="sans-serif", fontstyle="italic")
    draw_box(ax, x, 5.0, w, h, "Cross-encoder Reranking\n(accurate, top-N)", highlight=True, fontsize=10)

    draw_arrow(ax, x, 5.0 - h / 2, x, 3.8 + h / 2)

    # Result
    draw_box(ax, x, 3.8, w, h, "Precise Results\n→ User", highlight=False, fontsize=10)

    ax.set_xlim(0, 5.5)
    ax.set_ylim(3.0, 8.0)
    fig.savefig(OUT_DIR / "reranking_pipeline.png", dpi=DPI, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  ✓ reranking_pipeline.png")


# ════════════════════════════════════════════════════════════════════
# Figure 4: Distillation Pipeline (horizontal, 5 boxes) — NB08
# ════════════════════════════════════════════════════════════════════
def make_distillation_pipeline():
    fig, ax = plt.subplots(figsize=(12, 3))
    setup_ax(fig, ax, "Knowledge Distillation Pipeline")

    steps = [
        ("Unlabeled\nData", False),
        ("LLM Labels\n(Teacher)", True),
        ("Confidence\nFilter + Dedup", False),
        ("Train Student\n(TF-IDF / SBERT)", False),
        ("Fast Inference\n(no API needed)", True),
    ]

    w, h = 2.0, 1.2
    y = 1.5
    x_start = 1.5
    gap = 2.6

    for i, (label, hl) in enumerate(steps):
        x = x_start + i * gap
        draw_box(ax, x, y, w, h, label, highlight=hl, fontsize=10)
        if i < len(steps) - 1:
            draw_arrow(ax, x + w / 2, y, x + gap - w / 2, y, horizontal=True)

    ax.set_xlim(0, 13)
    ax.set_ylim(0, 3.2)
    fig.savefig(OUT_DIR / "distillation_pipeline.png", dpi=DPI, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  ✓ distillation_pipeline.png")


# ════════════════════════════════════════════════════════════════════
# Figure 5: Distillation Concept (compact, 3 items) — NB08
# ════════════════════════════════════════════════════════════════════
def make_distillation_concept():
    fig, ax = plt.subplots(figsize=(9, 2.5))
    setup_ax(fig, ax, "Knowledge Distillation: The Idea")

    y = 1.2
    w, h = 2.2, 1.0

    # Teacher
    draw_box(ax, 1.5, y, w, h, "LLM Teacher\n(slow, expensive,\nsmart)", highlight=True, fontsize=10)

    # Arrow with "labels" annotation
    draw_arrow(ax, 1.5 + w / 2, y, 4.5 - w / 2, y, horizontal=True)
    ax.text(3.5, y + 0.55, "labels", fontsize=10, color=ARROW, ha="center",
            fontfamily="sans-serif", fontstyle="italic")

    # Student
    draw_box(ax, 4.5, y, 1.6, h, "Labels", highlight=False, fontsize=10)

    # Arrow
    draw_arrow(ax, 4.5 + 0.8, y, 7.5 - w / 2, y, horizontal=True)
    ax.text(6.15, y + 0.55, "train", fontsize=10, color=ARROW, ha="center",
            fontfamily="sans-serif", fontstyle="italic")

    # Student model
    draw_box(ax, 7.5, y, w, h, "Student Model\n(fast, cheap,\nfocused)", highlight=True, fontsize=10)

    ax.set_xlim(0, 9)
    ax.set_ylim(0, 2.5)
    fig.savefig(OUT_DIR / "distillation_concept.png", dpi=DPI, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  ✓ distillation_concept.png")


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Generating figures to {OUT_DIR}/\n")
    make_bertopic_pipeline()
    make_setfit_pipeline()
    make_reranking_pipeline()
    make_distillation_pipeline()
    make_distillation_concept()
    print(f"\nDone! {len(list(OUT_DIR.glob('*.png')))} figures generated.")
