"""
Generate evaluation results table and success rate over epochs graph.
Data extracted from runs_record_manual.md evaluation logs.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Raw data from evaluation runs ──

eval_data = {
    25:  {"successes": 3,  "total": 20, "success_steps": [240, 110, 263]},
    50:  {"successes": 1,  "total": 20, "success_steps": [168]},
    75:  {"successes": 4,  "total": 20, "success_steps": [272, 113, 148, 99]},
    100: {"successes": 11, "total": 20, "success_steps": [144, 193, 82, 97, 390, 105, 92, 169, 118, 397, 421]},
}

epochs = sorted(eval_data.keys())
success_rates = []
avg_steps_to_success = []

for ep in epochs:
    d = eval_data[ep]
    rate = d["successes"] / d["total"] * 100
    success_rates.append(rate)
    steps = d["success_steps"]
    avg_steps_to_success.append(np.mean(steps) if steps else float("nan"))

# ═══════════════════════════════════════════════════════════
# Figure 1: Table
# ═══════════════════════════════════════════════════════════

fig_table, ax_t = plt.subplots(figsize=(10, 4))
ax_t.axis("off")

col_labels = ["Epoch", "Success Rate", "Successes", "Avg Steps to Success"]
cell_text = []
for ep, rate, avg_s in zip(epochs, success_rates, avg_steps_to_success):
    d = eval_data[ep]
    cell_text.append([
        str(ep),
        f"{rate:.0f}%",
        f"{d['successes']}/{d['total']}",
        f"{avg_s:.0f}" if not np.isnan(avg_s) else "—",
    ])

table = ax_t.table(
    cellText=cell_text,
    colLabels=col_labels,
    loc="center",
    cellLoc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.0, 2.0)

for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#2a6cb8")
    cell.set_text_props(color="white", fontweight="bold", fontsize=11)

for i in range(len(epochs)):
    rate = success_rates[i]
    if rate >= 50:
        color = "#d4edda"
    elif rate >= 15:
        color = "#fff3cd"
    else:
        color = "#f8d7da"
    for j in range(len(col_labels)):
        table[i + 1, j].set_facecolor(color)

col_widths = [0.12, 0.18, 0.15, 0.25]
for j, w in enumerate(col_widths):
    for i in range(len(epochs) + 1):
        table[i, j].set_width(w)

ax_t.set_title("Evaluation Results by Training Epoch (20 episodes each, pretrain split)",
               fontsize=13, fontweight="bold", pad=20)

fig_table.tight_layout()
fig_table.savefig("../figures/eval_results_table.png", dpi=150, bbox_inches="tight",
                  facecolor="white")

# ═══════════════════════════════════════════════════════════
# Figure 2: Success rate over epochs
# ═══════════════════════════════════════════════════════════

fig_graph, ax = plt.subplots(figsize=(8, 5))

ax.plot(epochs, success_rates, "o-", color="#2a6cb8", linewidth=2.5,
        markersize=10, markerfacecolor="white", markeredgewidth=2.5,
        markeredgecolor="#2a6cb8", zorder=5)

for ep, rate in zip(epochs, success_rates):
    ax.annotate(f"{rate:.0f}%", (ep, rate), textcoords="offset points",
                xytext=(0, 14), ha="center", fontsize=11, fontweight="bold",
                color="#2a6cb8")

ax.axhspan(30, 60, color="#6BBF6B", alpha=0.12, label="RoboCasa benchmark range (30–60%)")
ax.axhline(y=30, color="#6BBF6B", linewidth=0.8, linestyle="--", alpha=0.5)
ax.axhline(y=60, color="#6BBF6B", linewidth=0.8, linestyle="--", alpha=0.5)

ax.set_xlabel("Training Epoch", fontsize=12)
ax.set_ylabel("Success Rate (%)", fontsize=12)
ax.set_title("Policy Success Rate Over Training Epochs\n(20 episodes, pretrain split)",
             fontsize=13, fontweight="bold")
ax.set_xticks(epochs)
ax.set_xlim(15, 110)
ax.set_ylim(0, 75)
ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax.grid(axis="y", alpha=0.3)
ax.legend(loc="upper left", fontsize=9)

fig_graph.tight_layout()
fig_graph.savefig("../figures/success_rate_over_epochs.png", dpi=150,
                  bbox_inches="tight", facecolor="white")

plt.show()
print("Saved ../figures/eval_results_table.png")
print("Saved ../figures/success_rate_over_epochs.png")
