import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r = json.load(open("results/results.json"))["summary"]
names = list(r); cols = ["#9aa0a6", "#3b6ea5", "#d9822b", "#2e7d32"]
fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.8))
for ax, (m, lab) in zip(axes, [("nDCG@10", "nDCG@10 (graded relevance)"), ("P@5", "Precision@5"),
                               ("distinct orgs @10", "Distinct organisations in top 10")]):
    v = [r[n][m] for n in names]
    ax.barh(names[::-1], v[::-1], color=cols[::-1])
    for i, x in enumerate(v[::-1]):
        ax.text(x * 1.01, i, f"{x:.2f}" if x < 2 else f"{x:.1f}", va="center", fontsize=8)
    ax.set_title(lab, fontsize=9, loc="left"); ax.set_xlim(0, max(v) * 1.2)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8)
    if ax is not axes[0]: ax.set_yticklabels([])
fig.suptitle("30 volunteer profiles, 658 NYC opportunities", x=0.01, ha="left", fontsize=9)
fig.tight_layout(); fig.savefig("results/results.png", dpi=160)
