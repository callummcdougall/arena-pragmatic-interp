import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# --- Data (approximate from the chart) ---
models = [
    "Claude\nSonnet 3.7",
    "Claude\nSonnet 4",
    "Claude\nOpus 4.1",
    "Grok-4",
    "Gemini\n2.5 Pro",
    "o4 mini",
    "GPT-5",
]

# Each row: [Original, Without Chekhov's Gun, Without leadership complicity, Without agency, Without org wrongdoing]
means = np.array(
    [
        [0.60, 0.43, 0.34, 0.17, 0.20],
        [0.65, 0.65, 0.49, 0.26, 0.37],
        [0.77, 0.58, 0.57, 0.38, 0.26],
        [0.76, 0.71, 0.46, 0.30, 0.41],
        [0.77, 0.70, 0.48, 0.27, 0.40],
        [0.52, 0.47, 0.46, 0.23, 0.33],
        [0.25, 0.30, 0.17, 0.05, 0.10],
    ]
)

# CI upper bounds (light background bar tops)
uppers = np.array(
    [
        [0.75, 0.57, 0.47, 0.28, 0.30],
        [0.80, 0.78, 0.65, 0.38, 0.55],
        [0.90, 0.72, 0.73, 0.52, 0.40],
        [0.90, 0.85, 0.58, 0.42, 0.59],
        [0.87, 0.78, 0.65, 0.40, 0.55],
        [0.65, 0.64, 0.58, 0.35, 0.45],
        [0.42, 0.42, 0.30, 0.15, 0.22],
    ]
)

# CI lower bounds
lowers = np.array(
    [
        [0.45, 0.28, 0.22, 0.07, 0.10],
        [0.50, 0.52, 0.33, 0.13, 0.18],
        [0.63, 0.44, 0.42, 0.25, 0.12],
        [0.62, 0.58, 0.33, 0.18, 0.24],
        [0.67, 0.58, 0.32, 0.14, 0.25],
        [0.40, 0.32, 0.34, 0.12, 0.20],
        [0.10, 0.18, 0.05, 0.00, 0.02],
    ]
)

# --- Style config ---
condition_labels = [
    "Original",
    "Without Chekhov's Gun",
    "Without leadership complicity",
    "Without agency",
    "Without organizational\nwrongdoing",
]

# Colors for the solid bars
colors_bar = ["#6B6B6B", "#8B7BB8", "#E8A838", "#2D8A4E", "#2B5F94"]
# Colors for the light CI background bars
colors_ci = ["#E8E4DC", "#E0DBE8", "#F5E6C8", "#C8E0D0", "#C8D8E8"]

n_models = len(models)
n_conditions = len(condition_labels)
bar_width = 0.14

fig, ax = plt.subplots(figsize=(16, 8))

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for i in range(n_models):
    group_center = i
    offsets = np.arange(n_conditions) - (n_conditions - 1) / 2
    for j in range(n_conditions):
        x = group_center + offsets[j] * bar_width

        # 1) Wide light grey bar showing CI range (lower to upper)
        ci_bar_width = bar_width * 0.65
        ax.bar(
            x,
            uppers[i, j] - lowers[i, j],
            bottom=lowers[i, j],
            width=ci_bar_width,
            color="#EDEDED",
            edgecolor="none",
            zorder=1,
        )

        # 2) Thin solid bar from 0 up to the mean (lollipop stem)
        ax.plot(
            [x, x],
            [0, means[i, j]],
            color=colors_bar[j],
            linewidth=3.5,
            solid_capstyle="butt",
            zorder=2,
        )

        # 3) Circle at the top of the bar (at the mean)
        ax.plot(
            x,
            means[i, j],
            "o",
            color=colors_bar[j],
            markersize=7,
            zorder=3,
        )

        # 4) Horizontal ticks at CI upper and lower bounds
        tick_hw = ci_bar_width / 2
        for bound in [lowers[i, j], uppers[i, j]]:
            ax.plot(
                [x - tick_hw, x + tick_hw],
                [bound, bound],
                color=colors_bar[j],
                linewidth=2.5,
                zorder=3,
            )

# Axes formatting
ax.set_xticks(range(n_models))
ax.set_xticklabels(models, fontsize=12, fontfamily="sans-serif")
ax.set_ylabel("Mean Score", fontsize=14, fontfamily="sans-serif")
ax.set_ylim(0, 1.0)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
ax.tick_params(axis="y", labelsize=12)

for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(axis="x", length=0)
ax.tick_params(axis="y", length=0)

ax.set_axisbelow(True)

fig.suptitle(
    "Whistleblowing ablations",
    fontsize=22,
    fontweight="bold",
    fontfamily="sans-serif",
    y=0.97,
)

handles = [
    mpatches.Patch(facecolor=colors_ci[i], edgecolor=colors_bar[i], linewidth=1.5, label=condition_labels[i])
    for i in range(n_conditions)
]
ax.legend(
    handles=handles,
    loc="upper right",
    fontsize=11,
    frameon=True,
    facecolor="white",
    edgecolor="#CCCCCC",
    framealpha=1,
    borderpad=0.8,
)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(
    "/mnt/user-data/outputs/whistleblowing_ablations.png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor()
)
plt.show()
print("Done!")
