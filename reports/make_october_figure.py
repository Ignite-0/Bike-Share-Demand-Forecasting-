"""MAE by hour of day in October (Oct 15-28 forecasts): the blend vs both baselines.

    python reports/make_october_figure.py   # reads reports/october_results.csv
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SRC = "reports/october_results.csv"
OUT = "reports/figures/october_mae_by_hour.png"
SERIES = [  # column, label, color, line style
    ("mae", "Model (blend)", "#2a78d6", "-"),
    ("mae_published", "Published profile", "#eb6834", "--"),
    ("mae_fresh", "Same hour last week", "#1baf7a", ":"),
]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

df = pd.read_csv(SRC)
df = df[df["dim"] == "hour"].assign(hour=lambda d: d["group"].astype(int))

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor="#fcfcfb")
for ax, (target, title) in zip(axes, [("pickups", "Pick-ups"), ("dropoffs", "Drop-offs")]):
    d = df[df["target"] == target].sort_values("hour")
    for col, label, color, ls in SERIES:
        ax.plot(d["hour"], d[col], color=color, ls=ls, lw=2, label=label)
    ax.set_facecolor("#fcfcfb")
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlim(0, 23)
    ax.set_xlabel("Hour of day (New York time)", color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
axes[0].set_ylabel("MAE (trips per station-hour)", color=MUTED)
axes[0].set_ylim(bottom=0)
axes[0].legend(frameon=False, loc="upper left", labelcolor=INK)
fig.suptitle("October 2026, out of sample: mean absolute error by hour", x=0.01, ha="left", color=INK)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(OUT)
