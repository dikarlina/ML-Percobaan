import matplotlib.pyplot as plt
import numpy as np

# =========================
# DATA HASIL AKHIR
# =========================

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

polynomial = [80.36, 76.18, 57.43, 64.82]
sigmoid    = [79.65, 74.33, 58.59, 65.03]

# =========================
# BAR CHART
# =========================

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10,6))

bars1 = plt.bar(
    x - width/2,
    polynomial,
    width,
    label="Polynomial"
)

bars2 = plt.bar(
    x + width/2,
    sigmoid,
    width,
    label="Sigmoid"
)

# =========================
# LABEL
# =========================

plt.title(
    "Perbandingan Kinerja Kernel Polynomial dan Sigmoid",
    fontsize=14,
    fontweight="bold"
)

plt.ylabel("Persentase (%)")
plt.xticks(x, metrics)
plt.ylim(0, 100)

plt.legend()

# =========================
# VALUE DI ATAS BAR
# =========================

for bar in bars1:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.5,
        f"{height:.2f}",
        ha="center"
    )

for bar in bars2:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.5,
        f"{height:.2f}",
        ha="center"
    )

plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()

plt.savefig(
    "comparison_polynomial_vs_sigmoid.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()