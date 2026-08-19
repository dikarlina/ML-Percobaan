import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =========================
# 1. DATASET
# =========================
df = pd.read_csv("diabetes.csv")
print(df.info())
print(df.head())

# =========================
# 2. PREPROCESSING
# =========================
cols_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_zero] = df[cols_zero].replace(0, np.nan)
df = df.dropna().reset_index(drop=True)
print(f"Dataset setelah cleaning: {len(df)} data")

features = ["Glucose", "Insulin", "Age"]
X = df[features].values
y = df["Outcome"].values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

print(f"Fitur yang digunakan: {features}")
print(f"Distribusi kelas - Non-Diabetes (0): {np.sum(y==0)}, Diabetes (1): {np.sum(y==1)}")

# =========================
# 3. FUNGSI EVALUASI
# Kernel Sigmoid menggunakan gamma dan C.
# SOA mengoptimasi 2D: x[0] = gamma, x[1] = C
# coef0 dibiarkan default (0.0)
# =========================
def evaluate(X, y, gamma, C, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accs = []

    for train_idx, test_idx in kf.split(X):
        model = SVC(kernel="sigmoid", gamma=gamma, C=C, coef0=0.0)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))

    return np.mean(accs), np.std(accs), max(accs), accs

# =========================
# 4. FUNGSI EVALUASI LENGKAP
# =========================
def evaluate_full_metrics(X, y, gamma, C, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    accs, precs, recs, f1s = [], [], [], []
    cm_total = np.zeros((2, 2), dtype=int)

    for train_idx, test_idx in kf.split(X):
        model = SVC(kernel="sigmoid", gamma=gamma, C=C, coef0=0.0)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])

        accs.append(accuracy_score(y[test_idx], pred))
        precs.append(precision_score(y[test_idx], pred, pos_label=1, zero_division=0))
        recs.append(recall_score(y[test_idx], pred, pos_label=1, zero_division=0))
        f1s.append(f1_score(y[test_idx], pred, pos_label=1, zero_division=0))
        cm_total += confusion_matrix(y[test_idx], pred, labels=[0, 1])

    return {
        "acc":  np.mean(accs),
        "std":  np.std(accs),
        "prec": np.mean(precs),
        "rec":  np.mean(recs),
        "f1":   np.mean(f1s),
        "cm":   cm_total,
        "acc_per_fold": accs
    }

# =========================
# 5. SPIRAL OPTIMIZATION ALGORITHM (SOA)
# x[0] = gamma, x[1] = C  (keduanya harus positif)
# =========================
def SOA(X, y, theta, r, pop, k=6, iter_max=25):
    pop = np.array(pop, dtype=float)
    n_pop = len(pop)

    I = np.eye(2)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    def fitness(x):
        gamma = max(x[0], 1e-4)
        C     = max(x[1], 1e-4)
        acc, _, _, _ = evaluate(X, y, gamma=gamma, C=C, k=k)
        return acc

    scores_init = [fitness(p) for p in pop]
    best        = pop[np.argmax(scores_init)].copy()
    best_score  = max(scores_init)

    history_best    = [best.copy()]
    history_all_pop = [pop.copy()]

    for _ in range(iter_max):
        for i in range(n_pop):
            x     = pop[i]
            x_new = r * (R @ x) - (r * R - I) @ best
            # gamma dan C harus positif
            x_new = np.clip(x_new, 1e-4, 200)

            score = fitness(x_new)
            if score > best_score:
                best       = x_new.copy()
                best_score = score

            pop[i] = x_new

        history_best.append(best.copy())
        history_all_pop.append(pop.copy())

    best_gamma = max(best[0], 1e-4)
    best_C     = max(best[1], 1e-4)
    acc, std, _, _ = evaluate(X, y, gamma=best_gamma, C=best_C, k=k)

    return best_gamma, best_C, acc, std, history_best, history_all_pop

# =========================
# MODEL 1: PENCARIAN NILAI K TERBAIK
# =========================
print("\n" + "="*50)
print("MODEL 1: PENCARIAN NILAI K TERBAIK")
print("="*50)

k_vals     = list(range(2, 10))
k_accs     = []
k_highs    = []
k_stds     = []

best_k     = k_vals[0]
best_acc_k = 0

for k in k_vals:
    # default: gamma=1, C=1
    acc, std, high, _ = evaluate(X, y, gamma=1, C=1, k=k)
    k_accs.append(acc)
    k_highs.append(high)
    k_stds.append(std)
    print(f"  k = {k} | Avg Acc = {acc:.4f} | Highest = {high:.4f} | Std = {std:.4f}")

    if acc > best_acc_k:
        best_acc_k = acc
        best_k     = k

print(f"\n K Terbaik: k = {best_k} (Avg Accuracy = {best_acc_k:.4f})")

# ── Fig 1: Highest & Average Accuracy vs K ──
fig, ax = plt.subplots(figsize=(10, 5))
x_labels = [f"K{k}\n({std:.3f})" for k, std in zip(k_vals, k_stds)]
x_pos    = range(len(k_vals))

ax.plot(x_pos, [h*100 for h in k_highs], marker='o', color='blue',
        linewidth=2, markersize=8, label='highest accuracy')
ax.plot(x_pos, [a*100 for a in k_accs],  marker='o', color='green',
        linewidth=2, markersize=8, label='average accuracy')

for i, (h, a) in enumerate(zip(k_highs, k_accs)):
    ax.annotate(f"{h*100:.2f}", (i, h*100), textcoords="offset points",
                xytext=(0, 6), ha='center', fontsize=8, color='blue')
    ax.annotate(f"{a*100:.2f}", (i, a*100), textcoords="offset points",
                xytext=(0, -14), ha='center', fontsize=8, color='green')

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels, fontsize=9)
ax.set_xlabel("Parameter K (standard deviation)", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Graph of testing model 1 (Sigmoid Kernel)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_k_search.png", dpi=150)
plt.show()

# ── Fig 2: Spiral path untuk best k ──
_pop_init_k = [(1,10),(3,30),(5,50),(8,80),(10,100)]
_, _, _, _, _hb_k, _ha_k = SOA(
    X, y, theta=np.pi/4, r=0.5,
    pop=_pop_init_k,
    k=best_k
)
_harr_k   = np.array(_ha_k)
_hbest_k  = np.array(_hb_k)
_npop_k   = _harr_k.shape[1]
_colors_k = plt.cm.tab10(np.linspace(0, 1, _npop_k))

fig, ax = plt.subplots(figsize=(8, 6))
for p in range(_npop_k):
    traj = _harr_k[:, p, :]
    ax.plot(traj[:, 0], traj[:, 1], color=_colors_k[p], linewidth=1.5, alpha=0.8)
    gm0, c0 = _pop_init_k[p]
    ax.annotate(f"({gm0},{c0})", (traj[0, 0], traj[0, 1]),
                fontsize=8, ha='left', va='bottom', color=_colors_k[p])

bx, by = _hbest_k[-1, 0], _hbest_k[-1, 1]
ax.scatter(bx, by, color='black', s=80, zorder=10, marker='s')
ax.annotate(f"({bx:.8f}, {by:.8f})",
            (bx, by), textcoords="offset points", xytext=(5, 5),
            fontsize=8, color='black')

ax.set_xlabel("γ", fontsize=11)
ax.set_ylabel("C", fontsize=11)
ax.set_title("Spiral graph of the test model 1 (Sigmoid Kernel)", fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_k_spiral.png", dpi=150)
plt.show()


# =========================
# MODEL 2: PENCARIAN NILAI THETA TERBAIK
# =========================
print("\n" + "="*50)
print("MODEL 2: PENCARIAN NILAI THETA TERBAIK")
print("="*50)

theta_list   = [np.pi/2, np.pi/4, np.pi/8, np.pi/16, np.pi/32, np.pi/64, np.pi/128]
theta_labels = ["π/2", "π/4", "π/8", "π/16", "π/32", "π/64", "π/128"]

theta_accs   = []
theta_highs  = []
theta_stds   = []
theta_gammas = []
theta_Cs     = []
theta_histories_all = []

best_theta     = theta_list[0]
best_acc_theta = 0

default_pop = [(1,10),(2,20),(3,30),(4,40),(5,50),
               (6,60),(7,70),(8,80),(9,90),(10,100)]

for theta, label in zip(theta_list, theta_labels):
    gamma, C, acc, std, hb, ha = SOA(
        X, y, theta=theta, r=0.5, pop=default_pop, k=best_k
    )
    _, _, high, _ = evaluate(X, y, gamma=gamma, C=C, k=best_k)
    theta_accs.append(acc)
    theta_highs.append(high)
    theta_stds.append(std)
    theta_gammas.append(gamma)
    theta_Cs.append(C)
    theta_histories_all.append((hb, ha, gamma, C))
    print(f"  θ = {label:6s} | Avg={acc:.4f} | Highest={high:.4f} | Std={std:.4f} | γ={gamma:.4f}, C={C:.4f}")

    if acc > best_acc_theta:
        best_acc_theta = acc
        best_theta     = theta

best_theta_label = theta_labels[theta_list.index(best_theta)]
print(f"\n Theta Terbaik: θ = {best_theta_label} (Avg Accuracy = {best_acc_theta:.4f})")

# ── Fig 3: Highest & Average vs Theta ──
fig, ax = plt.subplots(figsize=(10, 5))
x_pos       = range(len(theta_list))
x_labels_th = [f"{lbl}\n({std:.3f})" for lbl, std in zip(theta_labels, theta_stds)]

ax.plot(x_pos, [h*100 for h in theta_highs], marker='o', color='blue',
        linewidth=2, markersize=8, label='highest accuracy')
ax.plot(x_pos, [a*100 for a in theta_accs],  marker='o', color='green',
        linewidth=2, markersize=8, label='average accuracy')

for i, (h, a) in enumerate(zip(theta_highs, theta_accs)):
    ax.annotate(f"{h*100:.2f}", (i, h*100), textcoords="offset points",
                xytext=(0, 6), ha='center', fontsize=8, color='blue')
    ax.annotate(f"{a*100:.2f}", (i, a*100), textcoords="offset points",
                xytext=(0, -14), ha='center', fontsize=8, color='green')

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels_th, fontsize=9)
ax.set_xlabel("Parameter θ (standard deviation)", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Graph of testing model 2 (Sigmoid Kernel)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_theta_search.png", dpi=150)
plt.show()

# ── Fig 4: Spiral path untuk best theta ──
best_th_idx              = theta_list.index(best_theta)
_hb_th, _ha_th, _, _    = theta_histories_all[best_th_idx]
_harr_th  = np.array(_ha_th)
_hbarr_th = np.array(_hb_th)
_npop_th  = _harr_th.shape[1]
_cols_th  = plt.cm.tab10(np.linspace(0, 1, _npop_th))

fig, ax = plt.subplots(figsize=(8, 6))
for p in range(_npop_th):
    traj = _harr_th[:, p, :]
    ax.plot(traj[:, 0], traj[:, 1], color=_cols_th[p], linewidth=1.5, alpha=0.8)
    gm0, c0 = default_pop[p]
    ax.annotate(f"({gm0},{c0})", (traj[0, 0], traj[0, 1]),
                fontsize=7, ha='left', va='bottom', color=_cols_th[p])

bx, by = _hbarr_th[-1, 0], _hbarr_th[-1, 1]
ax.scatter(bx, by, color='black', s=80, zorder=10, marker='s')
ax.annotate(f"({bx:.8f}, {by:.8f})",
            (bx, by), textcoords="offset points", xytext=(5, 5),
            fontsize=8, color='black')

ax.set_xlabel("γ", fontsize=11)
ax.set_ylabel("C", fontsize=11)
ax.set_title("Spiral graph of the test model 2 (Sigmoid Kernel)", fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_theta_spiral.png", dpi=150)
plt.show()

# =========================
# MODEL 3: PENCARIAN NILAI R TERBAIK
# =========================
print("\n" + "="*50)
print("MODEL 3: PENCARIAN NILAI R TERBAIK")
print("="*50)

r_vals      = np.round(np.arange(0.1, 1.0, 0.1), 1)
r_accs      = []
r_highs     = []
r_stds      = []
r_histories = []
r_gammas    = []
r_Cs        = []

best_r     = r_vals[0]
best_acc_r = 0

for r in r_vals:
    gamma, C, acc, std, hb, ha = SOA(
        X, y, theta=best_theta, r=r, pop=default_pop, k=best_k
    )
    _, _, high, _ = evaluate(X, y, gamma=gamma, C=C, k=best_k)
    r_accs.append(acc)
    r_highs.append(high)
    r_stds.append(std)
    r_gammas.append(gamma)
    r_Cs.append(C)
    r_histories.append((hb, ha, gamma, C))
    print(f"  r = {r:.1f} | Avg={acc:.4f} | Highest={high:.4f} | Std={std:.4f} | γ={gamma:.4f}, C={C:.4f}")

    if acc > best_acc_r:
        best_acc_r = acc
        best_r     = r

print(f"\n R Terbaik: r = {best_r:.1f} (Avg Accuracy = {best_acc_r:.4f})")

# ── Fig 5: Highest & Average vs R ──
fig, ax = plt.subplots(figsize=(10, 5))
x_pos_r    = range(len(r_vals))
x_labels_r = [f"r={r:.1f}\n({std:.3f})" for r, std in zip(r_vals, r_stds)]

ax.plot(x_pos_r, [h*100 for h in r_highs], marker='o', color='blue',
        linewidth=2, markersize=8, label='highest accuracy')
ax.plot(x_pos_r, [a*100 for a in r_accs],  marker='o', color='green',
        linewidth=2, markersize=8, label='average accuracy')

for i, (h, a) in enumerate(zip(r_highs, r_accs)):
    ax.annotate(f"{h*100:.2f}", (i, h*100), textcoords="offset points",
                xytext=(0, 6), ha='center', fontsize=8, color='blue')
    ax.annotate(f"{a*100:.2f}", (i, a*100), textcoords="offset points",
                xytext=(0, -14), ha='center', fontsize=8, color='green')

ax.set_xticks(x_pos_r)
ax.set_xticklabels(x_labels_r, fontsize=9)
ax.set_xlabel("Parameter r (standard deviation)", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Graph of testing model 3 (Sigmoid Kernel)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_r_search.png", dpi=150)
plt.show()

# ── Fig 6: Spiral path untuk best r ──
best_r_idx             = list(r_vals).index(best_r)
_hb_r, _ha_r, _, _    = r_histories[best_r_idx]
_harr_r  = np.array(_ha_r)
_hbarr_r = np.array(_hb_r)
_npop_r  = _harr_r.shape[1]
_cols_r  = plt.cm.tab10(np.linspace(0, 1, _npop_r))

fig, ax = plt.subplots(figsize=(8, 6))
for p in range(_npop_r):
    traj = _harr_r[:, p, :]
    ax.plot(traj[:, 0], traj[:, 1], color=_cols_r[p], linewidth=1.5, alpha=0.8)
    gm0, c0 = default_pop[p]
    ax.annotate(f"({gm0},{c0})", (traj[0, 0], traj[0, 1]),
                fontsize=7, ha='left', va='bottom', color=_cols_r[p])

bx, by = _hbarr_r[-1, 0], _hbarr_r[-1, 1]
ax.scatter(bx, by, color='black', s=80, zorder=10, marker='s')
ax.annotate(f"({bx:.8f}, {by:.8f})",
            (bx, by), textcoords="offset points", xytext=(5, 5),
            fontsize=8, color='black')

ax.set_xlabel("γ", fontsize=11)
ax.set_ylabel("C", fontsize=11)
ax.set_title("Spiral graph of the test model 3 (Sigmoid Kernel)", fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_r_spiral.png", dpi=150)
plt.show()

# =========================
# MODEL 4: PENCARIAN POPULASI TERBAIK
# =========================
print("\n" + "="*50)
print("MODEL 4: PENCARIAN POPULASI TERBAIK (M1–M9)")
print("="*50)

# Populasi berisi (gamma, C) — keduanya dioptimasi
pop_models = {
    "M1": [(1,10), (10,100)],
    "M2": [(1,10), (5,50), (10,100)],
    "M3": [(1,10), (4,40), (6,60), (10,100)],
    "M4": [(1,10), (3,30), (5,50), (8,80), (10,100)],
    "M5": [(1,10), (2,20), (4,40), (6,60), (8,80), (10,100)],
    "M6": [(1,10), (2,20), (4,40), (5,50), (6,60), (8,80), (10,100)],
    "M7": [(1,10), (2,20), (3,30), (4,40), (6,60), (8,80), (9,90), (10,100)],
    "M8": [(1,10), (2,20), (3,30), (4,40), (5,50), (7,70), (8,80), (9,90), (10,100)],
    "M9": [(1,10), (2,20), (3,30), (4,40), (5,50), (6,60), (7,70), (8,80), (9,90), (10,100)]
}

model_names  = []
model_accs   = []
model_highs  = []
model_stds   = []
model_gammas = []
model_Cs     = []
model_histories = []

best_model_name      = None
best_acc_model       = 0
best_gamma_final     = None
best_C_final         = None
best_history         = None
best_history_all_pop = None
best_pop_used        = None

for name, pop in pop_models.items():
    gamma, C, acc, std, history_best, history_all_pop = SOA(
        X, y, theta=best_theta, r=best_r, pop=pop, k=best_k
    )
    _, _, high, _ = evaluate(X, y, gamma=gamma, C=C, k=best_k)
    model_names.append(name)
    model_accs.append(acc)
    model_highs.append(high)
    model_stds.append(std)
    model_gammas.append(gamma)
    model_Cs.append(C)
    model_histories.append((history_best, history_all_pop, pop, gamma, C))

    print(f"  {name} | Avg={acc:.4f} | Highest={high:.4f} | Std={std:.4f} | γ={gamma:.4f}, C={C:.4f} | pop={len(pop)}")

    if acc > best_acc_model:
        best_acc_model       = acc
        best_model_name      = name
        best_gamma_final     = gamma
        best_C_final         = C
        best_history         = history_best
        best_history_all_pop = history_all_pop
        best_pop_used        = pop

print(f"\n Model Terbaik: {best_model_name} (Avg Accuracy = {best_acc_model:.4f})")
print(f"    Gamma = {best_gamma_final:.4f}, C = {best_C_final:.4f}")

# ── Fig 7: Highest & Average vs Population Model ──
fig, ax = plt.subplots(figsize=(11, 5))
x_pos_m    = range(len(model_names))
x_labels_m = [f"{n}\n({std:.3f})" for n, std in zip(model_names, model_stds)]

ax.plot(x_pos_m, [h*100 for h in model_highs], marker='o', color='blue',
        linewidth=2, markersize=8, label='highest accuracy')
ax.plot(x_pos_m, [a*100 for a in model_accs],  marker='o', color='green',
        linewidth=2, markersize=8, label='average accuracy')

for i, (h, a) in enumerate(zip(model_highs, model_accs)):
    ax.annotate(f"{h*100:.2f}", (i, h*100), textcoords="offset points",
                xytext=(0, 6), ha='center', fontsize=8, color='blue')
    ax.annotate(f"{a*100:.2f}", (i, a*100), textcoords="offset points",
                xytext=(0, -14), ha='center', fontsize=8, color='green')

ax.set_xticks(x_pos_m)
ax.set_xticklabels(x_labels_m, fontsize=9)
ax.set_xlabel("Model Populasi (standard deviation)", fontsize=11)
ax.set_ylabel("Accuracy", fontsize=11)
ax.set_title("Graph of testing model 4 (Sigmoid Kernel)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_population_search.png", dpi=150)
plt.show()

# ── Fig 8: Spiral path untuk best population model ──
_harr_m  = np.array(best_history_all_pop)
_hbarr_m = np.array(best_history)
_npop_m  = _harr_m.shape[1]
_cols_m  = plt.cm.tab10(np.linspace(0, 1, _npop_m))

fig, ax = plt.subplots(figsize=(9, 6))
for p in range(_npop_m):
    traj = _harr_m[:, p, :]
    ax.plot(traj[:, 0], traj[:, 1], color=_cols_m[p], linewidth=1.5, alpha=0.8)
    gm0, c0 = best_pop_used[p]
    ax.annotate(f"({gm0},{c0})", (traj[0, 0], traj[0, 1]),
                fontsize=8, ha='left', va='bottom', color=_cols_m[p])

bx, by = _hbarr_m[-1, 0], _hbarr_m[-1, 1]
ax.scatter(bx, by, color='black', s=80, zorder=10, marker='s')
ax.annotate(f"({bx:.8f}, {by:.8f})",
            (bx, by), textcoords="offset points", xytext=(5, 5),
            fontsize=8, color='black')

ax.set_xlabel("γ", fontsize=11)
ax.set_ylabel("C", fontsize=11)
ax.set_title(f"Spiral graph of the test model 4 ({best_model_name}) – Sigmoid Kernel", fontsize=13)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("plot_population_spiral.png", dpi=150)
plt.show()

# =========================
# EVALUASI MODEL FINAL (Full Metrics)
# =========================
print("\n" + "="*50)
print("EVALUASI MODEL FINAL")
print("="*50)

final = evaluate_full_metrics(X, y, best_gamma_final, best_C_final, best_k)

# Visualisasi Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = final["cm"]
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Diabetes (0)", "Diabetes (1)"],
            yticklabels=["Non-Diabetes (0)", "Diabetes (1)"],
            ax=ax, linewidths=0.5, linecolor='gray')
ax.set_xlabel("Prediksi", fontsize=12)
ax.set_ylabel("Aktual", fontsize=12)
ax.set_title(
    f"Confusion Matrix – SVM Sigmoid + SOA\n"
    f"(k={best_k}, θ={best_theta_label}, r={best_r:.1f}, Model={best_model_name})",
    fontsize=12
)

tn, fp, fn, tp = cm.ravel()
ax.text(0.5, -0.15, f"TN={tn}  FP={fp}  FN={fn}  TP={tp}",
        transform=ax.transAxes, ha='center', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig("plot_confusion_matrix.png", dpi=150)
plt.show()

# =========================
# TABEL REKAPITULASI HASIL
# =========================
print("\n" + "="*60)
print("REKAPITULASI PARAMETER DAN HASIL TERBAIK")
print("="*60)

print(f"\n  Parameter Terbaik:")
print(f"  {'Kernel':<25} = Sigmoid (coef0=0.0)")
print(f"  {'k (K-Fold)':<25} = {best_k}")
print(f"  {'θ (Sudut Spiral)':<25} = {best_theta_label}")
print(f"  {'r (Radius Spiral)':<25} = {best_r:.1f}")
print(f"  {'Model Populasi':<25} = {best_model_name}")
print(f"  {'Gamma (γ)':<25} = {best_gamma_final:.6f}")
print(f"  {'C (Cost)':<25} = {best_C_final:.6f}")

print(f"\n  Performa Model (K-Fold CV, k={best_k}):")
print(f"  {'Accuracy':<25} = {final['acc']:.4f} ({final['acc']*100:.2f}%)")
print(f"  {'Standard Deviation':<25} = {final['std']:.4f}")
print(f"  {'Precision':<25} = {final['prec']:.4f} ({final['prec']*100:.2f}%)")
print(f"  {'Recall':<25} = {final['rec']:.4f} ({final['rec']*100:.2f}%)")
print(f"  {'F1-Score':<25} = {final['f1']:.4f} ({final['f1']*100:.2f}%)")

print(f"\n  Confusion Matrix:")
print(f"  {'':>20} Prediksi 0   Prediksi 1")
print(f"  {'Aktual 0':>20}  {cm[0,0]:>8}     {cm[0,1]:>8}   (TN={cm[0,0]}, FP={cm[0,1]})")
print(f"  {'Aktual 1':>20}  {cm[1,0]:>8}     {cm[1,1]:>8}   (FN={cm[1,0]}, TP={cm[1,1]})")

print(f"\n  Accuracy per Fold:")
for i, acc_fold in enumerate(final["acc_per_fold"], 1):
    print(f"    Fold {i}: {acc_fold:.4f}")

print("\n" + "="*60)
print("SELESAI")
print("="*60)


print("Ayo Coba Lagi")
print("Bismillah Lancar Magang")