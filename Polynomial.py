import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

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

X = MinMaxScaler().fit_transform(X)
DEGREE = 4
COEF0 = 1
print(f"Fitur yang digunakan: {features}")
print(f"Distribusi kelas - Non-Diabetes (0): {np.sum(y==0)}, Diabetes (1): {np.sum(y==1)}")

# =========================
# 3. K-FOLD EVALUATION (accuracy only -> dipakai sebagai fitness SOA & pencarian parameter)
# =========================
def evaluate(X, y, gamma, C, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    accs = []

    for train, test in kf.split(X):
        model = SVC(
            kernel="poly",
            degree=DEGREE,
            coef0=COEF0,
            gamma=gamma,
            C=C
)
        model.fit(X[train], y[train])
        pred = model.predict(X[test])
        accs.append(accuracy_score(y[test], pred))

    return (np.mean(accs), np.std(accs), max(accs), accs)


# =========================
# 3b. K-FOLD EVALUATION LENGKAP (Accuracy, Precision, Recall, F1-Score)
# =========================
def evaluate_full_metrics(X, y, gamma, C, k):
    """
    Sama seperti evaluate(), tapi sekaligus menghitung precision, recall,
    f1-score, dan confusion matrix gabungan dari seluruh fold.
    Dipakai untuk laporan evaluasi akhir (bukan untuk proses pencarian
    parameter, supaya proses pencarian k/theta/r/population tetap cepat).
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    acc_list, prec_list, rec_list, f1_list = [], [], [], []
    acc_per_fold = []
    cm_total = np.zeros((2, 2), dtype=int)

    for train, test in kf.split(X):
        model = SVC(
            kernel="poly",
            degree=DEGREE,
            coef0=COEF0,
            gamma=gamma,
            C=C
        )
        model.fit(X[train], y[train])
        pred = model.predict(X[test])

        acc_list.append(accuracy_score(y[test], pred))
        acc_per_fold.append(
            accuracy_score(y[test], pred)
            )
        # pos_label=1 -> kelas "diabetes positif" dianggap kelas positif
        prec_list.append(precision_score(y[test], pred, pos_label=1, zero_division=0))
        rec_list.append(recall_score(y[test], pred, pos_label=1, zero_division=0))
        f1_list.append(f1_score(y[test], pred, pos_label=1, zero_division=0))

        cm_total += confusion_matrix(y[test], pred, labels=[0, 1])

    return {
        "accuracy_mean": np.mean(acc_list),
        "accuracy_std": np.std(acc_list),
        "precision_mean": np.mean(prec_list),
        "precision_std": np.std(prec_list),
        "recall_mean": np.mean(rec_list),
        "recall_std": np.std(rec_list),
        "f1_mean": np.mean(f1_list),
        "f1_std": np.std(f1_list),
        "confusion_matrix": cm_total,
        "acc_per_fold": acc_per_fold,
    }


# =========================
# 4. SPIRAL OPTIMIZATION
# =========================
def SOA(X, y, theta, r, pop, k=6, iter_max=25):

    pop = np.array(pop, dtype=float)

    I = np.eye(2)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    def fitness(x):
        acc, _, _, _ = evaluate(X, y, x[0], x[1], k)
        return acc

    best = pop[0]
    best_score = fitness(best)
    history_best = [best.copy()]
    history_all_pop = [pop.copy()]

    for _ in range(iter_max):
        for i in range(len(pop)):

            x = pop[i]

            x_new = r * (R @ x) - (r * R - I) @ best
            x_new = np.clip(x_new, 0.0001, 200)

            score = fitness(x_new)

            if score > best_score:
                best = x_new
                best_score = score

            pop[i] = x_new
        history_best.append(best.copy())
        history_all_pop.append(pop.copy())

    acc, std, _, _ = evaluate(X, y, best[0], best[1], k)

    return best, acc, std, history_best, history_all_pop

def visualize_spiral(history_best, history_all_pop, initial_pop):

    harr = np.array(history_all_pop)
    hbest = np.array(history_best)

    n_pop = harr.shape[1]
    colors = plt.cm.tab10(np.linspace(0,1,n_pop))

    plt.figure(figsize=(8,6))

    for p in range(n_pop):

        traj = harr[:,p,:]

        plt.plot(
            traj[:,0],
            traj[:,1],
            color=colors[p],
            linewidth=1.5
        )

        gamma0, c0 = initial_pop[p]

        plt.annotate(
            f"({gamma0},{c0})",
            (traj[0,0], traj[0,1]),
            fontsize=8
        )

    bx, by = hbest[-1]

    plt.scatter(
        bx,
        by,
        color='black',
        s=100,
        marker='s'
    )

    plt.annotate(
        f"Best ({bx:.4f}, {by:.4f})",
        (bx,by)
    )

    plt.xlabel("Gamma")
    plt.ylabel("C")
    plt.title("Spiral Optimization Trajectory")
    plt.grid(True)

    plt.show()

# =========================
# 5. MODEL 1 - K SEARCH (2-9)
# =========================
best_k, best_acc = 0, 0
k_accs = []
k_highs = []
k_stds = []

for k in range(2, 10):
    acc, std, high, _ = evaluate(
    X,
    y,
    1,
    1,
    k
)
    print("k =", k, "acc =", acc)

    if acc > best_acc:
        best_k, best_acc = k, acc

    k_accs.append(acc)
    k_highs.append(high)
    k_stds.append(std)

print("\nBEST k:", best_k)
fig, ax = plt.subplots(figsize=(10,5))

x_labels = [
    f"K{k}\n({std:.3f})"
    for k,std in zip(
        range(2,10),
        k_stds
    )
]

x_pos = range(len(k_accs))

ax.plot(
    x_pos,
    [h*100 for h in k_highs],
    marker='o',
    color='blue',
    label='highest accuracy'
)

ax.plot(
    x_pos,
    [a*100 for a in k_accs],
    marker='o',
    color='green',
    label='average accuracy'
)
for i,(h,a) in enumerate(zip(k_highs, k_accs)):

    ax.annotate(
        f"{h*100:.2f}",
        (i,h*100),
        textcoords="offset points",
        xytext=(0,6),
        ha='center'
    )

    ax.annotate(
        f"{a*100:.2f}",
        (i,a*100),
        textcoords="offset points",
        xytext=(0,-14),
        ha='center'
    )
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)

ax.set_xlabel(
    "Parameter K (standard deviation)"
)

ax.set_ylabel("Accuracy")
ax.set_title(
    "Graph of testing model 1"
)

ax.legend()
ax.grid(True)

plt.show()
# =========================
# SPIRAL K
# =========================
_, _, _, hb_k, ha_k = SOA(
    X,
    y,
    theta=np.pi/4,
    r=0.5,
    pop=[(1,10),(3,30),(5,50),(8,80),(10,100)],
    k=best_k
)

visualize_spiral(
    hb_k,
    ha_k,
    [(1,10),(3,30),(5,50),(8,80),(10,100)]
)
# =========================
# 5a. DEGREE SEARCH
# =========================

degree_list = [2, 3, 4, 5]
degree_accs = []
best_degree = 0
best_acc_degree = 0

print("\n====================")
print("DEGREE SEARCH")
print("====================")
degree_highs = []
degree_stds = []
for degree in degree_list:

    kf = KFold(
        n_splits=best_k,
        shuffle=True,
        random_state=42
    )

    accs = []

    for train, test in kf.split(X):

        model = SVC(
            kernel="poly",
            degree=degree,
            coef0=1,
            gamma=1,
            C=1
        )

        model.fit(X[train], y[train])

        pred = model.predict(X[test])

        accs.append(
            accuracy_score(y[test], pred)
        )

    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    high_acc = max(accs)
    degree_highs.append(high_acc)
    degree_stds.append(std_acc)
    degree_accs.append(mean_acc)

    print(
        f"Degree={degree} | "
        f"Mean={mean_acc:.4f} | "
        f"Std={std_acc:.4f}"
    )

    if mean_acc > best_acc_degree:
        best_degree = degree
        best_acc_degree = mean_acc

print("\nBEST DEGREE =", best_degree)
fig, ax = plt.subplots(figsize=(10,5))

x_pos = range(len(degree_list))

x_labels = [
    f"D{d}\n({std:.3f})"
    for d,std in zip(
        degree_list,
        degree_stds
    )
]

ax.plot(
    x_pos,
    [h*100 for h in degree_highs],
    marker='o',
    color='blue',
    label='highest accuracy'
)

ax.plot(
    x_pos,
    [a*100 for a in degree_accs],
    marker='o',
    color='green',
    label='average accuracy'
)
for i, (h, a) in enumerate(zip(degree_highs, degree_accs)):

    ax.annotate(
        f"{h*100:.2f}",
        (i, h*100),
        textcoords="offset points",
        xytext=(0, 6),
        ha='center',
        fontsize=8,
        color='blue'
    )

    ax.annotate(
        f"{a*100:.2f}",
        (i, a*100),
        textcoords="offset points",
        xytext=(0, -14),
        ha='center',
        fontsize=8,
        color='green'
    )

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)

ax.set_xlabel(
    "Degree (standard deviation)"
)

ax.set_ylabel("Accuracy")
ax.set_title(
    "Graph of degree testing"
)

ax.legend()
ax.grid(True)

plt.show()

DEGREE = best_degree
COEF0 = 1

# =========================
# 6. MODEL 2 - THETA
# =========================
theta_list = [np.pi/2, np.pi/4, np.pi/8, np.pi/16, np.pi/32, np.pi/64, np.pi/128]

best_theta, best_acc = 0, 0
theta_accs = []
theta_highs = []
theta_stds = []
theta_histories = []

for theta in theta_list:

    best, acc, std, hb, ha = SOA(
        X, y,
        theta=theta,
        r=0.5,
        pop=[(1,10),(3,30),(5,50),(8,80),(10,100)],
        k=best_k
    )
    _, _, high, _ = evaluate(
    X,
    y,
    best[0],
    best[1],
    best_k
    )
    print("theta =", theta, "acc =", acc)
    theta_accs.append(acc)
    theta_highs.append(high)
    theta_stds.append(std)
    theta_histories.append((hb, ha))
    if acc > best_acc:
        best_theta, best_acc = theta, acc

print("\nBEST theta:", best_theta)
labels = [
    "π/2",
    "π/4",
    "π/8",
    "π/16",
    "π/32",
    "π/64",
    "π/128"
]

fig, ax = plt.subplots(figsize=(10,5))

x_pos = range(len(theta_list))

x_labels = [
    f"{lbl}\n({std:.3f})"
    for lbl,std in zip(
        labels,
        theta_stds
    )
]

ax.plot(
    x_pos,
    [h*100 for h in theta_highs],
    marker='o',
    color='blue',
    linewidth=2,
    label='highest accuracy'
)

ax.plot(
    x_pos,
    [a*100 for a in theta_accs],
    marker='o',
    color='green',
    linewidth=2,
    label='average accuracy'
)
for i, (h, a) in enumerate(zip(theta_highs, theta_accs)):

    ax.annotate(
        f"{h*100:.2f}",
        (i, h*100),
        textcoords="offset points",
        xytext=(0, 6),
        ha='center',
        fontsize=8,
        color='blue'
    )

    ax.annotate(
        f"{a*100:.2f}",
        (i, a*100),
        textcoords="offset points",
        xytext=(0, -14),
        ha='center',
        fontsize=8,
        color='green'
    )

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)

ax.set_xlabel(
    "Theta (standard deviation)"
)

ax.set_ylabel("Accuracy")

ax.set_title(
    "Graph of testing model 2"
)

ax.legend()
ax.grid(True)

plt.show()
best_theta_idx = theta_accs.index(max(theta_accs))

hb_th, ha_th = theta_histories[best_theta_idx]

visualize_spiral(
    hb_th,
    ha_th,
    [(1,10),(3,30),(5,50),(8,80),(10,100)]
)

# =========================
# 7. MODEL 3 - r
# =========================
best_r, best_acc = 0, 0
r_values = []
r_accs = []
r_highs = []
r_stds = []
r_histories = []

for r in np.arange(0.1, 1.0, 0.1):

    best, acc, std, hb, ha = SOA(
        X, y,
        theta=best_theta,
        r=r,
        pop=[(1,10),(3,30),(5,50),(8,80),(10,100)],
        k=best_k
    )
    _, _, high, _ = evaluate(
    X,
    y,
    best[0],
    best[1],
    best_k
    )

    print("r =", r, "acc =", acc)
    r_values.append(r)
    r_accs.append(acc)
    r_highs.append(high)
    r_stds.append(std)
    r_histories.append((hb, ha))

    if acc > best_acc:
        best_r, best_acc = r, acc

print("\nBEST r:", best_r)
fig, ax = plt.subplots(figsize=(10,5))

x_pos = range(len(r_values))

x_labels = [
    f"r={r:.1f}\n({std:.3f})"
    for r,std in zip(
        r_values,
        r_stds
    )
]

ax.plot(
    x_pos,
    [h*100 for h in r_highs],
    marker='o',
    color='blue',
    linewidth=2,
    label='highest accuracy'
)

ax.plot(
    x_pos,
    [a*100 for a in r_accs],
    marker='o',
    color='green',
    linewidth=2,
    label='average accuracy'
)
for i, (h, a) in enumerate(zip(r_highs, r_accs)):

    ax.annotate(
        f"{h*100:.2f}",
        (i, h*100),
        textcoords="offset points",
        xytext=(0, 6),
        ha='center',
        fontsize=8,
        color='blue'
    )

    ax.annotate(
        f"{a*100:.2f}",
        (i, a*100),
        textcoords="offset points",
        xytext=(0, -14),
        ha='center',
        fontsize=8,
        color='green'
    )

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)

ax.set_xlabel(
    "Parameter r (standard deviation)"
)

ax.set_ylabel("Accuracy")

ax.set_title(
    "Graph of testing model 3"
)

ax.legend()
ax.grid(True)

plt.show()
best_r_idx = r_accs.index(max(r_accs))

hb_r, ha_r = r_histories[best_r_idx]

visualize_spiral(
    hb_r,
    ha_r,
    [(1,10),(3,30),(5,50),(8,80),(10,100)]
)

# =========================
# 8. MODEL 4 - POPULATION
# =========================
pop_models = {
    "M1": [(1,10),(10,100)],
    "M2": [(1,10),(5,50),(10,100)],
    "M3": [(1,10),(4,40),(6,60),(10,100)],
    "M4": [(1,10),(3,30),(5,50),(8,80),(10,100)],
    "M5": [(1,10),(2,20),(4,40),(6,60),(8,80),(10,100)],
    "M6": [(1,10),(2,20),(4,40),(5,50),(6,60),(8,80),(10,100)],
    "M7": [(1,10),(2,20),(3,30),(4,40),(6,60),(8,80),(9,90),(10,100)],
    "M8": [(1,10),(2,20),(3,30),(4,40),(5,50),(7,70),(8,80),(9,90),(10,100)],
    "M9": [(1,10),(2,20),(3,30),(4,40),(5,50),(6,60),(7,70),(8,80),(9,90),(10,100)]
}

pop_names = []
pop_accs = []
pop_highs = []
pop_stds = []

best_model = None
best_acc = 0

# menyimpan spiral terbaik tiap model
all_best_histories = {}

for name, pop in pop_models.items():

    best, acc, std, history_best, history_all_pop = SOA(
        X,
        y,
        theta=best_theta,
        r=best_r,
        pop=pop,
        k=best_k
    )

    _, _, high, _ = evaluate(
        X,
        y,
        best[0],
        best[1],
        best_k
    )

    print(name, acc)

    pop_names.append(name)
    pop_accs.append(acc)
    pop_highs.append(high)
    pop_stds.append(std)

    # simpan spiral terbaik model ini
    all_best_histories[name] = history_best

    if acc > best_acc:
        best_acc = acc
        best_model = (name, best)
        best_history = history_best
        best_history_all = history_all_pop
        best_pop = pop

# =========================
# GRAFIK ACCURACY
# =========================
fig, ax = plt.subplots(figsize=(10,5))

x_pos = range(len(pop_names))

x_labels = [
    f"{m}\n({std:.3f})"
    for m, std in zip(pop_names, pop_stds)
]

ax.plot(
    x_pos,
    [h*100 for h in pop_highs],
    marker='o',
    color='blue',
    linewidth=2,
    label='highest accuracy'
)

ax.plot(
    x_pos,
    [a*100 for a in pop_accs],
    marker='o',
    color='green',
    linewidth=2,
    label='average accuracy'
)

for i, (h, a) in enumerate(zip(pop_highs, pop_accs)):

    ax.annotate(
        f"{h*100:.2f}",
        (i, h*100),
        textcoords="offset points",
        xytext=(0,6),
        ha='center',
        fontsize=8,
        color='blue'
    )

    ax.annotate(
        f"{a*100:.2f}",
        (i, a*100),
        textcoords="offset points",
        xytext=(0,-14),
        ha='center',
        fontsize=8,
        color='green'
    )

ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)

ax.set_xlabel(
    "Population Model (standard deviation)"
)

ax.set_ylabel("Accuracy")
ax.set_title(
    "Graph of testing model 4"
)

ax.legend()
ax.grid(True)

plt.show()

# =========================
# VISUALISASI 9 SPIRAL
# =========================
plt.figure(figsize=(10,8))

for model_name, history in all_best_histories.items():

    history = np.array(history)

    plt.plot(
        history[:,0],
        history[:,1],
        marker='o',
        linewidth=2,
        label=model_name
    )

    # label nama model di titik awal
    plt.annotate(
        model_name,
        (history[0,0], history[0,1]),
        fontsize=8
    )

# tandai solusi terbaik keseluruhan
plt.scatter(
    best_model[1][0],
    best_model[1][1],
    color='black',
    marker='s',
    s=150,
    label='Global Best'
)

plt.annotate(
    f"Best ({best_model[1][0]:.4f}, {best_model[1][1]:.4f})",
    (
        best_model[1][0],
        best_model[1][1]
    ),
    fontsize=9
)

plt.xlabel("Gamma")
plt.ylabel("C")
plt.title("SOA Best Trajectory for Population Models")
plt.grid(True)
plt.legend()

plt.show()

# =========================
# 9. FINAL RESULT
# =========================
gamma, C = best_model[1]

final_metrics = evaluate_full_metrics(X, y, gamma, C, best_k)

print("\n====================")
print("FINAL RESULT POLYNOMIAL")
print("====================")

print("k =", best_k)
print("theta =", best_theta)
print("r =", best_r)
print("degree =", DEGREE)
print("coef0 =", COEF0)
print("gamma =", gamma)
print("C =", C)
print("POP MODEL =", best_model[0])

print("\n--- METRIK EVALUASI (rata-rata k-fold) ---")
print(f"Accuracy  : {final_metrics['accuracy_mean']*100:.2f}%  (std = {final_metrics['accuracy_std']*100:.2f}%)")
print(f"Precision : {final_metrics['precision_mean']*100:.2f}%  (std = {final_metrics['precision_std']*100:.2f}%)")
print(f"Recall    : {final_metrics['recall_mean']*100:.2f}%  (std = {final_metrics['recall_std']*100:.2f}%)")
print(f"F1-Score  : {final_metrics['f1_mean']*100:.2f}%  (std = {final_metrics['f1_std']*100:.2f}%)")

cm = final_metrics["confusion_matrix"]
print("\nConfusion Matrix (akumulasi seluruh fold):")
print("                  Pred Negatif   Pred Positif")
print(f"Aktual Negatif        {cm[0][0]:>4}            {cm[0][1]:>4}")
print(f"Aktual Positif        {cm[1][0]:>4}            {cm[1][1]:>4}")
print(f"\n  Accuracy per Fold:")
for i, acc_fold in enumerate(final_metrics["acc_per_fold"], 1):
    print(f"    Fold {i}: {acc_fold:.4f}")
    
plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Negatif','Positif'],
    yticklabels=['Negatif','Positif']
)

plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix Polynomial SVM-SOA")

plt.show()