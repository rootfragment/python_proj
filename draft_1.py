# %% [markdown]
# # Mini Project: Network Attack Detection on IoT Traffic (RT-IoT2022)
# Exploring **NumPy** and **Matplotlib** with very simple machine learning.
# 
# **Goal:** classify a network flow as *normal* IoT traffic or an *attack*.
# 
# Steps: load data, explore with plots, build a normal-vs-attack label, preprocess with NumPy, train k-NN (from scratch in NumPy) plus a Decision Tree and Logistic Regression, then evaluate with plots.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
plt.rcParams["figure.dpi"] = 100

# Put the unzipped dataset file next to this notebook (the file is named RT_IOT2022, with no extension)
DATA_PATH = "RT_IOT2022"

# %% [markdown]
# ## 1. Load the data

# %%
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print("Missing values:", df.isna().sum().sum())
df.head()

# %% [markdown]
# ## 2. Explore: how many flows of each type?

# %%
counts = df["Attack_type"].value_counts()
print(counts)

plt.figure(figsize=(10, 5))
plt.bar(counts.index, counts.values, color="steelblue")
plt.yscale("log")   # log scale because DOS_SYN_Hping dominates
plt.xticks(rotation=60, ha="right")
plt.ylabel("Number of flows (log scale)")
plt.title("Flows per traffic type")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Make a simple two-class label
# `0` = normal IoT traffic, `1` = attack. This is much easier than predicting all 12 classes.

# %%
normal_types = ["MQTT_Publish", "Thing_Speak", "Wipro_bulb"]
df["label"] = (~df["Attack_type"].isin(normal_types)).astype(int)

print(df["label"].value_counts().rename({0: "normal", 1: "attack"}))

plt.figure(figsize=(5, 4))
plt.bar(["Normal", "Attack"], df["label"].value_counts().sort_index().values, color=["seagreen", "indianred"])
plt.title("Normal vs attack flows (imbalanced!)")
plt.ylabel("Number of flows")
plt.show()

# %% [markdown]
# ## 4. Choose features
# We drop `Unnamed: 0` (row index) and `id.orig_p` (random source port) and use about 10 numeric features.

# %%
features = [
    "flow_duration", "fwd_pkts_tot", "bwd_pkts_tot", "flow_pkts_per_sec",
    "fwd_pkts_payload.tot", "bwd_pkts_payload.tot",
    "flow_SYN_flag_count", "flow_RST_flag_count",
    "fwd_init_window_size", "bwd_init_window_size",
]

X_all = df[features].to_numpy(dtype=float)   # NumPy array
y_all = df["label"].to_numpy()
print("X shape:", X_all.shape, "| y shape:", y_all.shape)

# %% [markdown]
# ## 5. Histograms: normal vs attack (log scale)

# %%
fig, axes = plt.subplots(2, 5, figsize=(20, 7))
for ax, i in zip(axes.ravel(), range(len(features))):
    col = np.log10(X_all[:, i] + 1)        # log10(x+1) handles skewed data and zeros
    ax.hist(col[y_all == 0], bins=40, alpha=0.6, label="normal", color="seagreen", density=True)
    ax.hist(col[y_all == 1], bins=40, alpha=0.6, label="attack", color="indianred", density=True)
    ax.set_title(features[i], fontsize=10)
    ax.set_xlabel("log10(value + 1)")
axes[0, 0].legend()
plt.suptitle("Feature distributions: normal vs attack", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Boxplot and scatter plot

# %%
# Boxplot of packets per second (log scale)
idx = features.index("flow_pkts_per_sec")
plt.figure(figsize=(6, 5))
plt.boxplot([np.log10(X_all[y_all == 0, idx] + 1), np.log10(X_all[y_all == 1, idx] + 1)],
            )
plt.xticks([1, 2], ["normal", "attack"])   # works on every Matplotlib version
plt.ylabel("log10(flow_pkts_per_sec + 1)")
plt.title("Packets per second: normal vs attack")
plt.show()

# Scatter plot of two features on a random subset of 3000 points
sub = np.random.choice(len(X_all), 3000, replace=False)
a, b = features.index("fwd_pkts_tot"), features.index("flow_pkts_per_sec")
plt.figure(figsize=(7, 5))
plt.scatter(np.log10(X_all[sub, a] + 1), np.log10(X_all[sub, b] + 1),
            c=y_all[sub], cmap="coolwarm", s=8, alpha=0.6)
plt.xlabel("log10(fwd_pkts_tot + 1)")
plt.ylabel("log10(flow_pkts_per_sec + 1)")
plt.title("Scatter plot (blue = normal, red = attack)")
plt.show()

# %% [markdown]
# ## 7. Correlation heatmap (computed with NumPy)

# %%
corr = np.corrcoef(X_all, rowvar=False)

plt.figure(figsize=(9, 7))
im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(im)
plt.xticks(range(len(features)), features, rotation=90)
plt.yticks(range(len(features)), features)
for i in range(len(features)):
    for j in range(len(features)):
        plt.text(j, i, f"{corr[i, j]:.1f}", ha="center", va="center", fontsize=7)
plt.title("Feature correlation heatmap")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Prepare the data with NumPy
# - Take a **balanced sample** (5000 normal + 5000 attack) so the model is not dominated by one class.
# - Shuffle, then split 80% train / 20% test.
# - **Normalize** with the training mean and standard deviation (after log transform to tame skewed values).

# %%
n_per_class = 5000
normal_idx = np.where(y_all == 0)[0]
attack_idx = np.where(y_all == 1)[0]

chosen = np.concatenate([
    np.random.choice(normal_idx, n_per_class, replace=False),
    np.random.choice(attack_idx, n_per_class, replace=False),
])
chosen = np.random.permutation(chosen)          # shuffle

X = np.log10(X_all[chosen] + 1)                 # log transform
y = y_all[chosen]

# 80/20 split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Normalize using TRAIN statistics only (avoids leaking test info)
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
std[std == 0] = 1.0                              # avoid division by zero
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

print("Train:", X_train.shape, "| Test:", X_test.shape)
print("Train class balance:", np.bincount(y_train))
print("Test class balance:", np.bincount(y_test))

# %% [markdown]
# ## 9. k-Nearest Neighbors written from scratch with NumPy
# For each test point, find the `k` closest training points (Euclidean distance) and take a majority vote.

# %%
def knn_predict(X_train, y_train, X_test, k=5, batch=500):
    preds = []
    train_sq = (X_train ** 2).sum(axis=1)
    for i in range(0, len(X_test), batch):
        xb = X_test[i:i + batch]
        # squared Euclidean distance: |a|^2 - 2ab + |b|^2 (vectorized)
        d = (xb ** 2).sum(axis=1)[:, None] - 2 * xb @ X_train.T + train_sq[None, :]
        nearest = np.argpartition(d, k, axis=1)[:, :k]    # indices of k smallest distances
        votes = y_train[nearest].sum(axis=1)               # number of "attack" neighbours
        preds.append((votes > k / 2).astype(int))
    return np.concatenate(preds)

y_pred_knn = knn_predict(X_train, y_train, X_test, k=5)
print("First 20 predictions:", y_pred_knn[:20])
print("First 20 actual     :", y_test[:20])

# %% [markdown]
# ## 10. Evaluation helpers (accuracy, precision, recall, confusion matrix)

# %%
def evaluate(y_true, y_pred, name="Model"):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    print(f"{name}: accuracy={accuracy:.4f}  precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}")
    return np.array([[tn, fp], [fn, tp]]), accuracy

def plot_confusion(cm, title):
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["normal", "attack"])
    plt.yticks([0, 1], ["normal", "attack"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    plt.title(title)
    plt.tight_layout()
    plt.show()

cm_knn, acc_knn = evaluate(y_test, y_pred_knn, "k-NN from scratch (k=5)")
plot_confusion(cm_knn, "k-NN from scratch: confusion matrix")

# %% [markdown]
# ## 11. Compare with scikit-learn k-NN and test different values of k

# %%
sk_knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
evaluate(y_test, sk_knn.predict(X_test), "scikit-learn k-NN (k=5)")

k_values = [1, 3, 5, 7, 9, 11, 15, 21]
accs = []
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    accs.append((model.predict(X_test) == y_test).mean())

plt.figure(figsize=(7, 4))
plt.plot(k_values, accs, marker="o")
plt.xlabel("k (number of neighbours)")
plt.ylabel("Test accuracy")
plt.title("k-NN accuracy vs k")
plt.grid(alpha=0.3)
plt.show()

# %% [markdown]
# ## 12. Decision Tree and feature importance

# %%
tree = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
cm_tree, acc_tree = evaluate(y_test, y_pred_tree, "Decision Tree (depth 5)")
plot_confusion(cm_tree, "Decision Tree: confusion matrix")

order = np.argsort(tree.feature_importances_)
plt.figure(figsize=(8, 5))
plt.barh(np.array(features)[order], tree.feature_importances_[order], color="darkorange")
plt.xlabel("Importance")
plt.title("Which features matter most? (Decision Tree)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 13. Logistic Regression (second simple model)

# %%
logreg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)
cm_lr, acc_lr = evaluate(y_test, y_pred_lr, "Logistic Regression")
plot_confusion(cm_lr, "Logistic Regression: confusion matrix")

# %% [markdown]
# ## 14. Final comparison

# %%
names = ["k-NN (scratch)", "Decision Tree", "Logistic Regression"]
scores = [acc_knn, acc_tree, acc_lr]

plt.figure(figsize=(7, 4))
bars = plt.bar(names, scores, color=["steelblue", "darkorange", "seagreen"])
plt.ylim(min(scores) - 0.05, 1.0)
plt.ylabel("Test accuracy")
plt.title("Model comparison")
for bar, s in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width() / 2, s + 0.002, f"{s:.3f}", ha="center")
plt.show()

# %% [markdown]
# ## 15. Conclusions (edit these in your own words)
# - The dataset is heavily imbalanced (mostly `DOS_SYN_Hping`), so I used a balanced sample and looked at precision and recall as well as accuracy.
# - Attack traffic such as SYN floods has very different packet counts, rates and flag counts from normal IoT traffic, so even simple models do well.
# - The feature importance plot shows which flow features separate attacks from normal traffic most clearly.
# - Limitations: only 10 features were used, one random sample and a single train/test split, and the data comes from one test setup.
