"""
MATH 2320 Final Project - Part 2: Diffusion Maps & Nonlinear Methods
Team: Deja Dunlap, Jenna Chow, Aditya Das, Undra Pillows

This script applies diffusion maps to the Spotify Tracks Dataset and compares
the resulting embedding to PCA, t-SNE, and UMAP using k-means clustering
and silhouette scores.

Run AFTER pca_analysis.py. Requires the same dataset.csv in the same folder.

Optional: install umap-learn for a UMAP comparison:
    pip3 install umap-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cdist

# ─────────────────────────────────────────────
# 1. LOAD & PREPROCESS (same as pca_analysis.py)
# ─────────────────────────────────────────────
AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
]

GENRES_TO_USE = [
    "pop", "rock", "hip-hop", "classical", "jazz",
    "electronic", "country", "r-n-b", "metal", "acoustic"
]

df = pd.read_csv("dataset.csv")
df_clean = df.dropna(subset=AUDIO_FEATURES + ["track_genre"])
df_clean = df_clean[df_clean["track_genre"].isin(GENRES_TO_USE)]

print(f"Loaded {df_clean.shape[0]} tracks across {df_clean['track_genre'].nunique()} genres.")

# Subsample for speed — diffusion maps require O(n^2) distance computation.
SAMPLE_SIZE = 2000
per_genre = SAMPLE_SIZE // len(GENRES_TO_USE)
frames = []
for genre in GENRES_TO_USE:
    g = df_clean[df_clean["track_genre"] == genre]
    frames.append(g.sample(min(len(g), per_genre), random_state=42))
df_sample = pd.concat(frames, ignore_index=True)

print(f"Working with {len(df_sample)} songs ({len(df_sample)//len(GENRES_TO_USE)} per genre).")

X_raw  = df_sample[AUDIO_FEATURES].values
genres = df_sample["track_genre"].values

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Shared color palette
palette    = sns.color_palette("tab10", n_colors=len(GENRES_TO_USE))
genre_list = sorted(GENRES_TO_USE)
color_map  = {g: palette[i] for i, g in enumerate(genre_list)}

# ─────────────────────────────────────────────
# 2. DIFFUSION MAPS — step by step
# ─────────────────────────────────────────────
print("\n── Computing pairwise distances ──")
D_sq = cdist(X_scaled, X_scaled, metric="sqeuclidean")

epsilon = np.median(D_sq)
print(f"Chosen epsilon (median heuristic): {epsilon:.4f}")

print("── Building affinity matrix W ──")
W = np.exp(-D_sq / epsilon)

print("── Normalizing to diffusion operator P ──")
degree = W.sum(axis=1)
D_inv  = np.diag(1.0 / degree)
P      = D_inv @ W

print("── Computing eigenvectors (symmetric form) ──")
D_sqrt_inv = np.diag(1.0 / np.sqrt(degree))
P_sym      = D_sqrt_inv @ W @ D_sqrt_inv

eigenvalues, eigenvectors_sym = np.linalg.eigh(P_sym)
idx          = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[idx]
eigenvectors_sym = eigenvectors_sym[:, idx]

eigenvectors = D_sqrt_inv @ eigenvectors_sym

psi = eigenvectors[:, 1:]   # skip trivial eigenvector
lam = eigenvalues[1:]

print(f"Top 5 non-trivial eigenvalues: {lam[:5].round(4)}")

t = 1
X_diff = (lam[:2] ** t) * psi[:, :2]
print("── Diffusion map embedding computed ──")

# ─────────────────────────────────────────────
# 3. PCA EMBEDDING (for comparison)
# ─────────────────────────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# ─────────────────────────────────────────────
# 3b. t-SNE EMBEDDING
# ─────────────────────────────────────────────
print("\n── t-SNE embedding (may take ~30 seconds) ──")
tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto",
            init="pca", max_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
print("── t-SNE embedding computed ──")

# ─────────────────────────────────────────────
# 3c. UMAP EMBEDDING (optional — pip3 install umap-learn)
# ─────────────────────────────────────────────
X_umap = None
sil_umap = None
try:
    import umap
    print("\n── UMAP embedding ──")
    reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap  = reducer.fit_transform(X_scaled)
    print("── UMAP embedding computed ──")
except ImportError:
    print("\n── UMAP not installed (optional). Run: pip3 install umap-learn ──")

# ─────────────────────────────────────────────
# 4. SIDE-BY-SIDE SCATTER: PCA vs DIFFUSION MAPS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

X_diff_vis = psi[:, :2]  # raw eigenvectors (no eigenvalue scaling)

for ax, (coords, title, xlabel, ylabel) in zip(axes, [
    (X_pca,      "PCA Embedding",
     f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
     f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"),
    (X_diff_vis, "Diffusion Map Embedding",
     r"$\psi_1$  (diffusion eigenvector 1)",
     r"$\psi_2$  (diffusion eigenvector 2)"),
]):
    for genre in genre_list:
        mask = genres == genre
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   label=genre, alpha=0.55, s=14, color=color_map[genre])
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.axhline(0, color="lightgray", linewidth=0.6, zorder=0)
    ax.axvline(0, color="lightgray", linewidth=0.6, zorder=0)
    xp = (coords[:, 0].max() - coords[:, 0].min()) * 0.08
    yp = (coords[:, 1].max() - coords[:, 1].min()) * 0.08
    ax.set_xlim(coords[:, 0].min() - xp, coords[:, 0].max() + xp)
    ax.set_ylim(coords[:, 1].min() - yp, coords[:, 1].max() + yp)

handles, labels_leg = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_leg, loc="lower center", ncol=5, fontsize=9,
           markerscale=2, frameon=True, bbox_to_anchor=(0.5, 0.0))
plt.suptitle("Songs in 2D: PCA vs Diffusion Maps", fontsize=13, fontweight="bold")
plt.subplots_adjust(top=0.92, bottom=0.22, left=0.07, right=0.97, wspace=0.3)
plt.savefig("pca_vs_diffusion.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: pca_vs_diffusion.png")

# ─────────────────────────────────────────────
# 5. EIGENVALUE SPECTRUM
# ─────────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.bar(range(1, 11), lam[:10], color="steelblue", alpha=0.8)
plt.xlabel("Eigenvector Index (non-trivial)")
plt.ylabel("Eigenvalue")
plt.title("Diffusion Map Eigenvalue Spectrum")
plt.xticks(range(1, 11))
plt.tight_layout()
plt.savefig("diffusion_eigenvalues.png", dpi=150)
plt.show()
print("Saved: diffusion_eigenvalues.png")

# ─────────────────────────────────────────────
# 6. t-SNE STANDALONE PLOT
# ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6.5))
for genre in genre_list:
    mask = genres == genre
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               label=genre, alpha=0.6, s=18, color=color_map[genre])
ax.set_xlabel("t-SNE Dimension 1", fontsize=11)
ax.set_ylabel("t-SNE Dimension 2", fontsize=11)
ax.set_title("t-SNE Embedding of Spotify Audio Features", fontsize=12, fontweight="bold")

handles, labels_leg = ax.get_legend_handles_labels()
ax.get_legend().remove() if ax.get_legend() else None
fig.legend(handles, labels_leg, loc="lower center", ncol=5, fontsize=9,
           markerscale=2, frameon=True, bbox_to_anchor=(0.5, 0.0))
plt.subplots_adjust(top=0.93, bottom=0.18, left=0.09, right=0.97)
plt.savefig("tsne_embedding.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: tsne_embedding.png")

# ─────────────────────────────────────────────
# 7. UMAP STANDALONE PLOT (if installed)
# ─────────────────────────────────────────────
if X_umap is not None:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for genre in genre_list:
        mask = genres == genre
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                   label=genre, alpha=0.6, s=18, color=color_map[genre])
    ax.set_xlabel("UMAP Dimension 1", fontsize=11)
    ax.set_ylabel("UMAP Dimension 2", fontsize=11)
    ax.set_title("UMAP Embedding of Spotify Audio Features", fontsize=12, fontweight="bold")

    handles, labels_leg = ax.get_legend_handles_labels()
    ax.get_legend().remove() if ax.get_legend() else None
    fig.legend(handles, labels_leg, loc="lower center", ncol=5, fontsize=9,
               markerscale=2, frameon=True, bbox_to_anchor=(0.5, 0.0))
    plt.subplots_adjust(top=0.93, bottom=0.18, left=0.09, right=0.97)
    plt.savefig("umap_embedding.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: umap_embedding.png")

# ─────────────────────────────────────────────
# 8. FOUR-METHOD COMPARISON FIGURE
#    PCA  |  Diffusion Maps  |  t-SNE  |  UMAP
#    (3-panel if UMAP not installed)
# ─────────────────────────────────────────────
panels_compare = [
    (X_pca,       "PCA",
     f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
     f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"),
    (X_diff_vis,  "Diffusion Maps",
     r"$\psi_1$", r"$\psi_2$"),
    (X_tsne,      "t-SNE",
     "t-SNE Dim 1", "t-SNE Dim 2"),
]
if X_umap is not None:
    panels_compare.append((X_umap, "UMAP", "UMAP Dim 1", "UMAP Dim 2"))

n_panels = len(panels_compare)
fig_w = 5.5 * n_panels
fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, 5.5))
if n_panels == 1:
    axes = [axes]

for ax, (coords, title, xlabel, ylabel) in zip(axes, panels_compare):
    for genre in genre_list:
        mask = genres == genre
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   label=genre, alpha=0.6, s=14, color=color_map[genre])
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)

handles, labels_leg = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_leg, loc="lower center", ncol=5, fontsize=9,
           markerscale=2, frameon=True, bbox_to_anchor=(0.5, 0.0))
plt.suptitle("Four Embedding Methods Compared: Genre Separation", fontsize=13, fontweight="bold")
plt.subplots_adjust(top=0.91, bottom=0.20, left=0.05, right=0.99, wspace=0.32)
plt.savefig("embedding_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: embedding_comparison.png")

# ─────────────────────────────────────────────
# 9. K-MEANS CLUSTERING + SILHOUETTE SCORES
# ─────────────────────────────────────────────
print("\n── K-means clustering ──")
N_CLUSTERS = len(GENRES_TO_USE)
N_INIT     = 10

kmeans_pca  = KMeans(n_clusters=N_CLUSTERS, n_init=N_INIT, random_state=42)
kmeans_diff = KMeans(n_clusters=N_CLUSTERS, n_init=N_INIT, random_state=42)
kmeans_tsne = KMeans(n_clusters=N_CLUSTERS, n_init=N_INIT, random_state=42)

labels_pca  = kmeans_pca.fit_predict(X_pca)
labels_diff = kmeans_diff.fit_predict(X_diff)
labels_tsne = kmeans_tsne.fit_predict(X_tsne)

sil_pca  = silhouette_score(X_pca,  labels_pca)
sil_diff = silhouette_score(X_diff, labels_diff)
sil_tsne = silhouette_score(X_tsne, labels_tsne)

print(f"  Silhouette score — PCA embedding:            {sil_pca:.4f}")
print(f"  Silhouette score — Diffusion map embedding:  {sil_diff:.4f}")
print(f"  Silhouette score — t-SNE embedding:          {sil_tsne:.4f}")

if X_umap is not None:
    kmeans_umap = KMeans(n_clusters=N_CLUSTERS, n_init=N_INIT, random_state=42)
    labels_umap = kmeans_umap.fit_predict(X_umap)
    sil_umap    = silhouette_score(X_umap, labels_umap)
    print(f"  Silhouette score — UMAP embedding:           {sil_umap:.4f}")

# Bar chart
method_labels = ["PCA", "Diffusion\nMaps", "t-SNE"]
scores_bar    = [sil_pca, sil_diff, sil_tsne]
bar_colors    = ["steelblue", "darkorange", "mediumpurple"]
if X_umap is not None:
    method_labels.append("UMAP")
    scores_bar.append(sil_umap)
    bar_colors.append("seagreen")

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(method_labels, scores_bar, color=bar_colors, alpha=0.85, width=0.45)
ax.set_ylabel("Silhouette Score")
ax.set_title("Clustering Quality by Embedding Method")
ax.axhline(0, color="black", linewidth=0.8)
for bar, score in zip(bars, scores_bar):
    ypos = bar.get_height() + 0.003 if score >= 0 else bar.get_height() - 0.015
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f"{score:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("silhouette_comparison.png", dpi=150)
plt.show()
print("Saved: silhouette_comparison.png")

# ─────────────────────────────────────────────
# 10. EPSILON SENSITIVITY ANALYSIS
# ─────────────────────────────────────────────
print("\n── Epsilon sensitivity ──")
eps_values = np.logspace(np.log10(epsilon/10), np.log10(epsilon*10), 15)
sil_scores_eps = []

for eps in eps_values:
    W_e   = np.exp(-D_sq / eps)
    deg_e = W_e.sum(axis=1)
    D_si  = np.diag(1.0 / np.sqrt(deg_e))
    Ps    = D_si @ W_e @ D_si
    ev, evec = np.linalg.eigh(Ps)
    idx_e = np.argsort(ev)[::-1]
    evec  = (np.diag(1.0 / np.sqrt(deg_e)) @ evec[:, idx_e])[:, 1:3]
    lam_e = np.sort(ev)[::-1][1:3]
    emb   = (lam_e ** 1) * evec
    try:
        km = KMeans(n_clusters=N_CLUSTERS, n_init=5, random_state=42).fit(emb)
        sil_scores_eps.append(silhouette_score(emb, km.labels_))
    except Exception:
        sil_scores_eps.append(np.nan)

plt.figure(figsize=(8, 4))
plt.semilogx(eps_values, sil_scores_eps, marker="o", color="darkorange")
plt.axvline(epsilon, color="gray", linestyle="--", label=f"Chosen \u03b5 = {epsilon:.2f}")
plt.xlabel("Epsilon (\u03b5)")
plt.ylabel("Silhouette Score")
plt.title("Diffusion Maps: Sensitivity to \u03b5")
plt.legend()
plt.tight_layout()
plt.savefig("epsilon_sensitivity.png", dpi=150)
plt.show()
print("Saved: epsilon_sensitivity.png")

# ─────────────────────────────────────────────
# 11. SPECTRAL CLUSTERING
# ─────────────────────────────────────────────
print("\n── Spectral clustering ──")

X_spectral = psi[:, :N_CLUSTERS]
kmeans_spectral = KMeans(n_clusters=N_CLUSTERS, n_init=N_INIT, random_state=42)
labels_spectral = kmeans_spectral.fit_predict(X_spectral)
sil_spectral    = silhouette_score(X_spectral, labels_spectral)

print(f"  Silhouette score — Spectral clustering (eigenvectors): {sil_spectral:.4f}")

# Three-panel: PCA | Diffusion Map | Spectral (all colored by true genre)
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

panels_spectral = [
    (X_pca,      "K-Means on PCA",
     f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)",
     f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"),
    (psi[:, :2], "Diffusion Map\n(\u03c8\u2081 vs \u03c8\u2082)",
     r"$\psi_1$", r"$\psi_2$"),
    (psi[:, :2], "Spectral Clustering\n(Graph Laplacian eigenvectors)",
     r"$\psi_1$", r"$\psi_2$"),
]

for ax, (coords, title, xlabel, ylabel) in zip(axes, panels_spectral):
    for genre in genre_list:
        mask = genres == genre
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   label=genre, color=color_map[genre], alpha=0.6, s=16)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.axhline(0, color="lightgray", linewidth=0.6, zorder=0)
    ax.axvline(0, color="lightgray", linewidth=0.6, zorder=0)
    xp = (coords[:, 0].max() - coords[:, 0].min()) * 0.08
    yp = (coords[:, 1].max() - coords[:, 1].min()) * 0.08
    ax.set_xlim(coords[:, 0].min() - xp, coords[:, 0].max() + xp)
    ax.set_ylim(coords[:, 1].min() - yp, coords[:, 1].max() + yp)

handles, labels_leg = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_leg, loc="lower center", ncol=5, fontsize=9,
           markerscale=2, frameon=True, bbox_to_anchor=(0.5, 0.0))
plt.suptitle("Genre Structure: Three Embedding Methods Compared", fontsize=12, fontweight="bold")
plt.subplots_adjust(top=0.91, bottom=0.24, left=0.05, right=0.99, wspace=0.35)
plt.savefig("spectral_clustering.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: spectral_clustering.png")

# Updated silhouette bar chart including spectral clustering
fig, ax = plt.subplots(figsize=(9, 4))
methods_all = ["PCA\n(k-means)", "Diffusion Map\n(k-means)", "Spectral\nClustering"]
scores_all  = [sil_pca, sil_diff, sil_spectral]
colors_all  = ["steelblue", "darkorange", "seagreen"]
bars = ax.bar(methods_all, scores_all, color=colors_all, alpha=0.8, width=0.4)
ax.set_ylabel("Silhouette Score")
ax.set_title("Clustering Quality: All Three Methods")
ax.axhline(0, color="black", linewidth=0.8)
for bar, score in zip(bars, scores_all):
    ypos = bar.get_height() + 0.003 if score >= 0 else bar.get_height() - 0.015
    ax.text(bar.get_x() + bar.get_width()/2, ypos,
            f"{score:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("silhouette_all_methods.png", dpi=150)
plt.show()
print("Saved: silhouette_all_methods.png")

# ─────────────────────────────────────────────
# 12. KNN GENRE CLASSIFICATION ACCURACY
# ─────────────────────────────────────────────
print("\n── KNN genre classification accuracy ──")

le = LabelEncoder()
y  = le.fit_transform(genres)

embeddings_for_knn = [
    ("PCA",               X_pca),
    ("Diffusion Map",     X_diff),
    ("t-SNE",             X_tsne),
    ("Spectral (eigenvecs)", X_spectral[:, :2]),
]
if X_umap is not None:
    embeddings_for_knn.append(("UMAP", X_umap))

knn_results = {}
for name, X_emb in embeddings_for_knn:
    X_tr, X_te, y_tr, y_te = train_test_split(X_emb, y, test_size=0.2, random_state=42, stratify=y)
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, knn.predict(X_te))
    knn_results[name] = acc
    print(f"  KNN accuracy — {name:25s}: {acc*100:.1f}%")

knn_colors = ["steelblue", "darkorange", "mediumpurple", "seagreen", "crimson"]
fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(list(knn_results.keys()), list(knn_results.values()),
              color=knn_colors[:len(knn_results)], alpha=0.85, width=0.45)
ax.set_ylabel("Genre Classification Accuracy")
ax.set_ylim(0, 1)
ax.set_title("KNN Genre Prediction Accuracy by Embedding Method")
ax.axhline(1/N_CLUSTERS, color="gray", linestyle="--",
           label=f"Random baseline ({100/N_CLUSTERS:.0f}%)")
ax.legend()
for bar, (name, acc) in zip(bars, knn_results.items()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{acc*100:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig("knn_accuracy.png", dpi=150)
plt.show()
print("Saved: knn_accuracy.png")

# ─────────────────────────────────────────────
# 13. FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n" + "="*55)
print("  RESULTS SUMMARY")
print("="*55)
print(f"  Silhouette — PCA (k-means):        {sil_pca:.4f}")
print(f"  Silhouette — Diffusion Map:         {sil_diff:.4f}")
print(f"  Silhouette — t-SNE:                 {sil_tsne:.4f}")
if sil_umap is not None:
    print(f"  Silhouette — UMAP:                  {sil_umap:.4f}")
print(f"  Silhouette — Spectral Clustering:   {sil_spectral:.4f}")
print()
for name, acc in knn_results.items():
    print(f"  KNN accuracy — {name:25s}: {acc*100:.1f}%")

all_sil = [("PCA", sil_pca), ("Diffusion Map", sil_diff),
           ("t-SNE", sil_tsne), ("Spectral", sil_spectral)]
if sil_umap is not None:
    all_sil.append(("UMAP", sil_umap))
best_sil = max(all_sil, key=lambda x: x[1])
best_knn = max(knn_results.items(), key=lambda x: x[1])
print()
print(f"  Best silhouette:   {best_sil[0]}  ({best_sil[1]:.4f})")
print(f"  Best KNN accuracy: {best_knn[0]}  ({best_knn[1]*100:.1f}%)")
print("="*55)
print("\n\u2713 All plots saved. Analysis complete.")
