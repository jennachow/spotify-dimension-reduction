"""
MATH 2320 Final Project - Part 1: PCA
Team: Deja Dunlap, Jenna Chow, Aditya Das, Undra Pillows

This script loads the Spotify Tracks Dataset and applies Principal Component Analysis
to explore the structure of the audio feature space.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

df = pd.read_csv("dataset.csv")

print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# ─────────────────────────────────────────────
# 2. SELECT AUDIO FEATURES
# ─────────────────────────────────────────────
# These are the quantitative audio features we use as our feature vectors in R^n
AUDIO_FEATURES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

# Drop rows with missing values in our feature columns or the genre column
df_clean = df.dropna(subset=AUDIO_FEATURES + ["track_genre"])

# Optional: limit to a subset of genres for cleaner visualizations
# Comment this block out if you want to use all genres
GENRES_TO_USE = [
    "pop", "rock", "hip-hop", "classical", "jazz",
    "electronic", "country", "r-n-b", "metal", "acoustic"
]
df_clean = df_clean[df_clean["track_genre"].isin(GENRES_TO_USE)]

print(f"\nCleaned dataset: {df_clean.shape[0]} tracks across {df_clean['track_genre'].nunique()} genres")

# ─────────────────────────────────────────────
# 3. STANDARDIZE THE DATA
# ─────────────────────────────────────────────
# Each feature has very different scales (e.g. tempo ~ 120, danceability ~ 0.7)
# Standardizing ensures each feature contributes equally to the covariance structure.
# This centers each column to mean 0 and scales to standard deviation 1.

X = df_clean[AUDIO_FEATURES].values          # shape: (n_songs, n_features)
genres = df_clean["track_genre"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)           # shape: (n_songs, n_features)

print(f"\nFeature matrix shape: {X_scaled.shape}")

# ─────────────────────────────────────────────
# 4. PCA
# ─────────────────────────────────────────────
# sklearn's PCA computes the SVD of the centered data matrix.
# Each principal component is an eigenvector of the covariance matrix X^T X / (n-1).

pca = PCA()  # keep all components first so we can see variance explained
pca.fit(X_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print("\nVariance explained by each PC:")
for i, (ev, cum) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
    print(f"  PC{i+1}: {ev:.4f}  (cumulative: {cum:.4f})")

# ─────────────────────────────────────────────
# 5. SCREE PLOT — how many PCs do we need?
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

n_components = len(explained_variance_ratio)

axes[0].bar(range(1, n_components + 1), explained_variance_ratio, color="steelblue", alpha=0.8)
axes[0].set_xlabel("Principal Component")
axes[0].set_ylabel("Proportion of Variance Explained")
axes[0].set_title("Scree Plot")
axes[0].set_xticks(range(1, n_components + 1))

axes[1].plot(range(1, n_components + 1), cumulative_variance, marker="o", color="darkorange")
axes[1].axhline(y=0.90, color="gray", linestyle="--", label="90% threshold")
axes[1].set_xlabel("Number of Principal Components")
axes[1].set_ylabel("Cumulative Variance Explained")
axes[1].set_title("Cumulative Variance Explained")
axes[1].set_xticks(range(1, n_components + 1))
axes[1].legend()

plt.tight_layout()
plt.savefig("scree_plot.png", dpi=150)
plt.show()
print("Saved: scree_plot.png")

# ─────────────────────────────────────────────
# 6. LOADINGS (COMPONENT DIRECTIONS)
# ─────────────────────────────────────────────
# Each row of pca.components_ is a principal direction (eigenvector).
# The loading tells us how much each original feature contributes to that PC.

loadings = pd.DataFrame(
    pca.components_.T,
    index=AUDIO_FEATURES,
    columns=[f"PC{i+1}" for i in range(n_components)]
)

print("\nPC Loadings (first 3 PCs):")
print(loadings[["PC1", "PC2", "PC3"]].round(3))

plt.figure(figsize=(8, 5))
sns.heatmap(
    loadings[["PC1", "PC2", "PC3"]],
    annot=True, fmt=".2f", cmap="coolwarm", center=0
)
plt.title("PCA Loadings — First 3 Principal Components")
plt.tight_layout()
plt.savefig("pca_loadings.png", dpi=150)
plt.show()
print("Saved: pca_loadings.png")

# ─────────────────────────────────────────────
# 7. PROJECT DATA ONTO FIRST 2 PCs
# ─────────────────────────────────────────────
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)       # shape: (n_songs, 2)

# ─────────────────────────────────────────────
# 8. SCATTER PLOT — songs in PC space, colored by genre
# ─────────────────────────────────────────────
df_pca = pd.DataFrame({"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "genre": genres})

# Sample for readability if dataset is very large
SAMPLE_SIZE = 3000
if len(df_pca) > SAMPLE_SIZE:
    df_pca = df_pca.sample(n=SAMPLE_SIZE, random_state=42)

palette = sns.color_palette("tab10", n_colors=len(GENRES_TO_USE))

fig, ax = plt.subplots(figsize=(10, 7))
sns.scatterplot(
    data=df_pca, x="PC1", y="PC2",
    hue="genre", palette=palette, alpha=0.5, s=15, linewidth=0, ax=ax
)
ax.set_xlabel(f"PC1 ({explained_variance_ratio[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({explained_variance_ratio[1]*100:.1f}% variance)")
ax.set_title("Songs Projected onto First Two Principal Components")
# Move legend below the plot so it never overlaps data
handles, labels = ax.get_legend_handles_labels()
ax.get_legend().remove()
fig.legend(handles, labels,
           loc="lower center", ncol=5, fontsize=8,
           markerscale=1.5, frameon=True,
           bbox_to_anchor=(0.5, 0.0))
plt.subplots_adjust(top=0.94, bottom=0.18, left=0.09, right=0.97)
plt.savefig("pca_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: pca_scatter.png")

# ─────────────────────────────────────────────
# 9. BIPLOT — overlay feature directions on PC scatter
# ─────────────────────────────────────────────
# A biplot shows both the data points and the feature loading vectors together.

fig, ax = plt.subplots(figsize=(10, 7))

# Sample points for the scatter
sample_idx = np.random.choice(len(X_pca), size=min(1000, len(X_pca)), replace=False)
ax.scatter(X_pca[sample_idx, 0], X_pca[sample_idx, 1], alpha=0.2, s=10, color="gray")

# Draw loading arrows
scale = 3  # scale arrows to be visible relative to data spread
for i, feature in enumerate(AUDIO_FEATURES):
    ax.arrow(
        0, 0,
        pca_2d.components_[0, i] * scale,
        pca_2d.components_[1, i] * scale,
        head_width=0.05, head_length=0.05, fc="crimson", ec="crimson"
    )
    ax.text(
        pca_2d.components_[0, i] * scale * 1.15,
        pca_2d.components_[1, i] * scale * 1.15,
        feature, fontsize=9, ha="center", color="crimson"
    )

ax.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("PCA Biplot — Audio Features in PC Space")
ax.axhline(0, color="black", linewidth=0.5)
ax.axvline(0, color="black", linewidth=0.5)
plt.tight_layout()
plt.savefig("pca_biplot.png", dpi=150)
plt.show()
print("Saved: pca_biplot.png")

print("\n✓ PCA analysis complete. Output plots saved to current directory.")
