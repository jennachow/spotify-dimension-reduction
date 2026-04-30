import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist

GENRES = ['pop', 'rock', 'hip-hop', 'classical', 'jazz',
          'electronic', 'country', 'r&b', 'metal', 'acoustic']
FEATURES = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
N_PER_GENRE = 200
RANDOM_STATE = 42


def load_data(path='dataset.csv'):
    df = pd.read_csv(path)
    df['track_genre'] = df['track_genre'].str.lower()
    genre_col = 'track_genre'
    samples = []
    for g in GENRES:
        sub = df[df[genre_col] == g].dropna(subset=FEATURES)
        if len(sub) >= N_PER_GENRE:
            samples.append(sub.sample(N_PER_GENRE, random_state=RANDOM_STATE))
        else:
            samples.append(sub)
    data = pd.concat(samples, ignore_index=True)
    X = StandardScaler().fit_transform(data[FEATURES].values)
    y = data[genre_col].values
    return X, y


def gaussian_kernel(X, epsilon):
    D2 = cdist(X, X, 'sqeuclidean')
    return np.exp(-D2 / (2 * epsilon))


def diffusion_map(X, epsilon=None, n_components=10, t=1):
    D2 = cdist(X, X, 'sqeuclidean')
    if epsilon is None:
        epsilon = np.median(D2)
    W = np.exp(-D2 / (2 * epsilon))
    d = W.sum(axis=1)
    D_inv = np.diag(1.0 / d)
    P = D_inv @ W
    D_half = np.diag(d ** 0.5)
    D_inv_half = np.diag(d ** -0.5)
    P_sym = D_half @ P @ D_inv_half
    eigvals, eigvecs = np.linalg.eigh(P_sym)
    idx = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    psi = D_inv_half @ eigvecs
    coords = psi[:, 1:n_components+1] * (eigvals[1:n_components+1] ** t)
    return coords, eigvals, psi


def plot_embedding(coords, labels, i, j, title, filename, xlabel=None, ylabel=None):
    genres = np.unique(labels)
    colors = cm.tab10(np.linspace(0, 1, len(genres)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for g, c in zip(genres, colors):
        mask = labels == g
        ax.scatter(coords[mask, i], coords[mask, j], c=[c], label=g,
                   alpha=0.6, s=15, linewidths=0)
    ax.set_xlabel(xlabel or f'Component {i+1}')
    ax.set_ylabel(ylabel or f'Component {j+1}')
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc='best')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def epsilon_sensitivity(X, y, epsilons, n_neighbors=10):
    scores = []
    for eps in epsilons:
        coords, _, _ = diffusion_map(X, epsilon=eps, n_components=3)
        Xtr, Xte, ytr, yte = train_test_split(coords, y, test_size=0.2,
                                               random_state=RANDOM_STATE,
                                               stratify=y)
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(Xtr, ytr)
        scores.append(knn.score(Xte, yte))
    return scores


def spectral_clustering_embed(X, epsilon=None, K=10):
    D2 = cdist(X, X, 'sqeuclidean')
    if epsilon is None:
        epsilon = np.median(D2)
    W = np.exp(-D2 / (2 * epsilon))
    d = W.sum(axis=1)
    D_inv_half = np.diag(d ** -0.5)
    L_sym = np.eye(len(X)) - D_inv_half @ W @ D_inv_half
    eigvals, eigvecs = np.linalg.eigh(L_sym)
    return eigvecs[:, 1:K+1], eigvals


def remove_outliers(X, y, threshold=3.5):
    mean_dists = cdist(X, X).mean(axis=1)
    cutoff = mean_dists.mean() + threshold * mean_dists.std()
    mask = mean_dists < cutoff
    print(f"Outlier removal: kept {mask.sum()} / {len(X)} points "
          f"(removed {(~mask).sum()})")
    return X[mask], y[mask]


def knn_accuracy(X_embed, y, n_neighbors=10, test_size=0.2):
    Xtr, Xte, ytr, yte = train_test_split(X_embed, y, test_size=test_size,
                                           random_state=RANDOM_STATE, stratify=y)
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(Xtr, ytr)
    return knn.score(Xte, yte)


def main():
    print("Loading data...")
    X, y = load_data()
    X, y = remove_outliers(X, y)

    print("Computing diffusion maps...")
    coords, eigvals, psi = diffusion_map(X, n_components=9)

    print("Plotting diffusion map eigenvalue spectrum...")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, 11), eigvals[1:11], color='steelblue')
    ax.set_xlabel('Eigenvector Index (non-trivial)')
    ax.set_ylabel('Eigenvalue')
    ax.set_title('Diffusion Map Eigenvalue Spectrum')
    plt.tight_layout()
    plt.savefig('diffusion_eigenvalues.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Plotting diffusion map: ψ₁ vs ψ₂ (eigenvectors 1 and 2)...")
    plot_embedding(coords, y, 0, 1,
                   title='Diffusion Map Embedding: ψ₁ vs ψ₂',
                   filename='diffusion_embedding_psi12.png',
                   xlabel='ψ₁ (Diffusion Component 1)',
                   ylabel='ψ₂ (Diffusion Component 2)')

    print("Plotting diffusion map: ψ₂ vs ψ₃ (eigenvectors 2 and 3)...")
    plot_embedding(coords, y, 1, 2,
                   title='Diffusion Map Embedding: ψ₂ vs ψ₃',
                   filename='diffusion_embedding_psi23.png',
                   xlabel='ψ₂ (Diffusion Component 2)',
                   ylabel='ψ₃ (Diffusion Component 3)')

    print("Running epsilon sensitivity analysis...")
    epsilons = np.logspace(-1, 2, 20)
    eps_scores = epsilon_sensitivity(X, y, epsilons)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogx(epsilons, eps_scores, 'o-', color='darkorange')
    ax.set_xlabel('ε (bandwidth parameter)')
    ax.set_ylabel('KNN Accuracy (k=10)')
    ax.set_title('Diffusion Maps: ε Sensitivity')
    ax.axvline(np.median(cdist(X, X, 'sqeuclidean')), linestyle='--',
               color='grey', label='median pairwise distance')
    ax.legend()
    plt.tight_layout()
    plt.savefig('epsilon_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Computing spectral clustering embedding...")
    sc_coords, sc_eigvals = spectral_clustering_embed(X, K=10)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    genres = np.unique(y)
    colors = cm.tab10(np.linspace(0, 1, len(genres)))

    for ax, (coord_pair, title) in zip(axes, [
        (coords[:, :2], 'Diffusion Map (ψ₁, ψ₂)'),
        (sc_coords[:, :2], 'Spectral Clustering (ψ₁, ψ₂)'),
        (coords[:, 1:3], 'Diffusion Map (ψ₂, ψ₃)'),
    ]):
        for g, c in zip(genres, colors):
            mask = y == g
            ax.scatter(coord_pair[mask, 0], coord_pair[mask, 1],
                       c=[c], label=g, alpha=0.6, s=12, linewidths=0)
        ax.set_title(title, fontsize=10)
        ax.tick_params(labelsize=7)

    axes[0].legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig('spectral_clustering.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=40, random_state=RANDOM_STATE, max_iter=1000)
    tsne_coords = tsne.fit_transform(X)
    plot_embedding(tsne_coords, y, 0, 1,
                   title='t-SNE Embedding of Spotify Audio Features',
                   filename='tsne_embedding.png',
                   xlabel='t-SNE Dimension 1',
                   ylabel='t-SNE Dimension 2')

    print("Evaluating KNN accuracy across methods...")
    methods = {
        'Diffusion Maps (ψ₁–ψ₂)': coords[:, :2],
        'Diffusion Maps (ψ₁–ψ₃)': coords[:, :3],
        'Spectral Clustering': sc_coords,
        't-SNE': tsne_coords,
    }
    accs = {name: knn_accuracy(emb, y) for name, emb in methods.items()}

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(accs.keys(), accs.values(), color=['steelblue', 'mediumseagreen', 'darkorange', 'orchid'])
    ax.axhline(0.1, linestyle='--', color='red', label='Random baseline (10%)')
    ax.set_ylabel('KNN Accuracy (k=10)')
    ax.set_title('Genre Classification Accuracy by Embedding Method')
    ax.set_ylim(0, 1)
    ax.legend()
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig('knn_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("Computing silhouette scores...")
    sil_scores = {}
    for name, emb in methods.items():
        km = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=10)
        labels_pred = km.fit_predict(emb)
        sil_scores[name] = silhouette_score(emb, labels_pred)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sil_scores.keys(), sil_scores.values(),
           color=['steelblue', 'mediumseagreen', 'darkorange', 'orchid'])
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Clustering Quality by Embedding Method')
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig('silhouette_all_methods.png', dpi=150, bbox_inches='tight')
    plt.close()

    print("\nResults summary:")
    print(f"{'Method':<30} {'KNN Acc':>10} {'Silhouette':>12}")
    print("-" * 55)
    for name in methods:
        print(f"{name:<30} {accs[name]:>10.3f} {sil_scores[name]:>12.3f}")


if __name__ == '__main__':
    main()
