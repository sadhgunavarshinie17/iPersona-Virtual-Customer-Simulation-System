import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def find_optimal_clusters(data, max_clusters=10, plot=True):
    """Find optimal number of clusters using multiple metrics (performance optimized)"""
    n_samples = len(data)
    max_k = min(max_clusters + 1, n_samples)
    K_range = range(2, max_k)
    results = np.zeros((len(K_range), 4))

    # Use MiniBatchKMeans for large datasets to speed up metric calculation
    use_mini = n_samples > 5000

    for i, k in enumerate(K_range):
        if use_mini:
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1024, max_iter=1000, n_init=10)
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, max_iter=1000, n_init=10, algorithm='elkan')
        labels = kmeans.fit_predict(data)
        # Only compute silhouette if k < n_samples (required by sklearn)
        sil = silhouette_score(data, labels) if n_samples > k else 0
        results[i] = [
            kmeans.inertia_,
            sil,
            calinski_harabasz_score(data, labels),
            davies_bouldin_score(data, labels)
        ]

    inertias, silhouette_scores, calinski_scores, davies_scores = results.T

    if plot:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        plots = [
            (ax1, inertias, 'bo-', 'Inertia', 'Elbow Method'),
            (ax2, silhouette_scores, 'ro-', 'Silhouette Score', 'Silhouette Analysis'),
            (ax3, calinski_scores, 'go-', 'Calinski-Harabasz Score', 'Calinski-Harabasz Index'),
            (ax4, davies_scores, 'mo-', 'Davies-Bouldin Score', 'Davies-Bouldin Index')
        ]
        for ax, metric, style, ylabel, title in plots:
            ax.plot(K_range, metric, style)
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True)
        plt.tight_layout()
        plt.show()

    # Normalize and combine metrics for ensemble selection
    def norm(x): return (x - np.min(x)) / (np.ptp(x)) if np.ptp(x) else np.zeros_like(x)
    normalized_scores = np.column_stack([
        norm(inertias),
        silhouette_scores,
        norm(calinski_scores),
        1 - norm(davies_scores)
    ])
    combined_scores = np.mean(normalized_scores, axis=1)
    optimal_k = list(K_range)[int(np.argmax(combined_scores))]

    return optimal_k, silhouette_scores

def fit_predict(df, preprocess_func):
    """Fit the clustering model and return predictions (performance optimized)"""
    # Preprocess data
    scaled_data, feature_names = preprocess_func(df)

    # Find optimal number of clusters
    if len(df) > 3:
        optimal_k, _ = find_optimal_clusters(scaled_data, plot=False)
        print(f"Optimal number of clusters (ensemble approach): {optimal_k}")

        # Use MiniBatchKMeans for large datasets
        if len(df) > 10000:
            kmeans = MiniBatchKMeans(n_clusters=optimal_k, random_state=42, batch_size=2048, max_iter=300, n_init=5)
        else:
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, max_iter=300, n_init=5, algorithm='elkan')

        clusters = kmeans.fit_predict(scaled_data)
        return clusters
    else:
        print("Not enough data for clustering.")
        return np.zeros(len(df), dtype=int)