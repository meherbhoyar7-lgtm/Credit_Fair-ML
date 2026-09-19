import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from src.data.load_data import load_data


def load_clustering_data():
    """
    Loads credit data and selects key financial continuous features for borrower segmentation.
    """
    df = load_data()
    if df is None:
        raise FileNotFoundError("Could not load german.csv data.")

    features = [
        "Duration_of_Credit_monthly",
        "Credit_Amount",
        "Age_years",
        "Instalment_per_cent"
    ]
    X = df[features].dropna().copy()
    return df, X, features


def train_kmeans(X, n_clusters=3, random_state=42):
    """
    Standardizes features and trains KMeans model on borrower credit profiles.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=random_state
    )
    clusters = kmeans.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, clusters)

    return kmeans, scaler, X_scaled, clusters, silhouette


def get_cluster_centers(kmeans, scaler, features):
    """
    Returns cluster centers transformed back to original physical units (months, currency, years, %).
    """
    centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
    centers_df = pd.DataFrame(centers_orig, columns=features)
    centers_df.index.name = "Cluster"
    return centers_df


def plot_and_save_clusters(X, clusters, kmeans, scaler, features, save_path=None):
    """
    Generates a 2D scatter plot of borrower clusters (Credit Amount vs Duration)
    with marked centroids and dark fintech styling.
    """
    if save_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        save_path = os.path.join(base_dir, "App", "static", "charts", "kmeans_clusters.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(9, 6), facecolor="#080f1f")
    ax = plt.gca()
    ax.set_facecolor("#0b1224")

    # Colors for clusters
    colors = ["#38bdf8", "#f43f5e", "#34d399", "#fbbf24", "#a855f7"]
    cluster_labels = [f"Cluster {i}" for i in range(kmeans.n_clusters)]

    for i in range(kmeans.n_clusters):
        mask = clusters == i
        ax.scatter(
            X.loc[mask, "Duration_of_Credit_monthly"],
            X.loc[mask, "Credit_Amount"],
            c=colors[i % len(colors)],
            label=cluster_labels[i],
            alpha=0.65,
            s=45,
            edgecolors="none"
        )

    # Plot inverse-transformed centroids
    centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
    duration_idx = features.index("Duration_of_Credit_monthly")
    amount_idx = features.index("Credit_Amount")

    ax.scatter(
        centers_orig[:, duration_idx],
        centers_orig[:, amount_idx],
        c="#ffffff",
        marker="X",
        s=160,
        linewidths=2,
        edgecolors="#080f1f",
        label="Centroids",
        zorder=5
    )

    ax.set_title("K-Means Borrower Risk Segmentation (Credit Fair)", color="#f8fafc", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Loan Duration (Months)", color="#cbd5e1", fontsize=11)
    ax.set_ylabel("Credit Amount (DM / Currency)", color="#cbd5e1", fontsize=11)
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, linestyle="--", alpha=0.2, color="#38bdf8")

    legend = ax.legend(facecolor="#080f1f", edgecolor="#334155", labelcolor="#f8fafc", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, facecolor=plt.gcf().get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Cluster visualization saved to: {save_path}")


def save_model(kmeans, scaler):
    """
    Saves trained KMeans model and scaler to Models directory.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(base_dir, "Models")
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump({"model": kmeans, "scaler": scaler}, os.path.join(models_dir, "kmeans.pkl"))
    print(f"K-Means model saved to: {os.path.join(models_dir, 'kmeans.pkl')}")


def main():
    print("Loading Credit Fair German Credit dataset for K-Means clustering...")
    df, X, features = load_clustering_data()
    print(f"Selected Financial Features ({len(features)}): {features}")
    print(f"Total Samples for Clustering: {len(X)}")

    kmeans, scaler, X_scaled, clusters, silhouette = train_kmeans(X, n_clusters=3, random_state=42)
    print(f"\nK-Means Clustering Executed Successfully (k={kmeans.n_clusters})")
    print(f"Inertia (WCSS): {kmeans.inertia_:.2f}")
    print(f"Silhouette Score: {silhouette:.4f}")

    X_result = X.copy()
    X_result["Cluster"] = clusters
    if "Creditability" in df.columns:
        X_result["Creditability"] = df["Creditability"]

    print("\nBorrowers per Cluster:")
    print(X_result["Cluster"].value_counts().sort_index())

    centers_df = get_cluster_centers(kmeans, scaler, features)
    print("\nCluster Centroids (Original Units):")
    print(centers_df.round(2))

    plot_and_save_clusters(X, clusters, kmeans, scaler, features)
    save_model(kmeans, scaler)
    print("\nK-Means execution completed.")


if __name__ == "__main__":
    main()