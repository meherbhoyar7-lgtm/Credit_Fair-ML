import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
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
    Loads German Credit dataset and extracts continuous financial features for clustering.
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


def find_optimal_k(X_scaled, max_k=10, save_path=None):
    """
    Computes WCSS (Inertia) for k in 1..max_k and Silhouette scores for k in 2..max_k.
    Plots and saves the diagnostic Elbow & Silhouette charts.
    """
    wcss = []
    silhouette_scores = []
    k_range = list(range(1, max_k + 1))

    for k in k_range:
        model = KMeans(
            n_clusters=k,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=42
        )
        model.fit(X_scaled)
        wcss.append(float(model.inertia_))

        if k >= 2:
            score = silhouette_score(X_scaled, model.labels_)
            silhouette_scores.append(float(score))
        else:
            silhouette_scores.append(None)

    print("\nWCSS (Inertia) Values:")
    for k, val in zip(k_range, wcss):
        sil_str = f" | Silhouette: {silhouette_scores[k-1]:.4f}" if silhouette_scores[k-1] is not None else ""
        print(f"  K = {k:2d} -> WCSS = {val:10.2f}{sil_str}")

    # Determine recommended optimal K: highest silhouette score between k=2 and k=5
    valid_sil = [(k, silhouette_scores[k-1]) for k in range(2, min(6, max_k + 1))]
    optimal_k = max(valid_sil, key=lambda item: item[1])[0]
    print(f"\nRecommended Optimal K (Elbow/Silhouette Heuristic): K = {optimal_k}")

    # Generate visual plot
    if save_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        save_path = os.path.join(base_dir, "App", "static", "charts", "kmeans_elbow.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), facecolor="#080f1f")

    # Panel 1: Elbow Curve (WCSS)
    ax1.set_facecolor("#0b1224")
    ax1.plot(k_range, wcss, marker='o', color="#38bdf8", linewidth=2.5, markersize=7, label="WCSS")
    ax1.axvline(x=optimal_k, color="#f43f5e", linestyle="--", linewidth=1.8, label=f"Optimal K = {optimal_k}")
    ax1.set_xlabel("Number of Clusters (K)", color="#cbd5e1", fontsize=11)
    ax1.set_ylabel("Within-Cluster Sum of Squares (WCSS)", color="#cbd5e1", fontsize=11)
    ax1.set_title("Elbow Method for Optimal K", color="#f8fafc", fontsize=13, fontweight="bold")
    ax1.set_xticks(k_range)
    ax1.tick_params(colors="#94a3b8")
    ax1.grid(True, linestyle="--", alpha=0.2, color="#38bdf8")
    ax1.legend(facecolor="#080f1f", edgecolor="#334155", labelcolor="#f8fafc")

    # Panel 2: Silhouette Score Curve
    ax2.set_facecolor("#0b1224")
    k_sil = list(range(2, max_k + 1))
    sil_values = [s for s in silhouette_scores if s is not None]
    ax2.plot(k_sil, sil_values, marker='s', color="#34d399", linewidth=2.5, markersize=7, label="Silhouette Score")
    ax2.axvline(x=optimal_k, color="#f43f5e", linestyle="--", linewidth=1.8, label=f"Peak K = {optimal_k}")
    ax2.set_xlabel("Number of Clusters (K)", color="#cbd5e1", fontsize=11)
    ax2.set_ylabel("Silhouette Score", color="#cbd5e1", fontsize=11)
    ax2.set_title("Silhouette Analysis across Clusters", color="#f8fafc", fontsize=13, fontweight="bold")
    ax2.set_xticks(k_sil)
    ax2.tick_params(colors="#94a3b8")
    ax2.grid(True, linestyle="--", alpha=0.2, color="#34d399")
    ax2.legend(facecolor="#080f1f", edgecolor="#334155", labelcolor="#f8fafc")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Elbow diagnostic plot saved to: {save_path}")

    return wcss, silhouette_scores, optimal_k


def create_model(k=3):
    """
    Creates a KMeans model instance.
    """
    model = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=10,
        max_iter=300,
        random_state=42
    )
    return model


def train_model(model, X_scaled):
    """
    Trains KMeans model on scaled input data and returns labels.
    """
    labels = model.fit_predict(X_scaled)
    print(f"\nK-Means model trained successfully with K={model.n_clusters}")
    return model, labels


def evaluate_model(model, X_scaled, labels):
    """
    Evaluates clustering model quality via inertia and silhouette score.
    """
    inertia = float(model.inertia_)
    n_iter = int(model.n_iter_)
    silhouette = float(silhouette_score(X_scaled, labels))

    print("\n--- Clustering Evaluation ---")
    print(f"  Inertia (WCSS):         {inertia:.2f}")
    print(f"  Iterations to Converge: {n_iter}")
    print(f"  Silhouette Score:       {silhouette:.4f}")

    return {
        "inertia": inertia,
        "n_iter": n_iter,
        "silhouette": silhouette
    }


def display_clusters(X_df, labels, model, scaler, feature_names=None, save_path=None):
    """
    Generates and saves the cluster scatter chart.
    """
    if save_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        save_path = os.path.join(base_dir, "App", "static", "charts", "kmeans_elbow_clusters.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(9, 6), facecolor="#080f1f")
    ax = plt.gca()
    ax.set_facecolor("#0b1224")

    colors = ["#38bdf8", "#f43f5e", "#34d399", "#fbbf24", "#a855f7"]
    for i in range(model.n_clusters):
        mask = labels == i
        ax.scatter(
            X_df.loc[mask, "Duration_of_Credit_monthly"],
            X_df.loc[mask, "Credit_Amount"],
            c=colors[i % len(colors)],
            label=f"Segment {i}",
            alpha=0.65,
            s=45
        )

    # Centroids
    centers_orig = scaler.inverse_transform(model.cluster_centers_)
    ax.scatter(
        centers_orig[:, 0],
        centers_orig[:, 1],
        c="#ffffff",
        marker="X",
        s=160,
        linewidths=2,
        edgecolors="#080f1f",
        label="Centroids",
        zorder=5
    )

    ax.set_title("Optimal K-Means Borrower Segments (Credit Fair)", color="#f8fafc", fontsize=13, fontweight="bold")
    ax.set_xlabel("Duration of Credit (Months)", color="#cbd5e1", fontsize=11)
    ax.set_ylabel("Credit Amount (DM)", color="#cbd5e1", fontsize=11)
    ax.tick_params(colors="#94a3b8")
    ax.grid(True, linestyle="--", alpha=0.2, color="#38bdf8")
    ax.legend(facecolor="#080f1f", edgecolor="#334155", labelcolor="#f8fafc")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, facecolor=plt.gcf().get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"Segment cluster plot saved to: {save_path}")


def save_model(model, scaler=None):
    """
    Saves the KMeans elbow model to disk.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "Models", "kmeans_elbow.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler}, model_path)
    print(f"Model saved successfully to: {model_path}")


def main():
    print("=== Credit Fair — K-Means Optimal K & Elbow Analysis ===")
    df, X, features = load_clustering_data()
    print(f"Dataset Shape: {df.shape}")
    print(f"Clustering Features: {features}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 1. Elbow Method to find optimal K
    wcss, sil_scores, optimal_k = find_optimal_k(X_scaled, max_k=10)

    # 2. Train model using optimal K
    model = create_model(k=optimal_k)
    model, labels = train_model(model, X_scaled)

    # 3. Evaluate
    metrics = evaluate_model(model, X_scaled, labels)

    # 4. Display & Save Visualizations
    display_clusters(X, labels, model, scaler, feature_names=features)

    # 5. Save model
    save_model(model, scaler)

    print("\nK-Means Elbow analysis successfully completed.")


if __name__ == "__main__":
    main()