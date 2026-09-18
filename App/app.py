import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, send_file
from src.data.load_data import load_data, get_summary
from src.data.eda import perform_eda
from src.data.preprocess import (
    split_data,
    engineer_features,
    identify_features,
    handle_missing_values,
    standardize_data,
    one_hot_encode_data,
    ordinal_encode_data
)

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, silhouette_score
import joblib

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dataset")
def dataset():
    df = load_data()
    summary = get_summary(df)
    first_rows = df.head().to_html(classes="table table-striped", index=False) if df is not None else "<p>No data</p>"
    return render_template("load_dataset.html", summary=summary, first_rows=first_rows)

@app.route("/eda")
def eda():
    df = load_data()
    perform_eda(df)
    return render_template("eda.html")

@app.route("/preprocess")
def preprocess():
    df = load_data()
    if df is None:
        return render_template("preprocess.html", train_rows="<p>No data</p>", test_rows="<p>No data</p>")

    X_train, X_test, y_train, y_test = split_data(df, target_column="Creditability", stratify=True)

    # Save as simple train/test for download
    import pandas as pd
    train_df = X_train.copy()
    train_df["Creditability"] = y_train
    test_df = X_test.copy()
    test_df["Creditability"] = y_test

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_df.to_csv(os.path.join(base_dir, "Data", "train.csv"), index=False)
    test_df.to_csv(os.path.join(base_dir, "Data", "test.csv"), index=False)

    train_rows = train_df.head().to_html(classes="table table-striped", index=False)
    test_rows = test_df.head().to_html(classes="table table-striped", index=False)

    return render_template("preprocess.html", train_rows=train_rows, test_rows=test_rows)

@app.route("/models")
def models():
    df = load_data()
    if df is None:
        return render_template(
            "models.html",
            linear_results=[],
            logistic_accuracy=0,
            logistic_report={},
            tree_results=[],
            kmeans_results={}
        )

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "Models")
    os.makedirs(models_dir, exist_ok=True)

    # ===== PREPROCESSING FEATURES =====
    continuous_features = [
        "Duration_of_Credit_monthly", "Credit_Amount", "Instalment_per_cent",
        "Duration_in_Current_address", "Age_years", "No_of_Credits_at_this_Bank",
        "No_of_dependents",
    ]
    one_hot_features = [
        "Purpose", "Sex_Marital_Status", "Guarantors",
        "Type_of_apartment", "Telephone", "Foreign_Worker",
    ]
    ordinal_features = [
        "Account_Balance", "Payment_Status_of_Previous_Credit",
        "Value_Savings_Stocks", "Length_of_current_employment",
        "Most_valuable_available_asset", "Concurrent_Credits", "Occupation",
    ]

    # ===== 1. LINEAR REGRESSION (predict Credit_Amount) =====
    X_train_lr, X_test_lr, y_train_lr, y_test_lr = split_data(
        df, target_column="Credit_Amount", drop_columns=["Creditability"]
    )
    X_train_lr = engineer_features(X_train_lr)
    X_test_lr = engineer_features(X_test_lr)

    cont_lr = [
        "Duration_of_Credit_monthly", "Instalment_per_cent",
        "Duration_in_Current_address", "Age_years", "No_of_Credits_at_this_Bank",
        "No_of_dependents", "Duration_Years", "Log_Duration", "Duration_Squared",
        "Instalment_Commitment_Index", "Age_to_Duration_Ratio", "Log_Age",
        "Credits_per_Age", "Dependents_per_Adult", "Wealth_Index",
        "Employment_Payment_Score", "Financial_Burden_Ratio"
    ]

    X_train_lr, X_test_lr, _ = handle_missing_values(X_train_lr, X_test_lr, cont_lr)
    X_train_lr, X_test_lr, _ = standardize_data(X_train_lr, X_test_lr, cont_lr)
    X_train_lr, X_test_lr, _ = one_hot_encode_data(X_train_lr, X_test_lr, one_hot_features)
    X_train_lr, X_test_lr, _ = ordinal_encode_data(X_train_lr, X_test_lr, ordinal_features)

    lr_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1, max_iter=20000),
        "ElasticNet Regression": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000),
    }

    linear_results = []
    for name, model in lr_models.items():
        model.fit(X_train_lr, y_train_lr)
        y_pred = model.predict(X_test_lr)
        metrics = {
            "name": name,
            "mae": mean_absolute_error(y_test_lr, y_pred),
            "mse": mean_squared_error(y_test_lr, y_pred),
            "rmse": mean_squared_error(y_test_lr, y_pred) ** 0.5,
            "r2": r2_score(y_test_lr, y_pred),
            "is_best": False,
        }
        linear_results.append(metrics)

    best_idx = max(range(len(linear_results)), key=lambda i: linear_results[i]["r2"])
    linear_results[best_idx]["is_best"] = True

    # ===== 2. CLASSIFICATION DATASET (predict Creditability) =====
    X_train_lg, X_test_lg, y_train_lg, y_test_lg = split_data(
        df, target_column="Creditability", stratify=True
    )

    X_train_lg, X_test_lg, _ = handle_missing_values(X_train_lg, X_test_lg, continuous_features)
    X_train_lg, X_test_lg, _ = standardize_data(X_train_lg, X_test_lg, continuous_features)
    X_train_lg, X_test_lg, _ = one_hot_encode_data(X_train_lg, X_test_lg, one_hot_features)
    X_train_lg, X_test_lg, _ = ordinal_encode_data(X_train_lg, X_test_lg, ordinal_features)

    # 2a. Logistic Regression
    from sklearn.metrics import confusion_matrix
    log_model = LogisticRegression(max_iter=1000, random_state=42)
    log_model.fit(X_train_lg, y_train_lg)
    logistic_accuracy = log_model.score(X_test_lg, y_test_lg)
    y_pred_lg = log_model.predict(X_test_lg)
    logistic_report = classification_report(y_test_lg, y_pred_lg, output_dict=True)
    cm = confusion_matrix(y_test_lg, y_pred_lg)
    tn, fp, fn, tp = cm.ravel()
    confusion_data = {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "total": int(len(y_test_lg))
    }
    joblib.dump(log_model, os.path.join(models_dir, "logistic_regression.pkl"))

    # 2b. Tree & Ensemble Models
    tree_candidates = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_features="sqrt", random_state=42, oob_score=True),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    }

    tree_results = []
    for name, model in tree_candidates.items():
        model.fit(X_train_lg, y_train_lg)
        y_pred = model.predict(X_test_lg)
        acc = accuracy_score(y_test_lg, y_pred)
        rep = classification_report(y_test_lg, y_pred, output_dict=True)

        res = {
            "name": name,
            "accuracy": acc,
            "precision_0": rep['0']['precision'],
            "recall_0": rep['0']['recall'],
            "f1_0": rep['0']['f1-score'],
            "precision_1": rep['1']['precision'],
            "recall_1": rep['1']['recall'],
            "f1_1": rep['1']['f1-score'],
            "weighted_f1": rep['weighted avg']['f1-score'],
            "report": rep,
            "is_best": False,
            "oob_score": round(model.oob_score_, 4) if hasattr(model, "oob_score_") else None,
        }
        tree_results.append(res)
        joblib.dump(model, os.path.join(models_dir, f"{name.lower().replace(' ', '_')}.pkl"))

    best_tree_idx = max(range(len(tree_results)), key=lambda i: tree_results[i]["accuracy"])
    tree_results[best_tree_idx]["is_best"] = True

    # ===== 3. K-MEANS & ELBOW CLUSTERING =====
    cluster_features = [
        "Duration_of_Credit_monthly", "Credit_Amount", "Age_years", "Instalment_per_cent"
    ]
    X_cluster = df[cluster_features].dropna().copy()
    scaler = StandardScaler()
    X_cluster_scaled = scaler.fit_transform(X_cluster)

    # Compute Elbow curve (WCSS) & Silhouette scores for K=1..10
    elbow_data = []
    for k in range(1, 11):
        km_temp = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=300, random_state=42)
        km_temp.fit(X_cluster_scaled)
        sil = float(silhouette_score(X_cluster_scaled, km_temp.labels_)) if k >= 2 else None
        elbow_data.append({
            "k": k,
            "wcss": round(float(km_temp.inertia_), 1),
            "silhouette": round(sil, 4) if sil is not None else None
        })

    # Fit 3-cluster customer risk segmentation
    km_optimal = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=42)
    clusters = km_optimal.fit_predict(X_cluster_scaled)
    sil_score = float(silhouette_score(X_cluster_scaled, clusters))
    inertia_score = float(km_optimal.inertia_)

    centers_orig = scaler.inverse_transform(km_optimal.cluster_centers_)

    # Create descriptive cluster profiles
    cluster_df = X_cluster.copy()
    cluster_df["cluster"] = clusters
    if "Creditability" in df.columns:
        cluster_df["Creditability"] = df["Creditability"]

    cluster_profiles = []
    titles = [
        {"name": "Low Exposure / Short Term", "tag": "Prime Tier", "color": "#38bdf8"},
        {"name": "Moderate Term / Balanced Burden", "tag": "Core Tier", "color": "#34d399"},
        {"name": "Extended Term / High Exposure", "tag": "Monitored Tier", "color": "#f43f5e"},
    ]

    for i in range(3):
        mask = cluster_df["cluster"] == i
        count = int(mask.sum())
        pct = round((count / len(cluster_df)) * 100, 1)
        good_pct = round(float(cluster_df.loc[mask, "Creditability"].mean() * 100), 1) if "Creditability" in cluster_df else 70.0
        
        cluster_profiles.append({
            "id": i,
            "name": titles[i]["name"],
            "tag": titles[i]["tag"],
            "color": titles[i]["color"],
            "count": count,
            "percent": pct,
            "avg_duration": round(float(centers_orig[i][0]), 1),
            "avg_amount": round(float(centers_orig[i][1]), 0),
            "avg_age": round(float(centers_orig[i][2]), 1),
            "avg_instalment": round(float(centers_orig[i][3]), 2),
            "good_credit_pct": good_pct,
        })

    # Sort cluster profiles by average credit amount
    cluster_profiles.sort(key=lambda c: c["avg_amount"])
    # Re-assign IDs for clean presentation
    for idx, cp in enumerate(cluster_profiles):
        cp["display_id"] = idx + 1

    kmeans_results = {
        "inertia": round(inertia_score, 1),
        "silhouette": round(sil_score, 4),
        "optimal_k": 3,
        "elbow_data": elbow_data,
        "cluster_profiles": cluster_profiles,
        "features": ["Duration (Months)", "Credit Amount (DM)", "Age (Years)", "Instalment (%)"]
    }

    joblib.dump({"model": km_optimal, "scaler": scaler}, os.path.join(models_dir, "kmeans.pkl"))

    return render_template(
        "models.html",
        linear_results=linear_results,
        logistic_accuracy=logistic_accuracy,
        logistic_report=logistic_report,
        confusion_data=confusion_data,
        tree_results=tree_results,
        kmeans_results=kmeans_results,
    )

@app.route("/download/train")
def download_train():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "Data", "train.csv")
    return send_file(file_path, as_attachment=True)

@app.route("/download/test")
def download_test():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "Data", "test.csv")
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, port=5007)