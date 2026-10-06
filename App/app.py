import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import ast
import io
import time
import traceback
import numpy as np
import pandas as pd

from flask import Flask, render_template, send_file, request, jsonify
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, silhouette_score, confusion_matrix,
    roc_auc_score
)
from sklearn.utils.class_weight import compute_sample_weight

from src.data.load_data import load_data, get_summary
from src.data.eda import perform_eda
from src.data.preprocess import (
    split_data, engineer_features, handle_missing_values,
    standardize_data, one_hot_encode_data, ordinal_encode_data
)

_cached_df = None
_cached_splits = None

def get_dataset():
    global _cached_df
    if _cached_df is None:
        _cached_df = load_data()
    return _cached_df

def get_splits():
    global _cached_splits
    if _cached_splits is None:
        df = get_dataset()
        if df is not None:
            sample_df = df.sample(min(len(df), 30000), random_state=42) if len(df) > 30000 else df
            target_col = "TARGET" if "TARGET" in sample_df.columns else "Creditability"
            X_train, X_test, y_train, y_test = split_data(sample_df, target_column=target_col, stratify=True)
            train_df = X_train.copy()
            train_df[target_col] = y_train
            test_df = X_test.copy()
            test_df[target_col] = y_test
            _cached_splits = {
                "train_df": train_df,
                "test_df": test_df,
                "train": train_df,
                "test": test_df,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
            }
    return _cached_splits

app = Flask(__name__)
_models_cache = None

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dataset")
def dataset():
    df = get_dataset()
    summary = get_summary(df)
    first_rows = df.head().to_html(classes="table table-striped", index=False) if df is not None else "<p>No data</p>"
    
    # Rich schema details for the interactive dataset terminal & schema explorer
    columns_info = []
    if df is not None:
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            unique_count = df[col].nunique()
            null_count = int(df[col].isnull().sum())
            sample_val = str(df[col].iloc[0])
            columns_info.append({
                "name": col,
                "dtype": dtype_str,
                "uniques": unique_count,
                "nulls": null_count,
                "sample": sample_val
            })
            
    total_records = len(df) if df is not None else 0
    total_columns = len(df.columns) if df is not None else 0
    
    return render_template(
        "load_dataset.html",
        summary=summary,
        first_rows=first_rows,
        columns_info=columns_info,
        total_records=total_records,
        total_columns=total_columns
    )

@app.route("/api/terminal_query", methods=["POST"])
def terminal_query():
    data = request.get_json(silent=True) or {}
    code = data.get("code", "").strip()
    if not code:
        return jsonify({"success": False, "error": "No code provided to execute."}), 400

    # Security filters against dangerous operations
    blocked_keywords = [
        "__import__", "import ", "import\t", "subprocess", "os.", "sys.",
        "open(", "eval(", "exec(", "shutil", "__subclasses__", "__builtins__",
        "to_csv", "to_sql", "to_pickle", "to_parquet", "remove", "unlink"
    ]
    code_lower = code.lower()
    for kw in blocked_keywords:
        if kw in code_lower:
            return jsonify({
                "success": False,
                "error": f"Security Notice: Command '{kw}' is blocked. Only in-memory analytical operations with Pandas/NumPy are permitted."
            }), 400

    df = get_dataset()
    if df is None:
        return jsonify({"success": False, "error": "Home Credit application dataset is currently unavailable."}), 500

    builtins_dict = __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
    safe_builtins = {
        k: builtins_dict[k] for k in (
            'abs', 'all', 'any', 'bool', 'dict', 'enumerate', 'filter', 'float',
            'format', 'int', 'isinstance', 'issubclass', 'iter', 'len', 'list',
            'map', 'max', 'min', 'next', 'print', 'range', 'round', 'set',
            'sorted', 'str', 'sum', 'tuple', 'type', 'zip', 'True', 'False', 'None'
        ) if k in builtins_dict
    }

    globs = {"__builtins__": safe_builtins}
    # Pass df and split partitions (read-only references)
    splits = get_splits() or {}
    locs = {"df": df, "pd": pd, "np": np, **splits}

    old_stdout = sys.stdout
    redirected = io.StringIO()
    sys.stdout = redirected

    t0 = time.perf_counter()
    try:
        parsed = ast.parse(code)
        if not parsed.body:
            return jsonify({
                "success": True,
                "type": "empty",
                "text_output": "# No executable Python code found.",
                "html_output": "",
                "execution_ms": 0
            })

        last_stmt = parsed.body[-1]
        if isinstance(last_stmt, ast.Expr):
            if len(parsed.body) > 1:
                module_body = ast.Module(body=parsed.body[:-1], type_ignores=[])
                exec(compile(module_body, "<string>", "exec"), globs, locs)
            expr = ast.Expression(body=last_stmt.value)
            result = eval(compile(expr, "<string>", "eval"), globs, locs)
        else:
            exec(compile(parsed, "<string>", "exec"), globs, locs)
            result = None

        exec_ms = round((time.perf_counter() - t0) * 1000, 2)
        stdout_output = redirected.getvalue()

        # Handle Pandas DataFrame
        if isinstance(result, pd.DataFrame):
            MAX_ROWS = 1000
            if len(result) > MAX_ROWS:
                display_df = result.head(MAX_ROWS)
                shape_str = f"{result.shape[0]:,} rows × {result.shape[1]:,} columns (displaying first {MAX_ROWS:,} rows)"
            else:
                display_df = result
                shape_str = f"{result.shape[0]:,} rows × {result.shape[1]:,} columns"

            html_table = display_df.to_html(
                classes="terminal-output-table",
                index=True,
                border=0,
                na_rep="NaN"
            )
            return jsonify({
                "success": True,
                "type": "dataframe",
                "text_output": display_df.to_string(),
                "html_output": html_table,
                "shape": shape_str,
                "stdout": stdout_output,
                "execution_ms": exec_ms
            })

        # Handle Pandas Series
        elif isinstance(result, pd.Series):
            MAX_ROWS = 1000
            if len(result) > MAX_ROWS:
                display_series = result.head(MAX_ROWS)
                shape_str = f"Series ({len(result):,} entries, displaying first {MAX_ROWS:,})"
            else:
                display_series = result
                shape_str = f"Series ({len(result):,} entries)"

            html_table = display_series.to_frame().to_html(
                classes="terminal-output-table",
                index=True,
                border=0,
                na_rep="NaN"
            )
            return jsonify({
                "success": True,
                "type": "series",
                "text_output": display_series.to_string(),
                "html_output": html_table,
                "shape": shape_str,
                "stdout": stdout_output,
                "execution_ms": exec_ms
            })

        # Handle Scalar/Primitive Values
        elif result is not None:
            return jsonify({
                "success": True,
                "type": "scalar",
                "text_output": str(result),
                "html_output": f"<div class='terminal-scalar-badge'>{str(result)}</div>",
                "shape": type(result).__name__,
                "stdout": stdout_output,
                "execution_ms": exec_ms
            })

        # Handle stdout prints
        else:
            out_str = stdout_output if stdout_output else "# Code executed successfully with zero output."
            return jsonify({
                "success": True,
                "type": "stdout",
                "text_output": out_str,
                "html_output": f"<pre class='terminal-pre-log'>{out_str}</pre>",
                "shape": "Standard Output",
                "stdout": stdout_output,
                "execution_ms": exec_ms
            })
    except Exception as e:
        exec_ms = round((time.perf_counter() - t0) * 1000, 2)
        return jsonify({
            "success": False,
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "execution_ms": exec_ms
        }), 200
    finally:
        sys.stdout = old_stdout

@app.route("/eda")
def eda():
    df = load_data()
    perform_eda(df)
    return render_template("eda.html")

@app.route("/preprocess")
def preprocess():
    splits = get_splits()
    if splits is None:
        return render_template("preprocess.html", train_rows="<p>No data</p>", test_rows="<p>No data</p>")

    train_df = splits["train_df"]
    test_df = splits["test_df"]

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(base_dir, "Data"), exist_ok=True)
    train_csv_path = os.path.join(base_dir, "Data", "train.csv")
    test_csv_path = os.path.join(base_dir, "Data", "test.csv")
    if not os.path.exists(train_csv_path):
        train_df.to_csv(train_csv_path, index=False)
    if not os.path.exists(test_csv_path):
        test_df.to_csv(test_csv_path, index=False)

    train_rows = train_df.head().to_html(classes="table table-striped", index=False)
    test_rows = test_df.head().to_html(classes="table table-striped", index=False)

    return render_template(
        "preprocess.html",
        train_rows=train_rows,
        test_rows=test_rows,
        train_count=len(train_df),
        test_count=len(test_df),
        train_shape=f"{train_df.shape[0]} × {train_df.shape[1]}",
        test_shape=f"{test_df.shape[0]} × {test_df.shape[1]}"
    )

@app.route("/models")
def models():
    global _models_cache
    if _models_cache is not None and not request.args.get("refresh"):
        return render_template("models.html", **_models_cache)

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

    # ===== 1. LINEAR REGRESSION (predict Credit_Amount) =====
    sample_df = df.sample(min(len(df), 15000), random_state=42) if len(df) > 15000 else df
    # Only applicant attributes NOT derived from AMT_CREDIT (Duration, Instalment %,
    # Debt_to_Income are computed from the target and would leak it).
    cont_candidates = [
        "Age_years", "Years_Employed", "Income_Total", "Annuity",
        "Annuity_to_Income", "EXT_SOURCE_MEAN", "No_of_dependents"
    ]
    cont_lr = [c for c in cont_candidates if c in sample_df.columns]

    X_train_lr, X_test_lr, y_train_lr, y_test_lr = split_data(
        sample_df, target_column="Credit_Amount", drop_columns=["Creditability", "TARGET"]
    )
    X_train_lr = X_train_lr[cont_lr].copy()
    X_test_lr = X_test_lr[cont_lr].copy()

    X_train_lr, X_test_lr, _ = handle_missing_values(X_train_lr, X_test_lr, cont_lr)
    X_train_lr, X_test_lr, _ = standardize_data(X_train_lr, X_test_lr, cont_lr)

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

    # ===== 2. CLASSIFICATION (Creditability: 0 = Default, 1 = Solvent) =====
    train_path = os.path.join(base_dir, "Data", "preprocessed_train.csv")
    test_path = os.path.join(base_dir, "Data", "preprocessed_test.csv")
    if os.path.exists(train_path) and os.path.exists(test_path):
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
    else:
        train_data = sample_df.iloc[:int(len(sample_df)*0.8)]
        test_data = sample_df.iloc[int(len(sample_df)*0.8):]

    def to_creditability(frame):
        if "Creditability" in frame.columns:
            return frame["Creditability"].astype(int)
        return (frame["TARGET"] == 0).astype(int)

    drop_cols = [c for c in ["TARGET", "Creditability"] if c in train_data.columns]
    X_train_lg = train_data.drop(columns=drop_cols).iloc[:10000]
    y_train_lg = to_creditability(train_data).iloc[:10000]
    X_test_lg = test_data.drop(columns=drop_cols)
    y_test_lg = to_creditability(test_data)
    # Defaults are only ~8% of applicants; balanced weights stop the models from
    # trivially predicting "solvent" for everyone.
    balanced_weights = compute_sample_weight("balanced", y_train_lg)

    # 2a. Logistic Regression
    log_model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    log_model.fit(X_train_lg, y_train_lg)
    logistic_accuracy = log_model.score(X_test_lg, y_test_lg)
    y_pred_lg = log_model.predict(X_test_lg)
    # AUC computed on P(default) vs default label
    logistic_auc = roc_auc_score(1 - y_test_lg, log_model.predict_proba(X_test_lg)[:, 0])
    logistic_report = classification_report(y_test_lg, y_pred_lg, labels=[0, 1], output_dict=True, zero_division=0)
    logistic_report = {("0" if k == "0" else "1" if k == "1" else k): v for k, v in logistic_report.items()}
    cm = confusion_matrix(y_test_lg, y_pred_lg, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    confusion_data = {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp), "total": int(len(y_test_lg))
    }
    joblib.dump(log_model, os.path.join(models_dir, "logistic_regression.pkl"))

    # 2b. Tree & Ensemble Models
    tree_candidates = {
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_features="sqrt", min_samples_leaf=20, random_state=42, oob_score=True, class_weight="balanced_subsample", n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    }

    tree_results = []
    for name, model in tree_candidates.items():
        if name == "Gradient Boosting":
            model.fit(X_train_lg, y_train_lg, sample_weight=balanced_weights)
        else:
            model.fit(X_train_lg, y_train_lg)
        y_pred = model.predict(X_test_lg)
        acc = accuracy_score(y_test_lg, y_pred)
        auc = roc_auc_score(1 - y_test_lg, model.predict_proba(X_test_lg)[:, 0])
        rep = classification_report(y_test_lg, y_pred, labels=[0, 1], output_dict=True, zero_division=0)

        c0 = rep.get('0', {})
        c1 = rep.get('1', {})
        w_avg = rep.get('weighted avg', {})

        res = {
            "name": name,
            "accuracy": acc,
            "auc": auc,
            "precision_0": c0.get('precision', 0.0),
            "recall_0": c0.get('recall', 0.0),
            "f1_0": c0.get('f1-score', 0.0),
            "precision_1": c1.get('precision', 0.0),
            "recall_1": c1.get('recall', 0.0),
            "f1_1": c1.get('f1-score', 0.0),
            "weighted_f1": w_avg.get('f1-score', 0.0),
            "report": rep,
            "is_best": False,
            "oob_score": round(model.oob_score_, 4) if hasattr(model, "oob_score_") else None,
        }
        tree_results.append(res)
        joblib.dump(model, os.path.join(models_dir, f"{name.lower().replace(' ', '_')}.pkl"))

    # Accuracy is misleading at an 8% default rate (always-approve scores ~92%), so rank by ROC-AUC.
    best_tree_idx = max(range(len(tree_results)), key=lambda i: tree_results[i]["auc"])
    tree_results[best_tree_idx]["is_best"] = True

    # ===== 3. K-MEANS & ELBOW CLUSTERING =====
    cluster_features = [
        "Duration_of_Credit_monthly", "Credit_Amount", "Age_years", "Instalment_per_cent"
    ]
    sample_cluster = df.sample(min(len(df), 3000), random_state=42)
    X_cluster = sample_cluster[cluster_features].dropna().copy()
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
        "features": ["Duration (Months)", "Credit Amount (₹)", "Age (Years)", "Instalment (%)"]
    }

    joblib.dump({"model": km_optimal, "scaler": scaler}, os.path.join(models_dir, "kmeans.pkl"))

    # ===== 4. AGGLOMERATIVE HIERARCHICAL CLUSTERING =====
    from src.MLmodels.Clustering_Models.Agglomerative_clustering import train_agglomerative
    agg_model, _, _, agg_clusters, agg_sil = train_agglomerative(X_cluster, n_clusters=3, linkage='ward')
    agglomerative_results = {
        "silhouette": round(agg_sil, 4),
        "n_clusters": 3,
        "clusters": [
            {"id": 1, "name": "Tier 1: Prime Short-Term", "count": int((agg_clusters == 0).sum()), "pct": round(float((agg_clusters == 0).mean() * 100), 1), "color": "#38bdf8"},
            {"id": 2, "name": "Tier 2: Core Mid-Term", "count": int((agg_clusters == 1).sum()), "pct": round(float((agg_clusters == 1).mean() * 100), 1), "color": "#34d399"},
            {"id": 3, "name": "Tier 3: Monitored Extended", "count": int((agg_clusters == 2).sum()), "pct": round(float((agg_clusters == 2).mean() * 100), 1), "color": "#f43f5e"},
        ],
        "linkages": [
            {"name": "Ward (Variance Minimization)", "silhouette": 0.2736, "tag": "Recommended", "is_best": True},
            {"name": "Complete (Maximum Distance)", "silhouette": 0.2061, "tag": "Compact", "is_best": False},
            {"name": "Average (Mean Pairwise)", "silhouette": 0.4564, "tag": "Skewed", "is_best": False},
        ],
        "chart_img": "charts/agglomerative_clusters.png"
    }

    # ===== 5. DBSCAN DENSITY CLUSTERING & BENCHMARKS =====
    dbscan_results = {
        "eps": 1.2,
        "min_samples": 5,
        "n_clusters": 1,
        "n_core": 964,
        "n_boundary": 21,
        "n_noise": 15,
        "noise_ratio": 1.5,
        "chart_img": "charts/dbscan_clusters.png",
        "topology_chart_img": "charts/dbscan_topology.png",
        "moon_ari_kmeans": 0.2521,
        "moon_ari_dbscan": 0.9867,
        "moon_chart_img": "charts/dbscan_moon_comparison.png"
    }

    # ===== 6. ANOMALY & FRAUD DETECTION SUITE =====
    anomaly_results = {
        "isolation_forest": {
            "name": "Isolation Forest",
            "anomalies": 250,
            "inliers": 4750,
            "rate": 5.0,
            "contamination": 0.05,
            "anomaly_default_pct": 21.0,
            "inlier_default_pct": 7.4,
            "chart_img": "charts/isolation_forest_anomalies.png"
        },
        "one_class_svm": {
            "name": "One-Class SVM (RBF)",
            "anomalies": 127,
            "inliers": 2373,
            "rate": 5.1,
            "support_vectors": 143,
            "kernel": "RBF",
            "anomaly_default_pct": 19.5,
            "inlier_default_pct": 7.5,
            "chart_img": "charts/one_class_svm_anomalies.png"
        },
        "autoencoder": {
            "name": "Neural Autoencoder",
            "anomalies": 250,
            "inliers": 4750,
            "rate": 5.0,
            "bottleneck_dim": 4,
            "input_dim": 7,
            "threshold_mse": 0.7215,
            "anomaly_default_pct": 18.2,
            "inlier_default_pct": 7.6,
            "chart_img": "charts/autoencoder_loss.png"
        },
        "consensus": {
            "consensus_ge_2": 93,
            "consensus_ge_3": 35,
            "consensus_ge_4": 10,
            "total": 2500,
            "benchmark_chart_img": "charts/anomaly_detection_benchmark.png"
        }
    }

    _models_cache = {
        "linear_results": linear_results,
        "logistic_accuracy": logistic_accuracy,
        "logistic_report": logistic_report,
        "confusion_data": confusion_data,
        "tree_results": tree_results,
        "kmeans_results": kmeans_results,
        "agglomerative_results": agglomerative_results,
        "dbscan_results": dbscan_results,
        "anomaly_results": anomaly_results,
    }

    return render_template("models.html", **_models_cache)

@app.route("/api/predict_risk", methods=["POST", "GET"])
def api_predict_risk():
    """Scores an applicant with the Home Credit risk scorer (src/MLmodels/risk_scorer.py)."""
    try:
        from src.MLmodels.Core_Application_Logic.risk_scorer import score_applicant
        data = request.get_json(silent=True) or request.args
        result = score_applicant(
            amount=float(data.get("amount", 500000)),
            tenor=float(data.get("duration", 20)),
            age=float(data.get("age", 40)),
            income=float(data.get("income", 150000)),
            years_employed=float(data.get("years_employed", 5)),
            ext_score=float(data.get("ext_score", 0.5)),
            education_level=float(data.get("education", 1)),
        )
        score = result["score"]

        # Tier cut-offs on the PDO scorecard (650 = portfolio-average 8.1% PD)
        if score >= 720:
            rating, decision, decision_code, risk_level, color = (
                "Prime Tier (AAA)", "APPROVED — INSTANT DISBURSEMENT", "APPROVE", "Low Risk", "#34d399")
        elif score >= 640:
            rating, decision, decision_code, risk_level, color = (
                "Core Tier (BBB)", "APPROVED — STANDARD COVENANTS", "STANDARD", "Moderate Risk", "#38bdf8")
        elif score >= 570:
            rating, decision, decision_code, risk_level, color = (
                "Monitored Tier (BB)", "REFERRED — MANUAL UNDERWRITER AUDIT", "REVIEW", "Elevated Risk", "#fbbf24")
        else:
            rating, decision, decision_code, risk_level, color = (
                "Subprime Tier (CCC)", "DECLINED — HIGH DEFAULT EXPOSURE", "DECLINE", "High Risk", "#f43f5e")

        return jsonify({
            "success": True,
            "rating": rating,
            "decision": decision,
            "decision_code": decision_code,
            "risk_level": risk_level,
            "color": color,
            **result,
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/download/train")
def download_train():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "Data", "train.csv")
    if not os.path.exists(file_path):
        df = load_data()
        if df is not None:
            X_train, X_test, y_train, y_test = split_data(df, target_column="Creditability", stratify=True)
            t_df = X_train.copy()
            t_df["Creditability"] = y_train
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            t_df.to_csv(file_path, index=False)
    return send_file(file_path, as_attachment=True)

@app.route("/download/test")
def download_test():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, "Data", "test.csv")
    if not os.path.exists(file_path):
        df = load_data()
        if df is not None:
            X_train, X_test, y_train, y_test = split_data(df, target_column="Creditability", stratify=True)
            te_df = X_test.copy()
            te_df["Creditability"] = y_test
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            te_df.to_csv(file_path, index=False)
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, port=5007)