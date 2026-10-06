# Credit Fair — Institutional Credit Risk & Underwriting Engine

An executive-grade machine learning platform and web intelligence terminal for consumer credit risk scoring, portfolio risk tiering, and explainable loan underwriting.

---

## 🌟 Key Platform Features

- **Interactive Underwriting Simulator**: Real-time loan scoring console featuring dynamic SVG gauge meter, calibrated Probability of Default (PD), monthly repayment burden, and live SHAP-style factor attribution.
- **Dynamic 2D Constellation & Particle Wave Canvas**: Interactive physics-based neural particle mesh responding to cursor proximity with ambient financial liquidity wave.
- **Live Underwriting Ticker Marquee**: Continuous infinite streaming tape displaying simulated applicant audit evaluations with real-time tier classification.
- **3D Interactive Model Matrix**: Hover parallax tilt cards detailing performance metrics for 7 benchmarked models.
- **End-to-End Pipeline Visualization**: 6-stage architectural flow from raw data ingestion to automated model serialization.
- **Supervised & Unsupervised Modeling Suite**:
  - **Random Forest Ensemble** (100 Trees, Out-of-Bag Validation)
  - **Gradient Boosted Decision Trees** (High-precision default detection)
  - **Pruned Decision Trees** (Explainable white-box regulatory compliance)
  - **Logistic Regression** (Calibrated probability outputs & odds ratios)
  - **Ridge, Lasso & ElasticNet** (Continuous credit limit prediction with L1/L2 shrinkage)
  - **K-Means Clustering** (Elbow WCSS & Silhouette analysis discovering 3 borrower archetypes)
- **28-Point Dark Executive Visual Analytics & Diagnostic Suite**: High-resolution dark-themed visual suite spanning 7 category tiers (Risk & Target, Credit & Exposure, Demographics & Cohorts, Correlations & Covariance, Clustering & Topology, Tree Architectures, and Anomaly & Fraud Detection), featuring category filter tabs, live keyword search, and interactive fullscreen inspection lightboxes.
- **Interactive In-Browser Dataset REPL Terminal**: Sandboxed Python 3.13 / Pandas terminal console on the Dataset page allowing analysts to execute live exploratory code (e.g. `df.describe()`, `df.groupby()`, filtering) with sub-5ms latency, command history navigation (`↑`/`↓`), quick-snippet chips, and interactive Table vs. ASCII rendering.
- **High-Performance Architecture**: Module-level caching for instantaneous sub-10ms response times, debounced risk prediction API (`/api/predict_risk`), and sandboxed execution API (`/api/terminal_query`).

---

## 🏗️ Architecture & Project Structure

```
CreditFare2.0/
├── App/
│   ├── app.py                  # Core Flask server, routing, caching & API endpoints
│   ├── static/
│   │   ├── css/
│   │   │   └── styles.css      # Dark executive fintech theme & keyframe animations
│   │   ├── charts/             # Generated Matplotlib/Seaborn diagnostic visual plots
│   │   └── videos/
│   │       └── jp_morgan_hq_trimmed.mp4 # Ambient corporate background loop
│   └── templates/
│       ├── base.html           # Edge-to-edge frosted glass navigation & footer
│       ├── home.html           # Animated executive landing page & live simulator
│       ├── load_dataset.html   # Dataset preview & schema summary
│       ├── eda.html            # Exploratory data analysis chart gallery
│       ├── preprocess.html     # Feature engineering & train/test split manager
│       └── models.html         # In-depth model benchmarking & clustering suite
├── Data/
│   ├── application_train.csv   # Home Credit Default Risk dataset (307,511 records, 122 features)
│   ├── train.csv & test.csv    # Exportable stratified partitions
│   └── preprocessed_*.csv      # Enriched feature datasets
├── Models/                     # Serialized production models (*.pkl)
├── src/
│   ├── data/
│   │   ├── load_data.py        # Dataset loading & summary generator
│   │   ├── eda.py              # Visual univariate & bivariate exploratory analysis
│   │   └── preprocess.py       # Imputation, standardization, and encoding
│   └── MLmodels/               # Individual model training & diagnostic scripts
├── main.py                     # Primary entrypoint runner
└── requirements.txt            # Python dependencies
```

---

## 🚀 Quick Start

### 1. Prerequisites & Environment Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
python3 main.py
# or directly:
python3 App/app.py
```
Open your browser and navigate to:
**`http://127.0.0.1:5007/`**

---

## 📡 REST API Documentation

### Predict Credit Risk & Underwriting Tier
- **Endpoint**: `GET /api/predict_risk` or `POST /api/predict_risk`
- **Query Parameters / JSON Payload**:
  - `amount` (float): Requested credit in ₹ (e.g. `500000`)
  - `duration` (float): Loan duration in months (e.g. `24`)
  - `age` (float): Applicant age in years (e.g. `35`)
  - `income` (float): Annual income in ₹ (e.g. `180000`)
  - `years_employed` (float): Employment tenure in years (e.g. `5.0`)
  - `ext_score` (float): External bureau score 0-1 (e.g. `0.5`)
  - `education` (int): Education level (`0`: `Lower Sec`, `1`: `Secondary`, `2`: `Higher Ed`, `3`: `Degree`)

- **Sample Response**:
```json
{
  "success": true,
  "score": 785,
  "rating": "Prime Tier (AAA)",
  "decision": "APPROVED — INSTANT DISBURSEMENT",
  "decision_code": "APPROVE",
  "risk_level": "Low Risk",
  "default_prob_pct": 4.8,
  "monthly_burden": 20833.0,
  "color": "#34d399",
  "drivers": [
    { "factor": "External Bureau Score", "impact": 65, "positive": true },
    { "factor": "Debt-to-Income Ratio", "impact": 20, "positive": true }
  ]
}
```

---

## ⚖️ Compliance & Benchmark Integrity
- **Home Credit Benchmark**: 307,511 real-world application portfolios evaluated across 122 socio-economic, financial, and credit bureau dimensions.
- **Fair Lending Standard**: Transparent decision paths audited to prevent demographic skew or disparity.
- **License**: MIT License.
