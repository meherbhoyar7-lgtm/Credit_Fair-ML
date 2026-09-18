import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from src.data.load_data import load_data
from src.data.preprocess import (
    split_data,
    engineer_features,
    identify_features,
    handle_missing_values,
    standardize_data,
    one_hot_encode_data,
    ordinal_encode_data
)


def create_models():
    """
    Creates a dictionary of linear regression models.
    """
    models = {
        "LinearRegression": LinearRegression(),
        "RidgeRegression": Ridge(alpha=1.0),
        "LassoRegression": Lasso(alpha=0.1, max_iter=20000),
        "ElasticNetRegression": ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000),
    }
    return models


def train_models(models, X_train, y_train):
    """
    Trains all models on the training data.
    """
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    return trained_models


def predict(model, X_test):
    """
    Makes predictions using a trained model.
    """
    y_pred = model.predict(X_test)
    return y_pred


def evaluate_model(y_test, y_pred):
    """
    Evaluates model performance using MAE, MSE, RMSE, and R2 metrics.
    """
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }


def main():
    df = load_data()

    print("Original Dataset Shape:")
    print(df.shape)

    # For regression, predict Credit_Amount as the continuous target
    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column="Credit_Amount",
        drop_columns=["Creditability"]
    )
    print("\nTraining Dataset Shape:")
    print(X_train.shape)
    print("\nTest Dataset Shape:")
    print(X_test.shape)

    # Engineer new domain-specific, interaction, and non-linear features
    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)
    print("\nFeature engineering completed. New shapes:")
    print("Train:", X_train.shape, "| Test:", X_test.shape)

    numerical_features, categorical_features = identify_features(X_train)
    print("\nNumerical Features:", numerical_features)
    print("Categorical Features:", categorical_features)

    # Continuous features for standardization (original + new engineered features)
    continuous_features = [
        # --- Original continuous features ---
        "Duration_of_Credit_monthly",
        "Instalment_per_cent",
        "Duration_in_Current_address",
        "Age_years",
        "No_of_Credits_at_this_Bank",
        "No_of_dependents",
        # --- New Engineered features ---
        "Duration_Years",               # Loan term scaled to years (Duration / 12)
        "Log_Duration",                # Log-transformed duration (reduces skew)
        "Duration_Squared",            # Polynomial term capturing non-linear duration risk
        "Instalment_Commitment_Index",  # Total instalment commitment factor (Duration * Instalment %)
        "Age_to_Duration_Ratio",        # Borrower age relative to loan term length
        "Log_Age",                     # Log-transformed borrower age
        "Credits_per_Age",             # Credit application velocity across adult life
        "Dependents_per_Adult",         # Financial dependent burden ratio relative to age
        "Wealth_Index",                # Composite score (Savings + Asset Tier + Account Balance)
        "Employment_Payment_Score",     # Job stability * past repayment record interaction
        "Financial_Burden_Ratio",       # Dependent instalment burden relative to savings
    ]

    # Nominal features → one-hot encode
    one_hot_features = [
        "Purpose",
        "Sex_Marital_Status",
        "Guarantors",
        "Type_of_apartment",
        "Telephone",
        "Foreign_Worker",
    ]

    # Ordinal features → ordinal encode
    ordinal_features = [
        "Account_Balance",
        "Payment_Status_of_Previous_Credit",
        "Value_Savings_Stocks",
        "Length_of_current_employment",
        "Most_valuable_available_asset",
        "Concurrent_Credits",
        "Occupation",
    ]

    # Handle missing values
    X_train, X_test, imputer = handle_missing_values(
        X_train, X_test, continuous_features
    )
    print("\nMissing values handled.")

    # Standardize continuous features
    X_train, X_test, scaler = standardize_data(
        X_train, X_test, continuous_features
    )
    print("Standardization completed.")

    # One-hot encode nominal features
    X_train, X_test, one_hot_encoder = one_hot_encode_data(
        X_train, X_test, one_hot_features
    )
    print("One-hot encoding completed.")

    # Ordinal encode ordered features
    X_train, X_test, ordinal_encoder = ordinal_encode_data(
        X_train, X_test, ordinal_features
    )
    print("Ordinal encoding completed.")
    print(f"\nFinal feature count after encoding: {X_train.shape[1]}")

    # Create and train models
    models = create_models()
    trained_models = train_models(models, X_train, y_train)
    print("\nModel training completed.")

    # Evaluate all models
    for name, model in trained_models.items():
        y_pred = predict(model, X_test)
        metrics = evaluate_model(y_test, y_pred)
        print(f"\n--- {name} ---")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  MSE:  {metrics['mse']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  R2:   {metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
