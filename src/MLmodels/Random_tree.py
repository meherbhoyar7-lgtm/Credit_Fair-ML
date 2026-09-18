import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

from src.data.load_data import load_data
from src.data.preprocess import (
    split_data,
    identify_features,
    handle_missing_values,
    one_hot_encode_data,
    ordinal_encode_data,
)


def load_preprocessed_data():
    """
    Loads preprocessed train and test CSV files.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(base_dir, "Data", "preprocessed_train.csv")
    test_path = os.path.join(base_dir, "Data", "preprocessed_test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def split_features_target(train_data, test_data, target_column="Creditability"):
    """
    Splits preprocessed data into features and target.
    """
    X_train = train_data.drop(columns=target_column)
    Y_train = train_data[target_column]
    X_test = test_data.drop(columns=target_column)
    Y_test = test_data[target_column]
    return X_train, Y_train, X_test, Y_test


def create_model():
    """
    Creates a Random Forest Classifier.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        random_state=42,
        oob_score=True,
    )
    return model


def train_model(model, X_train, y_train):
    """
    Trains the Random Forest model.
    """
    model.fit(X_train, y_train)
    print("\nRandom Forest Model Training Complete")
    if hasattr(model, "oob_score_"):
        print("OOB Score:", round(model.oob_score_, 4))
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model and prints accuracy and classification report.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("\nTest Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Bad Credit", "Good Credit"]))
    return y_pred


def display_model(model, feature_names):
    """
    Visualizes the first tree from the Random Forest ensemble and saves to static/charts.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    chart_path = os.path.join(base_dir, "App", "static", "charts", "random_forest_tree.png")
    os.makedirs(os.path.dirname(chart_path), exist_ok=True)

    plt.figure(figsize=(24, 12))
    first_tree = model.estimators_[0]

    tree.plot_tree(
        first_tree,
        feature_names=feature_names,
        class_names=["Bad Credit", "Good Credit"],
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=3,  # Limits plotted depth so the visual remains readable
    )
    plt.title("Random Forest (Tree 1 of 100) — Credit Risk Prediction (Credit Fair)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nRandom forest tree plot saved to: {chart_path}")


def save_model(model):
    """
    Saves the trained model to the Models directory.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "Models", "random_forest.pkl")
    joblib.dump(model, model_path)
    print("\nModel saved successfully at:")
    print(model_path)


def main():
    # --- Option 1: Use preprocessed CSV files ---
    # Requires running preprocess.py first to generate the CSV files
    try:
        train_data, test_data = load_preprocessed_data()
        print("\nLoaded preprocessed data.")
        print("Train data shape:", train_data.shape)
        print("Test data shape:", test_data.shape)

        X_train, Y_train, X_test, Y_test = split_features_target(train_data, test_data)

    except FileNotFoundError:
        # --- Option 2: Preprocess from scratch ---
        print("\nPreprocessed files not found. Running preprocessing...")
        df = load_data()
        print("Original Dataset Shape:", df.shape)

        X_train, X_test, Y_train, Y_test = split_data(
            df,
            target_column="Creditability",
            stratify=True
        )

        numerical_features, categorical_features = identify_features(X_train)

        continuous_features = [
            "Duration_of_Credit_monthly",
            "Credit_Amount",
            "Instalment_per_cent",
            "Duration_in_Current_address",
            "Age_years",
            "No_of_Credits_at_this_Bank",
            "No_of_dependents",
        ]

        one_hot_features = [
            "Purpose",
            "Sex_Marital_Status",
            "Guarantors",
            "Type_of_apartment",
            "Telephone",
            "Foreign_Worker",
        ]

        ordinal_features = [
            "Account_Balance",
            "Payment_Status_of_Previous_Credit",
            "Value_Savings_Stocks",
            "Length_of_current_employment",
            "Most_valuable_available_asset",
            "Concurrent_Credits",
            "Occupation",
        ]

        X_train, X_test, imputer = handle_missing_values(X_train, X_test, continuous_features)
        print("Missing values handled.")

        X_train, X_test, one_hot_encoder = one_hot_encode_data(X_train, X_test, one_hot_features)
        print("One-hot encoding completed.")

        X_train, X_test, ordinal_encoder = ordinal_encode_data(X_train, X_test, ordinal_features)
        print("Ordinal encoding completed.")

    print("\nTrain features shape:", X_train.shape)

    # Create and train model
    model = create_model()
    print("\nRandom Forest model created.")

    model = train_model(model, X_train, Y_train)

    # Evaluate
    evaluate_model(model, X_test, Y_test)

    # Visualize the first tree of the trained Random Forest
    display_model(model, feature_names=list(X_train.columns))

    # Save
    save_model(model)

    print("\nModel training completed.")


if __name__ == "__main__":
    main()