import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

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
    Creates a Gradient Boosting Classifier.
    """
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    return model


def train_model(model, X_train, y_train):
    """
    Trains the Gradient Boosting model.
    """
    model.fit(X_train, y_train)
    print("\nGradient Boosting Model Training Complete")
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


def save_model(model):
    """
    Saves the trained model to the Models directory.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "Models", "gradient_boosting.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print("\nModel saved successfully at:")
    print(model_path)


def main():
    try:
        train_data, test_data = load_preprocessed_data()
        print("\nLoaded preprocessed data.")
        print("Train data shape:", train_data.shape)
        print("Test data shape:", test_data.shape)
        X_train, Y_train, X_test, Y_test = split_features_target(train_data, test_data)
    except FileNotFoundError:
        print("\nPreprocessed files not found. Running preprocessing...")
        df = load_data()
        X_train, X_test, Y_train, Y_test = split_data(
            df,
            target_column="Creditability",
            stratify=True
        )
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
        X_train, X_test, _ = handle_missing_values(X_train, X_test, continuous_features)
        X_train, X_test, _ = one_hot_encode_data(X_train, X_test, one_hot_features)
        X_train, X_test, _ = ordinal_encode_data(X_train, X_test, ordinal_features)

    print("\nTrain features shape:", X_train.shape)
    model = create_model()
    print("\nGradient Boosting model created.")
    model = train_model(model, X_train, Y_train)
    evaluate_model(model, X_test, Y_test)
    save_model(model)
    print("\nGradient Boosting training completed.")


if __name__ == "__main__":
    main()
