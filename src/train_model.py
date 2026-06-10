"""Train, tune, evaluate, and save the final loan propensity model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    FEATURE_NAMES_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET_COL,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)
from src.evaluate_model import evaluate_classifier, metrics_to_dataframe


def load_train_test_data(
    train_path: str | Path = TRAIN_DATA_PATH,
    test_path: str | Path = TEST_DATA_PATH,
    target_col: str = TARGET_COL,
):
    """Load prepared train and test datasets."""
    train_path = Path(train_path)
    test_path = Path(test_path)

    if not train_path.exists():
        raise FileNotFoundError(f"Training dataset not found: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Testing dataset not found: {test_path}")

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    X_train = train_data.drop(columns=[target_col])
    y_train = train_data[target_col].astype(int)

    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col].astype(int)

    return X_train, X_test, y_train, y_test


def get_baseline_models(random_state: int = RANDOM_STATE) -> dict:
    """Return baseline models used during model development."""
    return {
        "Dummy Classifier - Majority Class": DummyClassifier(
            strategy="most_frequent",
            random_state=random_state,
        ),
        "Logistic Regression - Baseline": SklearnPipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(random_state=random_state, max_iter=1000),
                ),
            ]
        ),
        "Weighted Logistic Regression - Baseline": SklearnPipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        random_state=random_state,
                        max_iter=1000,
                    ),
                ),
            ]
        ),
        "Naive Bayes - Baseline": GaussianNB(),
        "SVM - Baseline": SklearnPipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", SVC(probability=True, random_state=random_state)),
            ]
        ),
        "Decision Tree - Baseline": DecisionTreeClassifier(random_state=random_state),
        "Random Forest - Baseline": RandomForestClassifier(random_state=random_state),
        "Hist Gradient Boosting - Baseline": HistGradientBoostingClassifier(
            random_state=random_state
        ),
        "AdaBoost - Baseline": AdaBoostClassifier(random_state=random_state),
    }


def train_and_evaluate_baselines(X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Train and evaluate all baseline models."""
    results = []

    for model_name, model in get_baseline_models().items():
        model.fit(X_train, y_train)
        results.append(
            evaluate_classifier(
                model,
                X_train,
                X_test,
                y_train,
                y_test,
                model_name,
                print_report=False,
            )
        )

    return metrics_to_dataframe(results)


def create_hybrid_resampled_data(X_train, y_train):
    """Apply SMOTE followed by random undersampling to training data."""
    resampling_pipeline = ImbPipeline(
        steps=[
            ("smote", SMOTE(sampling_strategy=0.3, random_state=RANDOM_STATE)),
            (
                "undersample",
                RandomUnderSampler(sampling_strategy=0.5, random_state=RANDOM_STATE),
            ),
        ]
    )

    return resampling_pipeline.fit_resample(X_train, y_train)


def tune_hist_gradient_boosting(
    X_train,
    y_train,
    n_iter: int = 50,
    scoring: str = "f1",
    cv: int = 5,
    random_state: int = RANDOM_STATE,
) -> RandomizedSearchCV:
    """Tune the Hist Gradient Boosting model using randomized search."""
    param_distributions = {
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_iter": [100, 200, 300, 500],
        "max_depth": [None, 3, 5, 10],
        "min_samples_leaf": [20, 30, 50],
        "l2_regularization": [0.0, 0.1, 0.5, 1.0],
    }

    random_search = RandomizedSearchCV(
        estimator=HistGradientBoostingClassifier(random_state=random_state),
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )

    random_search.fit(X_train, y_train)
    return random_search


def build_final_model(random_state: int = RANDOM_STATE) -> HistGradientBoostingClassifier:
    """Create the final tuned Hist Gradient Boosting model.

    These hyperparameters were selected from RandomizedSearchCV in the model
    development notebook.
    """
    return HistGradientBoostingClassifier(
        min_samples_leaf=20,
        max_iter=200,
        max_depth=5,
        learning_rate=0.2,
        l2_regularization=0.1,
        random_state=random_state,
    )


def save_artifacts(
    model,
    feature_names: list[str],
    model_path: str | Path = MODEL_PATH,
    feature_names_path: str | Path = FEATURE_NAMES_PATH,
) -> None:
    """Save the trained model and feature names."""
    model_path = Path(model_path)
    feature_names_path = Path(feature_names_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    feature_names_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path)
    joblib.dump(feature_names, feature_names_path)


def train_final_model(
    train_path: str | Path = TRAIN_DATA_PATH,
    test_path: str | Path = TEST_DATA_PATH,
    save_model: bool = True,
):
    """Train, evaluate, and optionally save the final production model."""
    X_train, X_test, y_train, y_test = load_train_test_data(train_path, test_path)

    final_model = build_final_model()
    final_model.fit(X_train, y_train)

    metrics = evaluate_classifier(
        final_model,
        X_train,
        X_test,
        y_train,
        y_test,
        "Hist Gradient Boosting - Tuned",
        print_report=True,
    )

    if save_model:
        save_artifacts(final_model, X_train.columns.tolist())

    return final_model, metrics


if __name__ == "__main__":
    model, final_metrics = train_final_model()
    print("\nFinal model metrics:")
    print(pd.DataFrame([final_metrics]))
    print(f"\nModel saved to: {MODEL_PATH}")
