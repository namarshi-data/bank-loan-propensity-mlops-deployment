"""Data Preparation and train-test split utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    CLEANED_DATA_PATH,
    ID_COL,
    LOG_FEATURES,
    MODEL_FEATURES,
    RANDOM_STATE,
    TARGET_COL,
    TEST_DATA_PATH,
    TEST_SIZE,
    TRAIN_DATA_PATH,
    ZIP_COL,
)


def load_cleaned_data(file_path: str | Path = CLEANED_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned customer dataset."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {file_path}")

    return pd.read_csv(file_path)


def prepare_modeling_dataset(
    data: pd.DataFrame,
    drop_identifier: bool = True,
    drop_zipcode: bool = True,
) -> pd.DataFrame:
    """Prepare cleaned data for model training.

    Drops non-predictive identifier and high-cardinality ZipCode columns when present.
    """
    modeling_data = data.copy()

    columns_to_drop = []

    if drop_identifier and ID_COL in modeling_data.columns:
        columns_to_drop.append(ID_COL)

    if drop_zipcode and ZIP_COL in modeling_data.columns:
        columns_to_drop.append(ZIP_COL)

    if columns_to_drop:
        modeling_data = modeling_data.drop(columns=columns_to_drop)

    if TARGET_COL not in modeling_data.columns:
        raise ValueError(f"Target column not found: {TARGET_COL}")

    missing_features = [
        col for col in MODEL_FEATURES
        if col not in modeling_data.columns
    ]

    if missing_features:
        raise ValueError(f"Missing modeling features: {missing_features}")

    modeling_data[TARGET_COL] = modeling_data[TARGET_COL].astype(int)

    return modeling_data[MODEL_FEATURES + [TARGET_COL]].copy()


def split_features_target(
    data: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictors and target variable."""
    if target_col not in data.columns:
        raise ValueError(f"Target column not found: {target_col}")

    X = data.drop(columns=[target_col]).copy()
    y = data[target_col].copy()

    return X, y


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train-test split."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )


def create_alternative_feature_sets(
    feature_names: list[str],
) -> dict[str, list[str]]:
    """Create feature sets for multicollinearity experiments."""
    return {
        "all_features": feature_names,
        "no_age": [col for col in feature_names if col != "Age"],
        "no_customersince": [col for col in feature_names if col != "CustomerSince"],
    }


def apply_log_transform(
    X: pd.DataFrame,
    log_features: list[str] | None = None,
) -> pd.DataFrame:
    """Apply log1p transformation to selected positively skewed features."""
    transformed_data = X.copy()
    log_features = log_features or LOG_FEATURES

    for col in log_features:
        if col not in transformed_data.columns:
            raise ValueError(f"Log feature not found: {col}")

        if (transformed_data[col] < 0).any():
            raise ValueError(
                f"Negative values found in {col}; log1p transformation is not valid."
            )

        transformed_data[col] = np.log1p(transformed_data[col])

    return transformed_data


def save_train_test_data(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    train_path: str | Path = TRAIN_DATA_PATH,
    test_path: str | Path = TEST_DATA_PATH,
    target_col: str = TARGET_COL,
) -> None:
    """Save prepared train and test datasets."""
    train_data = X_train.copy()
    train_data[target_col] = y_train.values

    test_data = X_test.copy()
    test_data[target_col] = y_test.values

    train_path = Path(train_path)
    test_path = Path(test_path)

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)


def run_data_preparation(
    cleaned_data_path: str | Path = CLEANED_DATA_PATH,
    train_output_path: str | Path = TRAIN_DATA_PATH,
    test_output_path: str | Path = TEST_DATA_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Run feature preparation and save train-test datasets."""
    data = load_cleaned_data(cleaned_data_path)
    modeling_data = prepare_modeling_dataset(data)
    X, y = split_features_target(modeling_data)
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)

    save_train_test_data(
        X_train,
        X_test,
        y_train,
        y_test,
        train_output_path,
        test_output_path,
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = run_data_preparation()
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape : {X_test.shape}")
