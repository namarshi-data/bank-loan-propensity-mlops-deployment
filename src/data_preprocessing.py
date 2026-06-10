"""Data loading, merging, cleaning, and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import (
    BINARY_COLS,
    CLEANED_DATA_PATH,
    ID_COL,
    MERGED_DATA_PATH,
    ORDINAL_COLS,
    RAW_DATA_1,
    RAW_DATA_2,
    TARGET_COL,
)


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    return pd.read_csv(file_path)


def validate_required_columns(
    data: pd.DataFrame,
    required_columns: Iterable[str],
    dataset_name: str = "dataset",
) -> None:
    """Validate that required columns are present in a DataFrame."""
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise ValueError(
            f"Missing required columns in {dataset_name}: {missing_columns}"
        )


def merge_customer_datasets(
    data1: pd.DataFrame,
    data2: pd.DataFrame,
    merge_key: str = ID_COL,
) -> pd.DataFrame:
    """Merge Data1 and Data2 using the unique customer identifier."""
    validate_required_columns(data1, [merge_key], "Data1")
    validate_required_columns(data2, [merge_key], "Data2")

    if data1[merge_key].duplicated().any():
        raise ValueError("Duplicate customer IDs found in Data1.")

    if data2[merge_key].duplicated().any():
        raise ValueError("Duplicate customer IDs found in Data2.")

    merged_data = pd.merge(
        data1,
        data2,
        on=merge_key,
        how="inner",
        validate="one_to_one",
    )

    if len(merged_data) != min(len(data1), len(data2)):
        raise ValueError(
            "Merged row count is lower than expected. Some customer IDs may not match."
        )

    return merged_data


def clean_customer_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean the merged customer dataset for analysis and modeling.

    Cleaning steps:
    - Remove records with missing target values.
    - Convert target variable to integer.
    - Convert ordinal variables to category.
    - Convert binary variables to compact integer type.
    - Validate duplicate records and duplicate customer IDs.
    """
    cleaned_data = data.copy()

    validate_required_columns(
        cleaned_data,
        [TARGET_COL, ID_COL, *ORDINAL_COLS, *BINARY_COLS],
        "customer data",
    )

    cleaned_data = cleaned_data.dropna(subset=[TARGET_COL]).copy()
    cleaned_data[TARGET_COL] = cleaned_data[TARGET_COL].astype(int)

    if cleaned_data.duplicated().any():
        raise ValueError("Duplicate records found after cleaning.")

    if cleaned_data[ID_COL].duplicated().any():
        raise ValueError("Duplicate customer IDs found after cleaning.")

    for col in ORDINAL_COLS:
        cleaned_data[col] = cleaned_data[col].astype("category")

    for col in BINARY_COLS:
        cleaned_data[col] = cleaned_data[col].astype("int8")

    return cleaned_data


def get_data_quality_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return a compact data quality summary for each column."""
    return pd.DataFrame(
        {
            "Feature": data.columns,
            "Data Type": data.dtypes.astype(str).values,
            "Non-Null Count": data.notna().sum().values,
            "Missing Values": data.isna().sum().values,
            "Unique Values": data.nunique(dropna=True).values,
        }
    )


def run_data_preprocessing(
    raw_data_1_path: str | Path = RAW_DATA_1,
    raw_data_2_path: str | Path = RAW_DATA_2,
    merged_output_path: str | Path = MERGED_DATA_PATH,
    cleaned_output_path: str | Path = CLEANED_DATA_PATH,
) -> pd.DataFrame:
    """Run the full data preprocessing workflow and save outputs."""
    data1 = load_dataset(raw_data_1_path)
    data2 = load_dataset(raw_data_2_path)

    merged_data = merge_customer_datasets(data1, data2)

    merged_output_path = Path(merged_output_path)
    merged_output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_data.to_csv(merged_output_path, index=False)

    cleaned_data = clean_customer_data(merged_data)

    cleaned_output_path = Path(cleaned_output_path)
    cleaned_output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_data.to_csv(cleaned_output_path, index=False)

    return cleaned_data


if __name__ == "__main__":
    final_data = run_data_preprocessing()
    print(f"Cleaned dataset created successfully with shape: {final_data.shape}")
