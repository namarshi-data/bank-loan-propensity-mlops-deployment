"""Project configuration and reusable constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATA_1 = RAW_DATA_DIR / "Data1.csv"
RAW_DATA_2 = RAW_DATA_DIR / "Data2.csv"
MERGED_DATA_PATH = INTERIM_DATA_DIR / "01_merged_customer_data.csv"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "02_cleaned_customer_data.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "03_train_data.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "03_test_data.csv"

MODEL_PATH = MODEL_DIR / "hist_gradient_boosting_model.joblib"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.joblib"

TARGET_COL = "LoanOnCard"
ID_COL = "ID"
ZIP_COL = "ZipCode"

NUMERICAL_COLS = [
    "Age",
    "CustomerSince",
    "HighestSpend",
    "MonthlyAverageSpend",
    "Mortgage",
]

ORDINAL_COLS = [
    "Level",
    "HiddenScore",
]

BINARY_COLS = [
    "Security",
    "FixedDepositAccount",
    "InternetBanking",
    "CreditCard",
]

MODEL_FEATURES = NUMERICAL_COLS + ORDINAL_COLS + BINARY_COLS
LOG_FEATURES = ["HighestSpend", "MonthlyAverageSpend", "Mortgage"]

RANDOM_STATE = 42
TEST_SIZE = 0.20
