import pytest
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocessing.preprocess import load_and_preprocess


SAMPLE_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "rows.csv")


@pytest.fixture
def preprocessed_df():
    """Load and preprocess the dataset once for all tests."""
    return load_and_preprocess(SAMPLE_CSV)


def test_dataset_file_exists():
    """Dataset file must exist before running pipeline."""
    assert os.path.exists(SAMPLE_CSV), f"Dataset not found at {SAMPLE_CSV}"


def test_output_has_correct_columns(preprocessed_df):
    """Preprocessed output must contain expected columns."""
    expected_cols = [
        "YEAR", "LocationAbbr", "MeasureDesc", "DataSource",
        "Response", "Sample_Size", "Gender", "Race", "Age",
        "Education", "Risk_Level"
    ]
    for col in expected_cols:
        assert col in preprocessed_df.columns, f"Missing column: {col}"


def test_no_nan_in_output(preprocessed_df):
    """Preprocessed DataFrame must not contain any NaN values."""
    assert preprocessed_df.isnull().sum().sum() == 0, "NaN values found after preprocessing"


def test_target_is_binary(preprocessed_df):
    """Risk_Level must only contain values 0 and 1."""
    unique_vals = set(preprocessed_df["Risk_Level"].unique())
    assert unique_vals.issubset({0, 1}), f"Unexpected values in Risk_Level: {unique_vals}"


def test_target_is_balanced(preprocessed_df):
    """Risk_Level should be roughly balanced (each class >= 20%)."""
    counts = preprocessed_df["Risk_Level"].value_counts(normalize=True)
    assert counts[0] >= 0.2, "Low Risk class is underrepresented (<20%)"
    assert counts[1] >= 0.2, "High Risk class is underrepresented (<20%)"


def test_output_has_rows(preprocessed_df):
    """Preprocessed output must have at least 1000 rows."""
    assert len(preprocessed_df) >= 1000, f"Too few rows: {len(preprocessed_df)}"


def test_data_value_column_dropped(preprocessed_df):
    """Original Data_Value column must be removed after labeling."""
    assert "Data_Value" not in preprocessed_df.columns, \
        "Data_Value column should be dropped after creating Risk_Level"
