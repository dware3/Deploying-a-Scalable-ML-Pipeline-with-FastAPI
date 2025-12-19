import pytest

# TODO: add necessary import
import pandas as pd
from fastapi.testclient import TestClient

from ml.data import process_data
from main import app


REQUIRED_COLUMNS = {
    "age",
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "salary",
    }

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    ]

@pytest.fixture(scope="module")
def client():
    """FastAPI test client fixture."""
    return TestClient(app)


@pytest.fixture
def raw_df():
    """Minimal raw input dataframe with the required schema."""
    return pd.DataFrame(
        {
            "age": [30],
            "workclass": ["Private"],
            "education": ["HS-grad"],
            "marital-status": ["Never-married"],
            "occupation": ["Sales"],
            "relationship": ["Not-in-family"],
            "race": ["White"],
            "sex": ["Male"],
            "salary": [0],
        }
    )


@pytest.fixture
def train_df():
    """Small training dataframe used for process_data unit tests."""
    return pd.DataFrame(
        {
            "age": [30, 50],
            "workclass": ["Private", "Self-emp-inc"],
            "education": ["HS-grad", "Doctorate"],
            "marital-status": ["Never-married", "Married-civ-spouse"],
            "occupation": ["Sales", "Exec-managerial"],
            "relationship": ["Not-in-family", "Husband"],
            "race": ["White", "White"],
            "sex": ["Male", "Male"],
            "salary": [0, 1],
        }
    )

@pytest.mark.parametrize(
    "mutation, expected_valid",
    [
        (lambda df: df, True),  # unchanged, should be valid
        (lambda df: df.assign(extra_col=123), False),  # extra column, invalid
        (lambda df: df.drop(columns=["education"]), False),  # missing column, invalid
    ],
    )

# TODO: implement the first test. Change the function name and input as needed

def test_raw_data_schema_exact(raw_df, mutation, expected_valid):
    """Raw input data contains exactly the required columns (no missing, no extra)."""
    df = mutation(raw_df.copy())
    is_valid = set(df.columns) == REQUIRED_COLUMNS
    assert is_valid == expected_valid

# TODO: implement the second test. Change the function name and input as needed
def test_processed_data_uses_required_columns(train_df):
    """
    process_data successfully processes the required columns and produces aligned outputs. Validates shape/row integrity and encoder configuration.
    """
    X, y, encoder, lb = process_data(
        train_df,
        categorical_features=CATEGORICAL_FEATURES,
        label="salary",
        training=True,
    )

    assert X.shape[0] == len(train_df)
    assert y.shape[0] == len(train_df)
    assert len(encoder.categories_) == len(CATEGORICAL_FEATURES)

# TODO: implement the third test. Change the function name and input as needed
@pytest.mark.parametrize("payload", [{}, {"age": 30}])
def test_api_rejects_invalid_input(client, payload):
    """API rejects invalid payloads with a validation error (422)."""
    response = client.post("/data/", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()




