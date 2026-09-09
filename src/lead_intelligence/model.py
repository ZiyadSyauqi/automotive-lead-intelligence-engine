"""Train-fitted preprocessing with an explicit feature allowlist."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lead_intelligence.data import CATEGORICAL, NUMERIC


def build_pipeline() -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                            ("encode", OneHotEncoder(handle_unknown="ignore"))])
    return Pipeline([
        ("preprocess", ColumnTransformer([("numeric", numeric, NUMERIC),
                                         ("categorical", categorical, CATEGORICAL)],
                                        remainder="drop")),
        ("classifier", LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)),
    ])
