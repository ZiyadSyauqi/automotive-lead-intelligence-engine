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


def build_challenger(seed: int = 42) -> Pipeline:
    """One restrained nonlinear configuration; dense encoding fits this small schema."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    pipeline = build_pipeline()
    pipeline.named_steps["preprocess"].set_params(sparse_threshold=0)
    pipeline.set_params(classifier=HistGradientBoostingClassifier(
        max_iter=100, learning_rate=.05, max_leaf_nodes=7,
        min_samples_leaf=40, l2_regularization=1.0,
        early_stopping=False, random_state=seed))
    return pipeline
