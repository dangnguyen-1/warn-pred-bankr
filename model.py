import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

SELECTED_FEATS = [
    "retained_earnings",
    "market_cap",
    "asset_turnover",
    "debt_to_asset_ratio",
    "quick_ratio",
    "current_ratio",
    "gross_profit_margin",
    "ros",
    "gross_profit",
    "long_term_debt",
    "current_assets",
    "current_liabilities",
]


def get_feature_cols(df):
    return [c for c in SELECTED_FEATS if c in df.columns]


def split_holdout(df, train_end_year=2014, test_start_year=2015):
    train_df = df[df["year"] <= train_end_year].copy()
    test_df = df[df["year"] >= test_start_year].copy()
    return train_df, test_df


def walk_forward_splits(df, start_year, end_year, min_train_years=7):
    """
    Expanding window:
      train: start_year..(y-1)
      val: y
    for y from (start_year + min_train_years) to end_year inclusive
    """
    years = sorted(df["year"].unique())
    years = [y for y in years if start_year <= y <= end_year]

    for i in range(min_train_years, len(years)):
        train_years = years[:i]
        val_year = years[i]

        train_df = df[df["year"].isin(train_years)].copy()
        val_df = df[df["year"] == val_year].copy()

        meta = {"train_end": train_years[-1], "val_year": val_year}
        yield train_df, val_df, meta


def _eval_threshold_metrics(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def _predict_labels(model, X, threshold=0.5):
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
        return (scores >= threshold).astype(int)
    scores = model.decision_function(X)
    return (scores >= 0.0).astype(int)


def train_logistic_regression(X_train, y_train):
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def run_walk_forward_cv(df, train_fn, feature_cols, start_year=1999, end_year=2014, min_train_years=7, threshold=0.5):
    fold_metrics = []
    cm_sum = np.array([[0, 0], [0, 0]], dtype=int)

    for train_df, val_df, meta in walk_forward_splits(
        df, start_year=start_year, end_year=end_year, min_train_years=min_train_years
    ):
        X_train = train_df[feature_cols]
        y_train = train_df["status_label"]
        X_val = val_df[feature_cols]
        y_val = val_df["status_label"]

        # If a fold has zero positives, skip it (metrics would be meaningless)
        if int((y_train == 1).sum()) == 0 or int((y_val == 1).sum()) == 0:
            continue

        model = train_fn(X_train, y_train)
        y_pred = _predict_labels(model, X_val, threshold=threshold)

        m = _eval_threshold_metrics(y_val, y_pred)
        cm_sum += m["confusion_matrix"]

        fold_metrics.append(
            {
                "val_year": meta["val_year"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            }
        )

    avg = {
        "precision": float(np.mean([m["precision"] for m in fold_metrics])) if fold_metrics else float("nan"),
        "recall": float(np.mean([m["recall"] for m in fold_metrics])) if fold_metrics else float("nan"),
        "f1": float(np.mean([m["f1"] for m in fold_metrics])) if fold_metrics else float("nan"),
        "confusion_matrix": cm_sum,
        "n_folds_used": len(fold_metrics),
    }
    return fold_metrics, avg


def evaluate_holdout(train_df, test_df, train_fn, feature_cols, threshold=0.5):
    X_train = train_df[feature_cols]
    y_train = train_df["status_label"]
    X_test = test_df[feature_cols]
    y_test = test_df["status_label"]

    model = train_fn(X_train, y_train)
    y_pred = _predict_labels(model, X_test, threshold=threshold)

    metrics = _eval_threshold_metrics(y_test, y_pred)
    return model, metrics