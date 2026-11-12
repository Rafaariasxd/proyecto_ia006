from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42
TOP_N = 10

MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}

DATE_COLS = ["arrival_date_year", "arrival_date_month", "arrival_date_week_number", "arrival_date_day_of_month"]

ADR_COL = ["adr"]
LOG_COLS = ["lead_time", "days_in_waiting_list", "previous_cancellations",
            "previous_bookings_not_canceled", "booking_changes"]
PLAIN_NUM_COLS = ["stays_in_weekend_nights", "stays_in_week_nights", "adults", "children", "babies",
                  "required_car_parking_spaces", "total_of_special_requests", "total_nights", "total_guests"]
BIN_COLS = ["is_repeated_guest", "arrival_is_weekend", "has_children"]
LOW_CAT_COLS = ["hotel", "meal", "market_segment", "distribution_channel", "deposit_type",
                "customer_type", "reserved_room_type", "arrival_month", "arrival_dow"]
HIGH_CAT_COLS = ["country", "agent"]
FEATURE_COLS = ADR_COL + LOG_COLS + PLAIN_NUM_COLS + BIN_COLS + LOW_CAT_COLS + HIGH_CAT_COLS

RAW_COLS = [
    "hotel", "lead_time", "arrival_date_year", "arrival_date_month", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
    "babies", "meal", "country", "market_segment", "distribution_channel", "is_repeated_guest",
    "previous_cancellations", "previous_bookings_not_canceled", "reserved_room_type", "booking_changes",
    "deposit_type", "agent", "days_in_waiting_list", "customer_type", "adr",
    "required_car_parking_spaces", "total_of_special_requests",
]


def to_text(series, empty):
    text = series.astype("object").where(series.notna(), empty).astype(str).str.strip()
    return text.replace({"nan": empty, "NaN": empty, "NULL": empty, "": empty})


def clean_uploaded(df):
    out = df.copy()
    out["children"] = pd.to_numeric(out["children"], errors="coerce").fillna(0).astype(int)
    agent = to_text(out["agent"], "None").str.replace(r"\.0$", "", regex=True)
    out["agent"] = agent.replace({"0": "None"})
    out["country"] = to_text(out["country"], "Unknown")
    out["meal"] = out["meal"].replace("Undefined", "SC")
    return out


def engineer_features(df):
    out = df.copy()
    month = out["arrival_date_month"].map(MONTHS)
    arrival = pd.to_datetime(dict(year=out["arrival_date_year"], month=month,
                                  day=out["arrival_date_day_of_month"]))
    out["arrival_month"] = month.astype(int).astype(str)
    out["arrival_dow"] = arrival.dt.dayofweek.astype(str)
    out["arrival_is_weekend"] = (arrival.dt.dayofweek >= 5).astype(int)
    out["total_nights"] = out["stays_in_weekend_nights"] + out["stays_in_week_nights"]
    out["total_guests"] = out["adults"] + out["children"] + out["babies"]
    out["has_children"] = ((out["children"] + out["babies"]) > 0).astype(int)
    out["agent"] = out["agent"].astype(str)
    out["country"] = out["country"].astype(str)
    return out.drop(columns=DATE_COLS)


def numeric_block(X, prep):
    out = X[ADR_COL + LOG_COLS + PLAIN_NUM_COLS].astype(float).copy()
    out["adr"] = np.log1p(out["adr"].clip(prep["adr_low"], prep["adr_high"]))
    for col in LOG_COLS:
        out[col] = np.log1p(out[col])
    return out


def categorical_block(X, prep):
    cat = X[LOW_CAT_COLS + HIGH_CAT_COLS].copy()
    for col in HIGH_CAT_COLS:
        cat[col] = cat[col].where(cat[col].isin(prep["top"][col]), "Other")
    return pd.get_dummies(cat.astype(str), dtype=float)


def encode(X, prep):
    num = numeric_block(X, prep)
    num = pd.DataFrame(prep["scaler"].transform(num), columns=num.columns, index=X.index)
    cat = categorical_block(X, prep).reindex(columns=prep["dummy_columns"], fill_value=0.0)
    return pd.concat([num, X[BIN_COLS].astype(float), cat], axis=1)


def fit_preprocessing(X, y=None, reducer=None):
    q1, q3 = X["adr"].quantile(0.25), X["adr"].quantile(0.75)
    iqr = q3 - q1
    prep = {
        "adr_low": max(float(q1 - 1.5 * iqr), 0.0),
        "adr_high": float(q3 + 1.5 * iqr),
        "top": {col: X[col].value_counts().head(TOP_N).index.tolist() for col in HIGH_CAT_COLS},
        "reducer": None,
    }
    prep["scaler"] = StandardScaler().fit(numeric_block(X, prep))
    prep["dummy_columns"] = categorical_block(X, prep).columns.tolist()
    if reducer == "PCA":
        prep["reducer"] = PCA(n_components=0.95, random_state=SEED).fit(encode(X, prep))
    elif reducer == "LDA":
        prep["reducer"] = LinearDiscriminantAnalysis(n_components=1).fit(encode(X, prep), y)
    return prep


def transform(X, prep):
    out = encode(X, prep)
    if prep["reducer"] is None:
        return out
    reduced = prep["reducer"].transform(out)
    names = [f"C{i + 1}" for i in range(reduced.shape[1])]
    return pd.DataFrame(reduced, columns=names, index=X.index)


def make_model(name, params, class_weight="balanced"):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=3000, class_weight=class_weight, random_state=SEED, **params)
    if name == "Decision Tree":
        return DecisionTreeClassifier(class_weight=class_weight, random_state=SEED, **params)
    return RandomForestClassifier(n_estimators=150, class_weight=class_weight, random_state=SEED, n_jobs=2, **params)


def make_folds(X, y, reducer=None, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    folds = []
    for train_idx, val_idx in skf.split(X, y):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        prep = fit_preprocessing(X_tr, y_tr, reducer)
        folds.append((transform(X_tr, prep), y_tr, transform(X_va, prep), y_va))
    return folds


def cross_validate(name, params, folds, class_weight="balanced"):
    train_f1, val_f1, val_precision, val_recall = [], [], [], []
    for X_tr, y_tr, X_va, y_va in folds:
        model = make_model(name, params, class_weight).fit(X_tr, y_tr)
        train_f1.append(f1_score(y_tr, model.predict(X_tr)))
        pred = model.predict(X_va)
        val_f1.append(f1_score(y_va, pred))
        val_precision.append(precision_score(y_va, pred))
        val_recall.append(recall_score(y_va, pred))
    return {
        "train_f1": float(np.mean(train_f1)),
        "val_f1": float(np.mean(val_f1)),
        "val_f1_std": float(np.std(val_f1)),
        "val_precision": float(np.mean(val_precision)),
        "val_recall": float(np.mean(val_recall)),
    }


def grid_search(name, grid, folds):
    keys = list(grid)
    best = None
    for values in product(*(grid[k] for k in keys)):
        params = dict(zip(keys, values))
        scores = cross_validate(name, params, folds)
        if best is None or scores["val_f1"] > best["val_f1"]:
            best = dict(scores, params=params)
    return best


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, prob),
    }
    return metrics, pred, prob


def predict_risk(df_raw, bundle):
    X = engineer_features(df_raw[RAW_COLS])[FEATURE_COLS]
    return bundle["model"].predict_proba(transform(X, bundle["prep"]))[:, 1]


def save_bundle(path, prep, model):
    joblib.dump({"prep": prep, "model": model}, path, compress=3)


def load_bundle(path):
    return joblib.load(path)


def evaluate_threshold(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
    }
    if len(np.unique(y_true)) == 2:
        metrics["ROC-AUC"] = roc_auc_score(y_true, scores)
    return metrics, pred
