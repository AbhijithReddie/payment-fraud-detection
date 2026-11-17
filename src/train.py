"""Train the fraud model from the command line (same steps as the notebook).

    python -m src.train --data data --out models

1. Builds past-only features (src/features.py).
2. Trains on the first 70% with early stopping on the next 15% (validation).
3. Picks the alert levels on validation.
4. Retrains on the first 85% and scores the last 15% (test) once.
5. Saves models/fraud_lgbm.txt and models/metadata.json.
"""
import argparse
import gc
import json
import os

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.features import (LABEL_DELAY_DAYS, build_features, category_columns, feature_columns, load_data,
                          to_matrix)

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=367, min_child_samples=22, colsample_bytree=0.5,
              subsample=0.94, subsample_freq=1, reg_lambda=0.32, reg_alpha=0.02, metric="average_precision",
              seed=42, verbose=-1)


def evaluate(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return {"PR-AUC": float(average_precision_score(y_true, y_score)),
            "ROC-AUC": float(roc_auc_score(y_true, y_score)),
            "Recall at 1% FPR": float(np.interp(0.01, fpr, tpr))}


def total_cost(flagged, y_true, amounts, false_alarm_cost=10):
    return amounts[~flagged & (y_true == 1)].sum() + false_alarm_cost * (flagged & (y_true == 0)).sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="models")
    parser.add_argument("--max-rounds", type=int, default=6000, help="lower it for a quick smoke test")
    args = parser.parse_args()

    df = load_data(args.data)
    val_start, test_start = int(len(df) * 0.70), int(len(df) * 0.85)
    y = df["isFraud"].values
    y_train, y_val, y_test = y[:val_start], y[val_start:test_start], y[test_start:]

    df = build_features(df, y)
    features = feature_columns(df)
    cat_cols = category_columns(df, features)
    categories = {c: sorted(df.loc[:test_start - 1, c].dropna().astype(str).unique()) for c in cat_cols}
    df[cat_cols] = df[cat_cols].astype(str).where(df[cat_cols].notna())
    amounts = df["TransactionAmt"].values.copy()
    X = to_matrix(df, features, categories, drop=True)
    del df
    gc.collect()
    cat_index = [features.index(c) for c in cat_cols]

    # Train with early stopping on validation
    train_set = lgb.Dataset(X[:val_start], y_train, feature_name=features, categorical_feature=cat_index)
    val_set = lgb.Dataset(X[val_start:test_start], y_val, reference=train_set)
    model = lgb.train(PARAMS, train_set, num_boost_round=args.max_rounds, valid_sets=[val_set],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
    val_scores = model.predict(X[val_start:test_start], num_iteration=model.best_iteration)

    # Alert levels chosen on validation
    share_1pct = float((val_scores >= np.quantile(val_scores[y_val == 0], 0.99)).mean())
    shares = np.round(np.arange(0.005, 0.3001, 0.005), 3)
    val_amounts = amounts[val_start:test_start]
    costs = [total_cost(val_scores >= np.quantile(val_scores, 1 - s), y_val, val_amounts) for s in shares]
    share_cost = float(shares[int(np.argmin(costs))])

    # Final model on train + validation, test scored once
    trees = max(1, int(model.best_iteration * 1.2))
    del model, train_set, val_set
    gc.collect()
    final = lgb.train(PARAMS, lgb.Dataset(X[:test_start], y[:test_start], feature_name=features,
                                          categorical_feature=cat_index), num_boost_round=trees)
    test_results = evaluate(y_test, final.predict(X[test_start:]))

    os.makedirs(args.out, exist_ok=True)
    final.save_model(os.path.join(args.out, "fraud_lgbm.txt"))
    metadata = {"label_delay_days": LABEL_DELAY_DAYS, "features": features, "categories": categories,
                "trees": trees, "params": PARAMS, "flag_share_1pct_fpr": share_1pct,
                "flag_share_cost_optimal": share_cost, "validation_results": evaluate(y_val, val_scores),
                "test_results": test_results}
    with open(os.path.join(args.out, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    print(json.dumps({"trees": trees, "validation": metadata["validation_results"], "test": test_results,
                      "flag_share_1pct_fpr": share_1pct, "flag_share_cost_optimal": share_cost}, indent=2))


if __name__ == "__main__":
    main()
