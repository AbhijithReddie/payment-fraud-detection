"""
Train the fraud model (same steps as the notebook).

Run from the project folder:  python -m src.train
"""
import json
import gc
import os

import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from src.features import LABEL_DELAY_DAYS, load_data, build_features, feature_columns, category_columns, to_matrix

DATA_FOLDER = "data"
MODEL_FOLDER = "models"

# Tuned LightGBM settings (Optuna) with learning rate 0.05
PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 367,
    "min_child_samples": 22,
    "colsample_bytree": 0.5,
    "subsample": 0.94,
    "subsample_freq": 1,
    "reg_lambda": 0.32,
    "reg_alpha": 0.02,
    "metric": "average_precision",
    "seed": 42,
    "verbose": -1,
}


# PR-AUC, ROC-AUC and share of fraud caught at a 1% false-alarm rate
def evaluate(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return {
        "PR-AUC": float(average_precision_score(y_true, y_score)),
        "ROC-AUC": float(roc_auc_score(y_true, y_score)),
        "Recall at 1% FPR": float(np.interp(0.01, fpr, tpr)),
    }


# Missed fraud costs its amount, each false alarm costs $10
def total_cost(flagged, y_true, amounts):
    missed_fraud = amounts[~flagged & (y_true == 1)].sum()
    false_alarms = (flagged & (y_true == 0)).sum()
    return missed_fraud + 10 * false_alarms


# 1. Read data and split by time: first 70% train, next 15% validation, last 15% test
df = load_data(DATA_FOLDER)
val_start = int(len(df) * 0.70)
test_start = int(len(df) * 0.85)
y = df["isFraud"].values
y_train = y[:val_start]
y_val = y[val_start:test_start]
y_test = y[test_start:]

# 2. Build features (past information only)
df = build_features(df, y)
features = feature_columns(df)
cat_cols = category_columns(df, features)

# 3. Turn text columns into numbers, using the categories seen before the test period
categories = {col: sorted(df.loc[:test_start - 1, col].dropna().astype(str).unique()) for col in cat_cols}
df[cat_cols] = df[cat_cols].astype(str).where(df[cat_cols].notna())
amounts = df["TransactionAmt"].values.copy()
X = to_matrix(df, features, categories, drop=True)
cat_index = [features.index(col) for col in cat_cols]
del df
gc.collect()

# 4. Train on the training period, stop when validation PR-AUC stops improving
train_set = lgb.Dataset(X[:val_start], y_train, feature_name=features, categorical_feature=cat_index)
val_set = lgb.Dataset(X[val_start:test_start], y_val, reference=train_set)
model = lgb.train(PARAMS, train_set, num_boost_round=6000, valid_sets=[val_set],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
val_scores = model.predict(X[val_start:test_start], num_iteration=model.best_iteration)

# 5. Choose the alert levels on validation
#    a) share of payments to flag so that 1% of genuine payments are flagged
share_1pct = float((val_scores >= np.quantile(val_scores[y_val == 0], 0.99)).mean())

#    b) share of payments to flag with the lowest total cost
val_amounts = amounts[val_start:test_start]
shares = np.round(np.arange(0.005, 0.3001, 0.005), 3)
costs = [total_cost(val_scores >= np.quantile(val_scores, 1 - share), y_val, val_amounts) for share in shares]
share_cost = float(shares[int(np.argmin(costs))])

# 6. Retrain on train + validation, then score the test period once
trees = max(1, int(model.best_iteration * 1.2))
del model, train_set, val_set
gc.collect()
dev_set = lgb.Dataset(X[:test_start], y[:test_start], feature_name=features, categorical_feature=cat_index)
final_model = lgb.train(PARAMS, dev_set, num_boost_round=trees)
test_results = evaluate(y_test, final_model.predict(X[test_start:]))

# 7. Save the model and everything needed to score new payments
os.makedirs(MODEL_FOLDER, exist_ok=True)
final_model.save_model(f"{MODEL_FOLDER}/fraud_lgbm.txt")

metadata = {
    "label_delay_days": LABEL_DELAY_DAYS,
    "features": features,
    "categories": categories,
    "trees": trees,
    "params": PARAMS,
    "flag_share_1pct_fpr": share_1pct,
    "flag_share_cost_optimal": share_cost,
    "validation_results": evaluate(y_val, val_scores),
    "test_results": test_results,
}
with open(f"{MODEL_FOLDER}/metadata.json", "w") as f:
    json.dump(metadata, f)

print("Trees:", trees)
print("Validation:", metadata["validation_results"])
print("Test:", test_results)
