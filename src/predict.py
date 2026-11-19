"""
Score new payments with the saved model.

Run from the project folder:  python -m src.predict

HISTORY_FOLDER: past payments with known labels (Kaggle format: train_transaction.csv + train_identity.csv)
NEW_FILE:       new payments to score (same columns, no isFraud column)
OUTPUT_FILE:    one row per new payment with its fraud score, flags and top reasons
"""
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import load_data, build_features, to_matrix

HISTORY_FOLDER = "data"
NEW_FILE = "new_payments.csv"
OUTPUT_FILE = "scored.csv"
MODEL_FOLDER = "models"

# 1. Load the model and its settings
model = lgb.Booster(model_file=f"{MODEL_FOLDER}/fraud_lgbm.txt")
with open(f"{MODEL_FOLDER}/metadata.json") as f:
    meta = json.load(f)

# 2. Put the new payments after the history (their outcome is unknown)
history = load_data(HISTORY_FOLDER)
new = pd.read_csv(NEW_FILE)
new = new.astype({col: "float32" for col in new.select_dtypes("float64").columns})
new["isFraud"] = np.nan
df = pd.concat([history, new], ignore_index=True)
df = df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
is_new = df["TransactionID"].isin(new["TransactionID"]).values

# 3. Build features exactly as in training
df = build_features(df, df["isFraud"].values, delay_days=meta["label_delay_days"])
scored = df.loc[is_new, ["TransactionID", "TransactionDT", "TransactionAmt"]].reset_index(drop=True)
cat_cols = list(meta["categories"])
df[cat_cols] = df[cat_cols].astype(str).where(df[cat_cols].notna())
X = to_matrix(df.loc[is_new], meta["features"], meta["categories"])

# 4. Score, and flag the top share of payments at each alert level
scores = model.predict(X)
scored["fraud_score"] = scores
scored["flag_1pct_fpr"] = (scores >= np.quantile(scores, 1 - meta["flag_share_1pct_fpr"])).astype(int)
scored["flag_cost_optimal"] = (scores >= np.quantile(scores, 1 - meta["flag_share_cost_optimal"])).astype(int)

# 5. Top 3 reasons (SHAP values) for each flagged payment
feature_names = np.array(meta["features"])
flagged_rows = np.flatnonzero(scored["flag_cost_optimal"].values)
scored["top_reasons"] = ""
if len(flagged_rows) > 0:
    shap_values = model.predict(X[flagged_rows], pred_contrib=True)[:, :-1]
    scored.loc[flagged_rows, "top_reasons"] = [", ".join(feature_names[np.argsort(-row)[:3]]) for row in shap_values]

# 6. Save
scored.to_csv(OUTPUT_FILE, index=False)
print(f"Scored {len(scored)} payments, {scored['flag_1pct_fpr'].sum()} flagged at the 1% level -> {OUTPUT_FILE}")
