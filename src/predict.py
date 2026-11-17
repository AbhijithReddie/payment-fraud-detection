"""Score new payments with the saved model (batch scoring).

    python -m src.predict --history data --new new_payments.csv --out scored.csv

`--history` is a folder with labelled past payments (train_transaction.csv and
train_identity.csv, same format as Kaggle). `--new` is a CSV of new payments: the
transaction columns, plus identity columns where available, and no `isFraud` column.

The new payments are added after the history, features are built exactly as in
training (past-only, labels at least 7 days old), and each new payment gets:
- `fraud_score`: model probability
- `flag_1pct_fpr`: 1 if it is in the top share of the batch matching a 1% false-alarm rate
- `flag_cost_optimal`: 1 if it is in the cost-optimal top share of the batch
- `top_reasons`: for flagged payments, the three features that pushed the score up the most (SHAP)
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features import build_features, load_data, to_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", default="data")
    parser.add_argument("--new", required=True)
    parser.add_argument("--model", default="models")
    parser.add_argument("--out", default="scored.csv")
    args = parser.parse_args()

    model = lgb.Booster(model_file=f"{args.model}/fraud_lgbm.txt")
    with open(f"{args.model}/metadata.json") as f:
        meta = json.load(f)

    history = load_data(args.history)
    new = pd.read_csv(args.new)
    new = new.astype({c: "float32" for c in new.select_dtypes("float64").columns})
    new["isFraud"] = np.nan  # outcome unknown
    df = pd.concat([history, new], ignore_index=True).sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    is_new = df["TransactionID"].isin(new["TransactionID"]).values

    df = build_features(df, df["isFraud"].values, delay_days=meta["label_delay_days"])
    scored = df.loc[is_new, ["TransactionID", "TransactionDT", "TransactionAmt"]].reset_index(drop=True)
    cat_cols = list(meta["categories"])
    df[cat_cols] = df[cat_cols].astype(str).where(df[cat_cols].notna())
    X = to_matrix(df.loc[is_new], meta["features"], meta["categories"])

    scores = model.predict(X)
    scored["fraud_score"] = scores
    scored["flag_1pct_fpr"] = (scores >= np.quantile(scores, 1 - meta["flag_share_1pct_fpr"])).astype(int)
    scored["flag_cost_optimal"] = (scores >= np.quantile(scores, 1 - meta["flag_share_cost_optimal"])).astype(int)

    # Reasons (SHAP) only for flagged payments: they are the ones an analyst reviews, and SHAP is slow
    names = np.array(meta["features"])
    flagged = np.flatnonzero(scored["flag_cost_optimal"].values)
    scored["top_reasons"] = ""
    if len(flagged):
        contributions = model.predict(X[flagged], pred_contrib=True)[:, :-1]
        scored.loc[flagged, "top_reasons"] = [", ".join(names[np.argsort(-c)[:3]]) for c in contributions]
    scored.to_csv(args.out, index=False)
    print(f"Scored {len(scored)} payments -> {args.out} ({scored['flag_1pct_fpr'].sum()} flagged at the 1% level)")


if __name__ == "__main__":
    main()
