"""
Checks that no feature uses information from the future.

Run from the project folder:  python -m pytest tests -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features import DAY, add_fraud_history, build_features, feature_columns, load_data


# One customer (the same key in every key column) paying on the given days
def one_customer(days):
    df = pd.DataFrame({"TransactionDT": np.array(days) * DAY})
    for key in ["uid", "uid2", "card1_addr1", "card1", "card_all", "device_key", "email_card"]:
        df[key] = "A"
    return df


def test_fraud_label_is_hidden_until_7_days_have_passed():
    # Fraud on day 0, then payments on days 3, 6, 7 and 10
    out = add_fraud_history(one_customer([0, 3, 6, 7, 10]), [1, 0, 0, 0, 0], delay_days=7)
    assert out["hist_uid_n"].tolist()[:3] == [0, 0, 0]      # days 3 and 6: not known yet
    assert out["hist_uid_fraud"].tolist()[3:] == [1, 1]     # day 7 onwards: known
    assert out["hist_uid_last_label"].tolist()[3] == 1


def test_own_label_is_never_used():
    with_fraud = add_fraud_history(one_customer([0, 20]), [0, 1], delay_days=7)
    without_fraud = add_fraud_history(one_customer([0, 20]), [0, 0], delay_days=7)
    pd.testing.assert_frame_equal(with_fraud, without_fraud)


def test_unknown_labels_are_ignored():
    # On day 30 only the day-10 label is known (the day-0 outcome is unknown)
    out = add_fraud_history(one_customer([0, 10, 30]), [np.nan, 1, 0], delay_days=7)
    assert out["hist_uid_n"].tolist()[2] == 1
    assert out["hist_uid_fraud"].tolist()[2] == 1


@pytest.mark.skipif(not os.path.exists("data/train_transaction.csv"), reason="needs the Kaggle data in data/")
def test_features_do_not_change_when_future_payments_are_removed():
    # If any feature looked at the future, removing the last 30,000 payments would change the earlier rows
    df = load_data("data").iloc[:150_000].reset_index(drop=True)
    labels = df["isFraud"].values
    full = build_features(df.copy(), labels)
    short = build_features(df.iloc[:120_000].copy(), labels[:120_000])
    cols = feature_columns(full)
    pd.testing.assert_frame_equal(full.loc[:119_999, cols], short[cols], check_dtype=False)
