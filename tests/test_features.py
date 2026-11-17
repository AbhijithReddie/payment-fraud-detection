"""Checks that no feature uses information from the future.

Run from the project folder:  python -m pytest tests -q
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features import DAY, add_fraud_history, build_features, feature_columns, load_data  # noqa: E402


def tiny_history_frame(days):
    """One customer (same key in every column) paying on the given days."""
    n = len(days)
    keys = {k: ["A"] * n for k in ["uid", "uid2", "card1_addr1", "card1", "card_all", "device_key", "email_card"]}
    return pd.DataFrame({"TransactionDT": np.array(days) * DAY, **keys})


def test_fraud_label_is_hidden_until_delay_has_passed():
    # Fraud on day 0, then payments on days 3, 6, 7 and 10
    df = tiny_history_frame([0, 3, 6, 7, 10])
    labels = np.array([1, 0, 0, 0, 0], dtype=float)
    out = add_fraud_history(df, labels, delay_days=7)
    # Days 3 and 6: the day-0 fraud is not 7 days old yet, so nothing is known
    assert out["hist_uid_n"].tolist()[:3] == [0, 0, 0]
    # Day 7 and later: the day-0 fraud is now known
    assert out["hist_uid_fraud"].tolist()[3:] == [1, 1]
    assert out["hist_uid_last_label"].tolist()[3] == 1


def test_own_label_never_used():
    df = tiny_history_frame([0, 20])
    with_fraud = add_fraud_history(df.copy(), np.array([0, 1.0]), delay_days=7)
    without_fraud = add_fraud_history(df.copy(), np.array([0, 0.0]), delay_days=7)
    cols = [c for c in with_fraud.columns if c.startswith("hist_")]
    pd.testing.assert_frame_equal(with_fraud[cols], without_fraud[cols])


def test_unknown_labels_are_ignored():
    df = tiny_history_frame([0, 10, 30])
    out = add_fraud_history(df, np.array([np.nan, 1, 0]), delay_days=7)
    # On day 30 only the day-10 label is known (day-0 outcome is unknown)
    assert out["hist_uid_n"].tolist()[2] == 1
    assert out["hist_uid_fraud"].tolist()[2] == 1


@pytest.mark.skipif(not os.path.exists("data/train_transaction.csv"), reason="needs the Kaggle data in data/")
def test_features_do_not_change_when_future_rows_are_removed():
    """Build features on 150,000 payments and on the first 120,000 of them.
    If any feature looked at the future, the shared 120,000 rows would differ."""
    df = load_data("data").iloc[:150_000].reset_index(drop=True)
    labels = df["isFraud"].values
    full = build_features(df.copy(), labels)
    short = build_features(df.iloc[:120_000].copy(), labels[:120_000])
    cols = feature_columns(full)
    pd.testing.assert_frame_equal(full.loc[:119_999, cols], short[cols], check_dtype=False)
