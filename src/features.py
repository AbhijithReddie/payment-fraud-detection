"""Feature engineering for the fraud model.

Production rule: every feature for a payment uses only information that exists
at the moment the payment is made.
- Customer statistics (averages, counts, distinct values) use the customer's
  earlier payments plus the current one, never later ones.
- Popularity counts use a rolling 30-day window, so they don't grow over time.
- Fraud labels are used only once they are LABEL_DELAY_DAYS old (time for a
  customer report or a chargeback to come in). A payment's own label is never used.
"""
import numpy as np
import pandas as pd

LABEL_DELAY_DAYS = 7
DAY = 86400

D_COLS = [f"D{k}" for k in range(1, 16) if k != 9]
KEY_COLS = ["uid", "uid2", "card1_addr1", "email_card", "card_all", "device_key"]
COUNT_COLS = ["card1", "card2", "card3", "card5", "addr1", "P_emaildomain", "R_emaildomain",
              "DeviceInfo", "id_30", "id_31", "id_33", "uid", "uid2", "card1_addr1"]
AGG_COLS = ["TransactionAmt", "D4n", "D9", "D10n", "D15n", "dist1", "C1", "C2", "C5", "C6", "C9",
            "C11", "C13", "C14", "V127", "V136", "V307", "V309", "V314", "V320", "id_02"]
NUNIQUE_COLS = ["P_emaildomain", "R_emaildomain", "dist1", "id_19", "id_20", "id_13", "M4", "M5",
                "DeviceInfo", "card4", "addr2"]
HISTORY_KEYS = ["uid", "uid2", "card1_addr1", "card1", "card_all", "device_key", "email_card"]
VELOCITY_KEYS = ["uid", "card1", "device_key", "email_card"]
NOT_FEATURES = ["TransactionID", "isFraud", "TransactionDT", "day"] + KEY_COLS + D_COLS


def load_data(folder="data"):
    """Read and join the transaction and identity files, in time order."""
    df = pd.read_csv(f"{folder}/train_transaction.csv").merge(
        pd.read_csv(f"{folder}/train_identity.csv"), on="TransactionID", how="left")
    float_cols = df.select_dtypes("float64").columns
    df = df.astype({col: "float32" for col in float_cols})
    return df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)


def _group_positions(df, key):
    """Sort payments by (key, time) once. Returns each payment's sort key, the
    sorted keys, its position in that order, and where its key's block starts."""
    codes = pd.factorize(df[key])[0].astype(np.int64)
    t = df["TransactionDT"].values.astype(np.int64)
    sort_key = codes * 10**8 + t
    order = np.argsort(sort_key, kind="stable")
    sorted_key = sort_key[order]
    pos = np.empty(len(order), dtype=np.int64)
    pos[order] = np.arange(len(order))
    start = np.searchsorted(sorted_key, codes * 10**8, side="left")
    return codes, t, sort_key, order, sorted_key, pos, start


def add_basic_features(df):
    df["day"] = (df["TransactionDT"] // DAY).astype("float32")
    df["hour"] = ((df["TransactionDT"] // 3600) % 24).astype("float32")
    df["weekday"] = (df["day"] % 7).astype("float32")
    df["log_amount"] = np.log1p(df["TransactionAmt"]).astype("float32")
    df["amount_cents"] = (df["TransactionAmt"] % 1).round(3).astype("float32")
    for col in D_COLS:
        df[col + "n"] = (df["day"] - df[col]).astype("float32")
    return df


def add_keys(df):
    """Customer ID (card + billing address + first-use date) and looser keys."""
    card1 = df["card1"].astype(str)
    addr1 = df["addr1"].fillna(-1).astype(str)
    email = df["P_emaildomain"].fillna("none").astype(str)
    df["uid"] = card1 + "_" + addr1 + "_" + df["D1n"].fillna(-9999).astype(str)
    df["uid2"] = df["uid"] + "_" + email
    df["card1_addr1"] = card1 + "_" + addr1
    df["email_card"] = card1 + "_" + email
    df["card_all"] = (card1 + "_" + df["card2"].fillna(-1).astype(str) + "_" + df["card3"].fillna(-1).astype(str)
                      + "_" + df["card5"].fillna(-1).astype(str) + "_" + df["card6"].fillna("none").astype(str) + "_" + addr1)
    df["device_key"] = (df["DeviceInfo"].fillna("none").astype(str) + "_" + df["id_30"].fillna("none").astype(str)
                        + "_" + df["id_31"].fillna("none").astype(str) + "_" + df["id_33"].fillna("none").astype(str))
    return df


def add_rolling_counts(df):
    """How many payments in the last 30 days (including this one) share this value."""
    for col in COUNT_COLS:
        _, _, sort_key, _, sorted_key, pos, _ = _group_positions(df, col)
        window_start = np.searchsorted(sorted_key, sort_key - 30 * DAY, side="right")
        df[col + "_count_30d"] = (pos - window_start + 1).astype("float32")
    return df


def add_customer_aggregates(df):
    """Running statistics of the customer's payments so far."""
    uid = df["uid"]
    values = df[AGG_COLS].astype("float64")
    n = values.notna().astype("float64").groupby(uid, sort=False).cumsum()
    total = values.fillna(0).groupby(uid, sort=False).cumsum()
    total_sq = (values.fillna(0) ** 2).groupby(uid, sort=False).cumsum()
    mean = total / n.where(n > 0)
    var = ((total_sq - total ** 2 / n.where(n > 0)) / (n - 1).where(n > 1)).clip(lower=0)
    for col in AGG_COLS:
        df[f"uid_{col}_mean"] = mean[col].astype("float32")
        df[f"uid_{col}_std"] = np.sqrt(var[col]).astype("float32")
    df["uid_amt_ratio"] = (df["TransactionAmt"] / df["uid_TransactionAmt_mean"]).astype("float32")

    for col in NUNIQUE_COLS:
        first_seen = (~df.duplicated(["uid", col])) & df[col].notna()
        df[f"uid_{col}_nunique"] = first_seen.groupby(uid, sort=False).cumsum().astype("float32")

    groups = df.groupby("uid", sort=False)
    df["uid_days_active"] = (df["day"] - groups["day"].cummin()).astype("float32")
    df["uid_seq"] = groups.cumcount().astype("float32")
    df["uid_dt_since_prev"] = groups["TransactionDT"].diff().astype("float32")
    return df


def add_fraud_history(df, labels, delay_days=LABEL_DELAY_DAYS):
    """Fraud history per key, using only labels at least `delay_days` old.
    `labels` may contain NaN for payments whose outcome is not known yet."""
    labels = np.asarray(labels, dtype=np.float64)
    for key in HISTORY_KEYS:
        codes, t, _, order, sorted_key, _, start = _group_positions(df, key)
        y_sorted = labels[order]
        known_sorted = ~np.isnan(y_sorted)
        fraud_sorted = np.where(known_sorted, y_sorted, 0)
        cum_known = np.concatenate([[0], np.cumsum(known_sorted)])
        cum_fraud = np.concatenate([[0], np.cumsum(fraud_sorted)])

        # Payments of this key made at least `delay_days` before this one
        end = np.searchsorted(sorted_key, codes * 10**8 + t - delay_days * DAY, side="right")
        end = np.maximum(end, start)
        n_known = cum_known[end] - cum_known[start]
        n_fraud = cum_fraud[end] - cum_fraud[start]

        # Most recent known label, and most recent known fraud
        idx = np.arange(len(order))
        last_known = np.maximum.accumulate(np.where(known_sorted, idx, -1))
        last_fraud = np.maximum.accumulate(np.where(fraud_sorted == 1, idx, -1))
        before = np.maximum(end - 1, 0)
        lk = np.where(end > start, last_known[before], -1)
        lf = np.where(end > start, last_fraud[before], -1)
        has_label = lk >= start
        has_fraud = lf >= start
        sorted_time = sorted_key % 10**8

        df[f"hist_{key}_n"] = n_known.astype("float32")
        df[f"hist_{key}_fraud"] = n_fraud.astype("float32")
        df[f"hist_{key}_rate"] = np.where(n_known > 0, n_fraud / np.maximum(n_known, 1), np.nan).astype("float32")
        df[f"hist_{key}_last_label"] = np.where(has_label, y_sorted[np.maximum(lk, 0)], np.nan).astype("float32")
        df[f"hist_{key}_days_since_fraud"] = np.where(
            has_fraud, (t - sorted_time[np.maximum(lf, 0)]) / DAY, np.nan).astype("float32")
    return df


def add_velocity(df):
    """Number and amount of this key's earlier payments in the last 1 hour, 24 hours and 7 days."""
    amounts = df["TransactionAmt"].values.astype(np.float64)
    for key in VELOCITY_KEYS:
        _, t, sort_key, order, sorted_key, pos, start = _group_positions(df, key)
        cum_amount = np.concatenate([[0], np.cumsum(amounts[order])])
        for name, seconds in [("1h", 3600), ("24h", DAY), ("7d", 7 * DAY)]:
            window_start = np.maximum(np.searchsorted(sorted_key, sort_key - seconds, side="left"), start)
            window_start = np.minimum(window_start, pos)
            df[f"vel_{key}_n_{name}"] = (pos - window_start).astype("float32")
            df[f"vel_{key}_amt_{name}"] = (cum_amount[pos] - cum_amount[window_start]).astype("float32")
        prev = np.where(pos > start, pos - 1, -1)
        df[f"vel_{key}_secs_since_prev"] = np.where(
            prev >= 0, t - sorted_key[np.maximum(prev, 0)] % 10**8, np.nan).astype("float32")
    return df


def build_features(df, labels, delay_days=LABEL_DELAY_DAYS):
    """All feature steps in order. `df` must be in time order."""
    df = add_basic_features(df)
    df = add_keys(df)
    df = add_rolling_counts(df)
    df = add_customer_aggregates(df)
    df = add_fraud_history(df, labels, delay_days)
    df = add_velocity(df)
    return df


def feature_columns(df):
    return [col for col in df.columns if col not in NOT_FEATURES]


def category_columns(df, features):
    return [col for col in features if df[col].dtype == object or str(df[col].dtype) in ("string", "str", "category")]


def to_matrix(df, features, categories, drop=False):
    """Features as one float32 array. Text columns become codes using the category
    lists saved at training time, so scoring uses exactly the same encoding."""
    X = np.empty((len(df), len(features)), dtype=np.float32)
    for i, col in enumerate(features):
        if col in categories:
            codes = pd.Categorical(df[col], categories=categories[col]).codes
            X[:, i] = np.where(codes < 0, np.nan, codes)
        else:
            X[:, i] = df[col].values
        if drop:
            df.drop(columns=col, inplace=True)
    return X
