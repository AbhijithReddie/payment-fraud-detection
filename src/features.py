"""
Feature engineering for the fraud model.

Golden rule: a feature for a payment may only use information that existed
when that payment was made. We never look at later payments, and a fraud label
is only used once it is 7 days old (the time it takes for fraud to be reported).
"""
import numpy as np
import pandas as pd

# A fraud label becomes "known" only 7 days after the payment
LABEL_DELAY_DAYS = 7

# Number of seconds in one day
DAY = 86400

# "Days since ..." columns that we turn into fixed dates (D9 is skipped)
D_COLS = [f"D{k}" for k in range(1, 16) if k != 9]

# Text keys we create to group payments (used to build features, not given to the model)
KEY_COLS = ["uid", "uid2", "card1_addr1", "email_card", "card_all", "device_key"]

# Columns whose 30-day popularity we count
COUNT_COLS = ["card1", "card2", "card3", "card5", "addr1", "P_emaildomain", "R_emaildomain",
              "DeviceInfo", "id_30", "id_31", "id_33", "uid", "uid2", "card1_addr1"]

# Columns we average for each customer
AGG_COLS = ["TransactionAmt", "D4n", "D9", "D10n", "D15n", "dist1", "C1", "C2", "C5", "C6", "C9",
            "C11", "C13", "C14", "V127", "V136", "V307", "V309", "V314", "V320", "id_02"]

# Columns where we count how many different values each customer has used
NUNIQUE_COLS = ["P_emaildomain", "R_emaildomain", "dist1", "id_19", "id_20", "id_13", "M4", "M5",
                "DeviceInfo", "card4", "addr2"]

# Keys for the fraud-history features
HISTORY_KEYS = ["uid", "uid2", "card1_addr1", "card1", "card_all", "device_key", "email_card"]

# Keys for the velocity (recent activity) features
VELOCITY_KEYS = ["uid", "card1", "device_key", "email_card"]

# Columns that are not model features
NOT_FEATURES = ["TransactionID", "isFraud", "TransactionDT", "day"] + KEY_COLS + D_COLS


def load_data(folder="data"):
    # Read both files and join them on TransactionID
    transactions = pd.read_csv(f"{folder}/train_transaction.csv")
    identity = pd.read_csv(f"{folder}/train_identity.csv")
    df = transactions.merge(identity, on="TransactionID", how="left")

    # float32 uses half the memory of float64
    float_cols = df.select_dtypes("float64").columns
    df = df.astype({col: "float32" for col in float_cols})

    # Oldest payment first
    df = df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    return df


def add_basic_features(df):
    # Day number, hour of day and day of week
    df["day"] = (df["TransactionDT"] // DAY).astype("float32")
    df["hour"] = ((df["TransactionDT"] // 3600) % 24).astype("float32")
    df["weekday"] = (df["day"] % 7).astype("float32")

    # Log of the amount, and the cents part
    df["log_amount"] = np.log1p(df["TransactionAmt"]).astype("float32")
    df["amount_cents"] = (df["TransactionAmt"] % 1).round(3).astype("float32")

    # "Days since X" becomes "the day X happened", which stays the same for a customer
    for col in D_COLS:
        df[col + "n"] = (df["day"] - df[col]).astype("float32")
    return df


def add_keys(df):
    # Text versions of the columns we combine (missing values get a placeholder)
    card1 = df["card1"].astype(str)
    addr1 = df["addr1"].fillna(-1).astype(str)
    email = df["P_emaildomain"].fillna("none").astype(str)

    # Customer ID: card + billing address + the day the card was first used
    df["uid"] = card1 + "_" + addr1 + "_" + df["D1n"].fillna(-9999).astype(str)

    # Looser keys
    df["uid2"] = df["uid"] + "_" + email
    df["card1_addr1"] = card1 + "_" + addr1
    df["email_card"] = card1 + "_" + email

    df["card_all"] = (card1 + "_" + df["card2"].fillna(-1).astype(str)
                      + "_" + df["card3"].fillna(-1).astype(str)
                      + "_" + df["card5"].fillna(-1).astype(str)
                      + "_" + df["card6"].fillna("none").astype(str)
                      + "_" + addr1)

    df["device_key"] = (df["DeviceInfo"].fillna("none").astype(str)
                        + "_" + df["id_30"].fillna("none").astype(str)
                        + "_" + df["id_31"].fillna("none").astype(str)
                        + "_" + df["id_33"].fillna("none").astype(str))
    return df


def add_rolling_counts(df):
    # How many payments in the last 30 days (including this one) share this card, email, device, ...
    for col in COUNT_COLS:
        groups = df[[col, "time", "TransactionDT"]].groupby(col, dropna=False)
        counts = groups.rolling("30D", on="time").count()["TransactionDT"]
        df[col + "_count_30d"] = counts.reset_index(level=0, drop=True).astype("float32")
    return df


def add_customer_aggregates(df):
    # Running average and spread of each column, over the customer's payments so far
    for col in AGG_COLS:
        values = df[col].astype("float64")
        count = values.notna().groupby(df["uid"], sort=False).cumsum()
        total = values.fillna(0).groupby(df["uid"], sort=False).cumsum()
        total_sq = (values.fillna(0) ** 2).groupby(df["uid"], sort=False).cumsum()

        mean = total / count
        variance = ((total_sq - total ** 2 / count) / (count - 1)).where(count > 1).clip(lower=0)

        df[f"uid_{col}_mean"] = mean.astype("float32")
        df[f"uid_{col}_std"] = np.sqrt(variance).astype("float32")

    # This amount compared with the customer's average amount
    df["uid_amt_ratio"] = (df["TransactionAmt"] / df["uid_TransactionAmt_mean"]).astype("float32")

    # How many different emails, devices, ... the customer has used so far
    for col in NUNIQUE_COLS:
        first_time_seen = (~df.duplicated(["uid", col])) & df[col].notna()
        df[f"uid_{col}_nunique"] = first_time_seen.groupby(df["uid"], sort=False).cumsum().astype("float32")

    # Days since the customer's first payment, payment number, seconds since the previous payment
    groups = df.groupby("uid", sort=False)
    df["uid_days_active"] = (df["day"] - groups["day"].cummin()).astype("float32")
    df["uid_seq"] = groups.cumcount().astype("float32")
    df["uid_dt_since_prev"] = groups["TransactionDT"].diff().astype("float32")
    return df


def add_fraud_history(df, labels, delay_days=LABEL_DELAY_DAYS):
    # labels: 1 = fraud, 0 = genuine, NaN = not known yet
    labels = pd.Series(labels, index=df.index, dtype="float64")
    is_known = labels.notna()
    is_fraud = labels == 1

    # Each payment looks back at its key's history as it was `delay_days` before it
    lookup = pd.DataFrame({"cutoff": df["TransactionDT"] - delay_days * DAY})

    for key in HISTORY_KEYS:
        # Running history of each key, after every payment
        groups = df[key]
        history = pd.DataFrame({
            key: df[key],
            "TransactionDT": df["TransactionDT"],
            "n_known": is_known.groupby(groups, dropna=False).cumsum(),
            "n_fraud": labels.fillna(0).groupby(groups, dropna=False).cumsum(),
            "last_label": labels.groupby(groups, dropna=False).ffill(),
            "last_fraud_time": df["TransactionDT"].where(is_fraud).groupby(groups, dropna=False).ffill(),
        })

        # For each payment, take the latest history row of the same key made on or before the cutoff time
        lookup[key] = df[key]
        past = pd.merge_asof(lookup, history, left_on="cutoff", right_on="TransactionDT", by=key)

        n_known = past["n_known"].fillna(0)
        n_fraud = past["n_fraud"].fillna(0)

        df[f"hist_{key}_n"] = n_known.astype("float32").values
        df[f"hist_{key}_fraud"] = n_fraud.astype("float32").values
        df[f"hist_{key}_rate"] = (n_fraud / n_known).where(n_known > 0).astype("float32").values
        df[f"hist_{key}_last_label"] = past["last_label"].astype("float32").values
        df[f"hist_{key}_days_since_fraud"] = ((df["TransactionDT"].values - past["last_fraud_time"]) / DAY).astype("float32").values
        lookup = lookup.drop(columns=key)
    return df


def add_velocity(df):
    # How many payments, and how much money, this key made in the last 1 hour, 24 hours and 7 days
    for key in VELOCITY_KEYS:
        groups = df[[key, "time", "TransactionDT", "TransactionAmt"]].groupby(key, dropna=False)
        for name in ["1h", "24h", "7D"]:
            window = groups.rolling(name, on="time", closed="both")
            count = window.count()["TransactionDT"].reset_index(level=0, drop=True)
            amount = window.sum()["TransactionAmt"].reset_index(level=0, drop=True)

            # The window includes the current payment, so subtract it to keep only earlier payments
            label = name.lower()
            df[f"vel_{key}_n_{label}"] = (count - 1).astype("float32")
            df[f"vel_{key}_amt_{label}"] = (amount - df["TransactionAmt"].astype("float64")).astype("float32")

        # Seconds since this key's previous payment
        df[f"vel_{key}_secs_since_prev"] = groups["TransactionDT"].diff().astype("float32")
    return df


def build_features(df, labels, delay_days=LABEL_DELAY_DAYS):
    # df must be in time order (oldest first)
    df = add_basic_features(df)
    df = add_keys(df)

    # A real date-time column, needed for the time windows
    df["time"] = pd.to_datetime(df["TransactionDT"], unit="s")

    df = add_rolling_counts(df)
    df = add_customer_aggregates(df)
    df = add_fraud_history(df, labels, delay_days)
    df = add_velocity(df)
    return df.drop(columns="time")


def feature_columns(df):
    return [col for col in df.columns if col not in NOT_FEATURES]


def category_columns(df, features):
    return [col for col in features if df[col].dtype == object or str(df[col].dtype) in ("string", "str", "category")]


def to_matrix(df, features, categories, drop=False):
    # One float32 table of all features. Text columns become numbers using the
    # category lists saved at training time, so scoring uses the same codes.
    X = np.empty((len(df), len(features)), dtype=np.float32)
    for i, col in enumerate(features):
        if col in categories:
            codes = pd.Categorical(df[col], categories=categories[col]).codes
            X[:, i] = np.where(codes < 0, np.nan, codes)
        else:
            X[:, i] = df[col].values

        # Optionally delete each column after copying it, to save memory
        if drop:
            df.drop(columns=col, inplace=True)
    return X
