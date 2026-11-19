# Payment Fraud Detection with Cost-Aware ML

A production-style fraud detection model for online card payments, built on the
[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset
(590,540 payments, 3.5% fraud).

**Result (test period, leakage-free): 67.2% of fraud caught at a 1% false-alarm rate**,
PR-AUC 0.744, ROC-AUC 0.957. A cost-aware alert level cuts total fraud cost by 64%
versus blocking nothing and by 41% versus the default 0.5 threshold.

## The idea

Fraud happens to *customers*, not single payments: once a card is stolen, several
payments in a row are fraud. The dataset has no customer column, so the model
**rebuilds a customer ID** (card + billing address + the date the card was first used)
and describes each customer with:

- **Fraud history:** earlier frauds, fraud rate, most recent known label and days since
  the last fraud, counting a label only once it is **at least 7 days old** (time for a
  customer report or chargeback).
- **Running behaviour:** average and spread of amounts and other fields, and number of
  distinct emails and devices, over the customer's payments **so far**.
- **Velocity:** how many payments, and how much money, the customer, card, device and email
  moved in the last hour, day and week.
- **30-day popularity counts** of cards, emails and devices.

## Production rules

| Rule | How it is enforced |
|---|---|
| No feature uses the future | Past-only computations in `src/features.py`; `tests/test_features.py` rebuilds features with later payments removed and checks nothing changes |
| Labels arrive late | `LABEL_DELAY_DAYS = 7`, fixed before training; a payment's own label is never used |
| Honest evaluation | Time split 70 / 15 / 15; settings, tree count and alert levels chosen on validation; test scored once |
| Same logic in training and scoring | One feature module shared by the notebook, `src/train.py` and `src/predict.py`; category encodings saved with the model |
| Explainable alerts | Every score comes with its top reasons (SHAP values from LightGBM) |

## Results

Alert levels were chosen on validation and applied to the test period:

| Policy | Fraud caught | False-alarm rate | Total cost* |
|---|---|---|---|
| Block nothing | 0% | 0% | $469,608 |
| Default threshold 0.5 | 54.0% | 0.19% | $285,844 |
| Flag top 3.40% (1% false-alarm level) | 67.7% | 1.08% | $225,891 |
| **Flag top 10.5% (cost-optimal)** | **84.3%** | 7.84% | **$169,206** |

\*Missed fraud costs its amount; each false alarm costs $10 of review time.

## Project structure

```
fraud_detection_final.ipynb   full walkthrough with outputs and charts
src/features.py               past-only feature engineering (shared)
src/train.py                  train, choose alert levels, evaluate, save model
src/predict.py                batch-score new payments with reasons
tests/test_features.py        leakage and label-delay tests
models/fraud_lgbm.txt         trained LightGBM model
models/metadata.json          features, category encodings, alert levels, metrics
```

## How to run

```bash
pip install -r requirements.txt

# Put train_transaction.csv and train_identity.csv from Kaggle in data/
python -m pytest tests -q     # leakage tests
python -m src.train           # trains and saves models/fraud_lgbm.txt + models/metadata.json
python -m src.predict         # scores new_payments.csv -> scored.csv (file names are set at the top of src/predict.py)
```

`scored.csv` contains `fraud_score`, `flag_1pct_fpr`, `flag_cost_optimal` and
`top_reasons` for each new payment.

## Limitations

- The customer ID is rebuilt from anonymised columns; a bank's real customer IDs would be more reliable.
- Scoring is batch. Real-time scoring would need a feature store that keeps each customer's running statistics and known labels up to date.
- The 7-day label delay is an assumption; if confirmations take longer, recall would be lower.
- Choices use one validation period rather than time-series cross-validation, to keep training time reasonable.

**Tech:** Python, Pandas, NumPy, LightGBM, Optuna, Scikit-learn, Matplotlib, Seaborn, Pytest
