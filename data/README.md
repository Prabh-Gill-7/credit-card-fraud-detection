# Dataset: Credit Card Fraud Detection

## Overview
This directory contains transaction data used for training and evaluating fraud detection models.

- **Primary File:** `creditcard.csv`
- **Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Origin:** Real European cardholder transactions recorded in September 2013.

## Dataset Schema
The dataset contains 31 numeric features:
- **`Time`**: Seconds elapsed between each transaction and the first transaction in the dataset.
- **`V1` - `V28`**: Principal components obtained via Principal Component Analysis (PCA) to protect confidential customer identity.
- **`Amount`**: Transaction monetary value.
- **`Class`**: Target response variable:
  - `0`: Legitimate transaction
  - `1`: Fraudulent transaction

## Synthetic Mock Dataset
If you do not have the complete Kaggle dataset (150 MB), you can generate a synthetic dataset matching the identical 31-column schema:
```bash
python scripts/generate_sample_data.py
```
