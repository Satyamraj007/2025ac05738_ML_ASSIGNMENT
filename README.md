# Credit Card Fraud Detection — ML Assignment 2

## a. Problem Statement

Credit card fraud causes billions in losses annually. This project builds a binary classification system to identify fraudulent transactions from a dataset of 20,000 credit card transactions. The goal is to correctly flag fraud (`is_fraud = 1`) while minimising false positives, using five classical ML algorithms and comparing their performance on a highly imbalanced dataset.

---

## b. Dataset Description

| Property | Value |
|---|---|
| Source | Synthetic dataset (`test_data.csv`) |
| Total instances | 20,000 |
| Features | 25 (after dropping `transaction_id`) |
| Target column | `is_fraud` (binary: 0 = legitimate, 1 = fraud) |
| Class distribution | 19,661 legitimate (98.3%) / 339 fraud (1.7%) |
| Missing values | None |

### Feature Overview

| Feature | Type | Description |
|---|---|---|
| amount_usd | Numeric | Transaction amount in USD |
| merchant_category | Categorical | Type of merchant (Groceries, Restaurants, etc.) |
| card_type | Categorical | Visa / Mastercard / Amex |
| auth_method | Categorical | OTP, 3D Secure, No Authentication, etc. |
| channel | Categorical | Online / POS |
| device_type | Categorical | Android Phone, Mac, etc. |
| is_foreign_transaction | Boolean | Transaction in foreign country |
| hours_since_last_txn | Numeric | Time gap since last transaction |
| txn_count_last_24h | Numeric | Number of transactions in last 24h |
| distance_from_home_km | Numeric | Distance from cardholder home |
| card_age_months | Numeric | Age of card in months |
| customer_age | Numeric | Age of customer |
| account_balance_usd | Numeric | Current account balance |
| is_new_merchant | Boolean | First transaction with this merchant |
| used_vpn | Boolean | VPN detected |
| ip_country_mismatch | Boolean | IP country differs from card country |
| billing_shipping_mismatch | Boolean | Billing and shipping addresses differ |
| cvv_retry_count | Numeric | Number of CVV retries |
| velocity_score | Numeric | Transaction velocity risk score |
| time_of_day_hour | Numeric | Hour of transaction (0–23) |
| day_of_week | Numeric | Day of week (0=Mon, 6=Sun) |
| is_ai_generated_scam_attempt | Boolean | Flagged as AI-generated scam |
| merchant_risk_score | Numeric | Risk score of merchant |
| prior_disputes | Numeric | Number of prior card disputes |

**Preprocessing:**
- Boolean columns converted to 0/1 integers
- Categorical columns encoded with OrdinalEncoder
- Numeric features scaled with StandardScaler for Logistic Regression, kNN and Naive Bayes
- All preprocessing baked into each pipeline via ColumnTransformer — no external artifacts
- Decision threshold set to 0.15 to improve recall on the minority fraud class
- Models trained on 80% (16,000 rows), evaluated on held-out 20% (4,000 rows)
- **No pre-trained model files** — all models are trained at runtime when the CSV is uploaded

---

## c. GitHub Repository Link

> **[https://github.com/Satyamraj007/2025ac05738_ML_ASSIGNMENT](https://github.com/Satyamraj007/2025ac05738_ML_ASSIGNMENT)**

Repository contents:
```
2025ac05738_ml_assignment/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    └── train_models.ipynb
```

---

## d. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.6100 | 0.9507 | 0.0406 | 0.9706 | 0.0780 | 0.1512 |
| Decision Tree | 0.8267 | 0.5978 | 0.0542 | 0.5588 | 0.0988 | 0.1327 |
| kNN | 0.9493 | 0.6132 | 0.1053 | 0.2647 | 0.1506 | 0.1443 |
| Naive Bayes | 0.8802 | 0.8923 | 0.0865 | 0.6324 | 0.1522 | 0.2026 |
| Random Forest (Ensemble) | 0.9095 | 0.9014 | 0.1090 | 0.6029 | 0.1847 | 0.2293 |

> All models evaluated on a held-out 20% test split (stratified, random_state=42). Decision threshold = 0.15.

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest recall (0.9706) and AUC (0.9507) — catches nearly all fraud and ranks probabilities best among linear models. Precision is very low (0.04) due to severe class imbalance, producing many false alarms. Best when missing fraud is far costlier than false positives. |
| Decision Tree | Moderate recall (0.5588) and low AUC (0.5978) — the weakest discriminator overall. A single tree struggles to capture complex fraud patterns even with `class_weight='balanced'`. Interpretable but outperformed by all other models on this dataset. |
| kNN | High accuracy (0.9493) but low recall (0.2647) and AUC (0.6132) on unseen data — the model fails to generalise well to the rare fraud class. Distance-based lookup works poorly when fraud points are sparse and the majority class dominates the neighbourhood. |
| Naive Bayes | Best balance of recall (0.6324) and F1 (0.1522) among the non-ensemble models. AUC of 0.8923 is strong. The Gaussian assumption holds reasonably for numeric features, making it an effective and fast baseline. |
| Random Forest (Ensemble) | Best overall performer — highest F1 (0.1847), highest MCC (0.2293), strong AUC (0.9014), and strong recall (0.6029). `balanced_subsample` across 300 estimators handles class imbalance better than any single model. Most reliable and robust choice for this dataset. |
| **Overall Winner** | **Random Forest** — best F1, MCC, and strong AUC on the held-out test set. Logistic Regression leads on recall and AUC alone, making it the better choice only when maximising fraud catch rate at the cost of many false alarms is acceptable. |

---

## e. Streamlit App Link

**Live App:** [https://2025ac05738mlassignment-nouwg329mm2gywjrsppg2w.streamlit.app/](https://2025ac05738mlassignment-nouwg329mm2gywjrsppg2w.streamlit.app/)

### App Features
- Automatically loads `test_data.csv` from the repository — no manual upload needed
- Optional CSV upload: upload a new `test_data.csv` to retrain all models on your data
- Trains all 5 models on 80% (16,000 rows) at startup, cached for the session
- Model selection dropdown — switch between models instantly without retraining
- Evaluation metrics display (Accuracy, AUC, Precision, Recall, F1, MCC) on held-out 20% (4,000 rows)
- All-models comparison table with colour highlighting
- Confusion matrix heatmap
- Classification report
- ROC curve
- Fraud probability distribution chart
