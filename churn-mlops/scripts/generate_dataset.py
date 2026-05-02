"""
Generate a synthetic Telco-like customer churn dataset for demo purposes.
Run this once: python scripts/generate_dataset.py
Outputs data/raw/telco_churn.csv
"""

import os
import random
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

N = 7043

contracts = ["Month-to-month", "One year", "Two year"]
internet_services = ["DSL", "Fiber optic", "No"]
payment_methods = [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)"
]
yes_no = ["Yes", "No"]
yes_no_nps = ["Yes", "No", "No phone service"]
yes_no_nis = ["Yes", "No", "No internet service"]

tenure = np.random.randint(0, 73, N)
contract = np.random.choice(contracts, N, p=[0.55, 0.24, 0.21])
internet_service = np.random.choice(internet_services, N, p=[0.34, 0.44, 0.22])
monthly_charges = np.round(np.random.uniform(18, 120, N), 2)
total_charges = np.round(tenure * monthly_charges + np.random.normal(0, 5, N), 2)
total_charges = np.clip(total_charges, 0, None)

# Churn probability: higher for month-to-month, high monthly charges, low tenure
churn_score = (
    0.35 * (contract == "Month-to-month").astype(float) +
    0.20 * (monthly_charges > 70).astype(float) +
    0.25 * (tenure < 12).astype(float) +
    0.10 * (internet_service == "Fiber optic").astype(float) +
    np.random.uniform(0, 0.15, N)
)
churn = (churn_score > 0.45).astype(int)

df = pd.DataFrame({
    "customerID": [f"CUST{str(i).zfill(5)}" for i in range(N)],
    "gender": np.random.choice(["Male", "Female"], N),
    "SeniorCitizen": np.random.choice([0, 1], N, p=[0.84, 0.16]),
    "Partner": np.random.choice(yes_no, N, p=[0.48, 0.52]),
    "Dependents": np.random.choice(yes_no, N, p=[0.30, 0.70]),
    "tenure": tenure,
    "PhoneService": np.random.choice(yes_no, N, p=[0.90, 0.10]),
    "MultipleLines": np.random.choice(yes_no_nps, N, p=[0.42, 0.48, 0.10]),
    "InternetService": internet_service,
    "OnlineSecurity": np.random.choice(yes_no_nis, N, p=[0.29, 0.49, 0.22]),
    "OnlineBackup": np.random.choice(yes_no_nis, N, p=[0.34, 0.44, 0.22]),
    "DeviceProtection": np.random.choice(yes_no_nis, N, p=[0.34, 0.44, 0.22]),
    "TechSupport": np.random.choice(yes_no_nis, N, p=[0.29, 0.49, 0.22]),
    "StreamingTV": np.random.choice(yes_no_nis, N, p=[0.38, 0.40, 0.22]),
    "StreamingMovies": np.random.choice(yes_no_nis, N, p=[0.39, 0.39, 0.22]),
    "Contract": contract,
    "PaperlessBilling": np.random.choice(yes_no, N, p=[0.59, 0.41]),
    "PaymentMethod": np.random.choice(payment_methods, N, p=[0.34, 0.23, 0.22, 0.21]),
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges.astype(str),
    "Churn": ["Yes" if c else "No" for c in churn],
})

os.makedirs(os.path.join("data", "raw"), exist_ok=True)
out_path = os.path.join("data", "raw", "telco_churn.csv")
df.to_csv(out_path, index=False)
print(f"Generated {N} rows -> {out_path}")
print(f"Churn rate: {churn.mean():.1%}")
