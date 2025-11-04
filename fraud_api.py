from fastapi import FastAPI
import pandas as pd
import pickle
import random
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)

# Load model
with open("fraud_detection.pkl", "rb") as f:
    model = pickle.load(f)

expected_features = getattr(model, "feature_name_", None)
logging.info(f"Model expected features: {expected_features}")

# Load dataset structure
df_sample = pd.read_csv("creditcard.csv", nrows=1)
dataset_features = [c for c in df_sample.columns if c != "Class"]

# Align features correctly
if expected_features is None:
    logging.warning("Model has no feature_name_. Using dataset columns.")
    expected_features = dataset_features
elif set(expected_features) != set(dataset_features):
    logging.warning("Feature mismatch detected. Using dataset column order instead.")
    expected_features = dataset_features
else:
    logging.info("Model features match dataset.")

@app.get("/transaction")
def get_transaction():
    """Simulate one transaction and predict fraud."""
    df = pd.read_csv("creditcard.csv").sample(1)
    X = df.drop(columns=["Class"])
    y = int(df["Class"].iloc[0])

    # Ensure column alignment
    for f in expected_features:
        if f not in X.columns:
            X[f] = 0
    X = X[expected_features]

    # Predict
    proba = model.predict_proba(X)[0][1]
    pred = int(proba > 0.5)

    # Return output
    return {
        "transaction": X.iloc[0].to_dict(),   # will now have "V1", "V2", ..., "Amount"
        "fraud_probability": float(proba),
        "is_fraud": bool(pred),
        "actual": y
    }