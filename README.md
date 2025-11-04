# Real-Time-Fraud-Detection-System-Using-a-Stacked-Hybrid-Scoring-Model
A real-time fraud detection framework built using a stacked hybrid scoring model that blends ensemble and statistical learning methods to identify fraudulent transactions with high precision. Designed for scalability and model interpretability.

## Features
- Live transaction streaming from FastAPI endpoint  
- Real-time fraud prediction with a Stacked hybrid scoring model (that combines the benefits of Random Forest, LightGBM and CatBoost)  
- SHAP-based local + global explanations  
- Automatic data drift detection using KS test  
- Live metrics (precision, recall, F1, confusion matrix, ROC curve)  
- Downloadable transaction report (CSV) 

## Tech Stack
- **Python 3.9+**
- **Streamlit** – interactive dashboard  
- **FastAPI** – live backend  
- **LightGBM, CatBoost, Random Forest** – fraud detection model  
- **SHAP**, **Seaborn**, **Matplotlib** – interpretability and visualization


# Real-Time Fraud Detection Dashboard
- Streamlit-based real-time fraud detection system that integrates with a FastAPI backend, providing live transaction streaming, SHAP explanations, drift detection, and model performance tracking.

# Install Dependencies
- pip install -r requirements.txt

# Run the FastAPI backend
- uvicorn fraud_api:app --reload

# Run the Streamlit app
- streamlit run fraud_detection.py

# How to use the Streamlit Dashboard
- Enter the FastAPI endpoint URL in the sidebar
- Adjust the probability threshold, stream count, and delay
- Click “🚀 Start Live Stream”
- Watch transactions stream live, see drift detection, and model metrics update

# Example Screenshots
- 

# Project Structure
fraud-detection-dashboard/
  │
  ├── fraud_detection.py                # Streamlit frontend
  ├── fraud_api.py                # FastAPI backend
  ├── fraud_detection.pkl   # Trained model
  ├── creditcard.csv        # Example dataset
  ├── requirements.txt
  └── README.md
