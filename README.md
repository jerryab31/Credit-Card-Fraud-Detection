# 💳 Credit Card Fraud Detection

This project uses machine learning to detect fraudulent credit card transactions based on anonymized data. It implements data preprocessing, SMOTE (oversampling), LightGBM with Optuna hyperparameter tuning, threshold optimization (TPR - FPR), SHAP explainability, and a user-friendly Streamlit interface.

---

## 🔍 Project Overview

- **Dataset**: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)  
- **Rows**: 284,807 transactions  
- **Class Imbalance**: 0.17% fraud cases (492 frauds)  
- **Goal**: Build a robust binary classifier using AUPRC as the key metric

---

## ⚙️ Features & Techniques

- SMOTE for handling class imbalance  
- PCA-based anonymized features (`V1` to `V28`)  
- LightGBM classifier with Optuna tuning  
- Threshold selection using TPR - FPR maximization  
- SHAP values for model explainability  
- Streamlit UI with 20 top SHAP features

---

## 📈 Model Performance
Evaluation Metric: AUPRC (Area Under Precision-Recall Curve)
Initial AUPRC (raw probabilities): 0.8562
Best threshold (based on F1-score): 0.9821

Post-threshold metrics:
    • Average Precision Score: 0.7467
    • Confusion Matrix & Classification Report used for detailed evaluation



| Metric             | Value     |
|--------------------|-----------|
| Average Precision  | ~0.85+     |
| Threshold ( F1-score maximization) |0.9821 |

---

## 🧪 File Structure

```
├── fraud_detection_app.py         # Streamlit UI
├── final_model.pkl                # Trained LightGBM pipeline
├── default_values.pkl             # Default values for input form
├── requirements.txt               # Python dependencies
└── README.md                      # Project overview
```

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run fraud_detection_app.py
```

---

## 🌐 Live Demo (Streamlit Cloud)

[👉 Launch Live App](https://credit-card-fraud-detection-jerryab31.streamlit.app/)

---

## 📊 Sample Output

Legitimate Transaction  
Probability: 0.93

Fraudulent Transaction  
Probability: 0.67

---

## 🧠 Author

**Jerry Abraham**  
