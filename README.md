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

| Metric             | Value     |
|--------------------|-----------|
| ROC-AUC Score      | 0.86+     |
| Average Precision  | ~0.78     |
| Threshold (TPR - FPR) | 0.33 |

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

[👉 Launch Live App](https://fraud-detection-jerryab31.streamlit.app/) <!-- Replace with your actual Streamlit Cloud link -->

---

## 📊 Sample Output

Legitimate Transaction  
Probability: 0.93

Fraudulent Transaction  
Probability: 0.67

---

## 🧠 Author

**Jerry Abraham**  
