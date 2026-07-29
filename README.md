# 💳 Credit Card Fraud Detection System

An end-to-end Machine Learning web application built with **Streamlit** to detect fraudulent credit card transactions based on PCA features and transaction metrics.

---

## 📊 Model Performance Benchmarks

Multiple classification models were trained and evaluated on the dataset (`Class` target variable). Here are the accuracy scores:

| Model Name                       |  Accuracy Score   |    Status     |
| :------------------------------- | :---------------: | :-----------: |
| **Random Forest Classifier** 🏆  | **0.980 (98.0%)** | **Selected**  |
| **Extra Trees Classifier** 🏆    | **0.980 (98.0%)** | Top Performer |
| **K-Neighbors Classifier**       | **0.980 (98.0%)** | Top Performer |
| **XGBoost Classifier**           | **0.980 (98.0%)** | Top Performer |
| **AdaBoost Classifier**          |   0.975 (97.5%)   |     Good      |
| **Decision Tree Classifier**     |   0.970 (97.0%)   |   Baseline    |
| **Gradient Boosting Classifier** |   0.970 (97.0%)   |   Baseline    |

---

## ⚙️ Feature Inputs (`X`) & Target (`Y`)

- **Target Variable (`Y`):** `Class` (0 = Legitimate, 1 = Fraudulent)
- **Input Features (`X`):**
  - `Time`: Seconds elapsed since the first transaction.
  - `V1, V2, V3, V4, V5`: PCA transformed feature vectors.
  - `Amount`: Transaction amount in currency units.

---

## 📁 Repository Structure

```text
├── credit_card_fraud_sy...csv   # Dataset file
├── model.ipynb                 # EDA & Training Notebook
├── app.py                      # Streamlit UI Application
├── model.pkl                    # Trained Model File
├── scaler.pkl                   # StandardScaler Object
├── column.pkl                   # Feature Column Metadata
├── requirements.txt             # Python Dependencies
└── README.md                    # Documentation




A high-accuracy Machine Learning project evaluating multiple classifiers (Random Forest, Extra Trees, XGBoost, KNN at 98% accuracy) on Credit Card Fraud Data. Features dynamic inputs like Time, PCA components (V1-V5), and Amount to predict transaction legitimacy (Class) via a fault-tolerant Streamlit web UI.
```
