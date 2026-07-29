# 💳 Credit Card Fraud & Anomaly Detection System

An interactive, production-ready **Machine Learning Web Application** built with **Streamlit** to detect fraudulent credit card transactions in real-time based on transformed PCA features and transaction metrics.

[![Live App](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B?style=for-the-badge&logo=streamlit)](YOUR_LIVE_APP_LINK_HERE)

---

## 🖼️ Application Preview & UI

![Credit Card Fraud Detection UI](YOUR_IMAGE_PATH_OR_URL_HERE)

> _Interactive Streamlit dashboard designed with business-friendly feature inputs, fault-tolerant logic, and instant fraud probability metrics._

---

## 🔗 Live Application

You can access and test the deployed application directly here:
👉 **[Click Here to Launch Live Demo](YOUR_LIVE_APP_LINK_HERE)**

---

## 🖼️ Key Features & Capabilities

- **User-Friendly Interface:** Business-oriented feature labels replacing raw mathematical PCA names ($V1$–$V5$) for seamless user experience.
- **Smart Decision Dashboard:** Clear status classification (**Legitimate** vs. **Fraudulent**), Risk Levels, and Model Confidence Score ($0-100\%$) instead of raw binary values ($0/1$).
- **Fault-Tolerant Pipeline:** Robust exception handling ensuring that the UI renders smoothly even if dependencies or `.pkl` files are missing.
- **High Accuracy Benchmarks:** Powered by high-precision ensemble tree classifiers trained on transaction risk patterns.

---

## 📊 Model Evaluation & Benchmarks

Multiple classification algorithms were evaluated to handle high variance and anomaly patterns. Here is the comparative accuracy metric across models:

| Model Name                       |  Accuracy Score   |       Status       |
| :------------------------------- | :---------------: | :----------------: |
| **Random Forest Classifier** 🏆  | **0.980 (98.0%)** | **Selected Model** |
| **Extra Trees Classifier** 🏆    | **0.980 (98.0%)** |   Top Performer    |
| **K-Neighbors Classifier**       | **0.980 (98.0%)** |   Top Performer    |
| **XGBoost Classifier**           | **0.980 (98.0%)** |   Top Performer    |
| **AdaBoost Classifier**          |   0.975 (97.5%)   |   High Precision   |
| **Decision Tree Classifier**     |   0.970 (97.0%)   |      Baseline      |
| **Gradient Boosting Classifier** |   0.970 (97.0%)   |      Baseline      |

> **Selected Baseline:** The **Random Forest Classifier** achieved **98.0% accuracy** with reliable probability estimation for risk grading.

---

## ⚙️ Model Features & Input Parameters

| UI Parameter Name                 | Feature Code | Description / Domain Context                                |
| :-------------------------------- | :----------: | :---------------------------------------------------------- |
| **Transaction Timestamp**         |    `Time`    | Elapsed time in seconds since the first dataset transaction |
| **Transaction Behavior Factor 1** |     `V1`     | Primary PCA vector for transaction behavior patterns        |
| **Account Activity Vector 2**     |     `V2`     | PCA vector capturing account usage variations               |
| **Security Risk Index 3**         |     `V3`     | PCA vector measuring security variance                      |
| **Anomalous Pattern Score 4**     |     `V4`     | PCA vector identifying spending behavior anomalies          |
| **Location/Device Metric 5**      |     `V5`     | PCA vector for contextual transaction parameters            |
| **Transaction Amount**            |   `Amount`   | Total monetary value of the transaction ($)                 |
| **Target Output**                 |   `Class`    | `0` = Legitimate Transaction, `1` = Fraudulent              |

---

## 📁 Repository Structure

```text
├── credit_card_fraud_synthetic.csv     # Primary Transaction Dataset
├── model.ipynb                         # Data Preprocessing, EDA & Model Training
├── .gitignore                          # Git Ignore Rules
├── .gitattributes                      # Git LFS Configuration
├── app.py                              # Fault-Tolerant Streamlit Web Application
├── model.pkl                           # Trained Machine Learning Model
├── scaler.pkl                          # Fitted StandardScaler Object
├── column.pkl                          # Feature Column Definitions
├── requirements.txt                    # Python Dependencies
└── README.md                           # Project Documentation



A high-accuracy Machine Learning project evaluating multiple classifiers (Random Forest, Extra Trees, XGBoost, KNN at 98% accuracy) on Credit Card Fraud Data. Features dynamic inputs like Time, PCA components (V1-V5), and Amount to predict transaction legitimacy (Class) via a fault-tolerant Streamlit web UI.
```
