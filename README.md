# 💳 Credit Default Risk Prediction

A production-ready machine learning web application to predict whether a credit card client will default in the next month.

This project uses a real-world dataset from the [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset), combined with feature engineering, model tuning, and SHAP explainability — all deployed via Streamlit.

---

## Demo

<img width="2806" height="1544" alt="image" src="https://github.com/user-attachments/assets/d17e2eb7-f5bb-4a07-8ba1-d1f0b6eaecd3" />


## Project Structure

```

credit_scoring_model/
├── app.py                 # main Streamlit app
├── requirements.txt       # python dependencies
├── models/                # trained model (.pkl)
├── data/                  
├── notebook/              # notebook for training & EDA

````


## Problem Statement

Predict whether a customer will default on their credit card payment next month based on demographic data, bill/payment history, and credit behavior.


## Dataset Description

- **Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
- **Rows:** 30,000
- **Columns:** 24 original + 5 engineered features

### Input Features:

| Column | Description |
|--------|-------------|
| `LIMIT_BAL` | Credit limit (NT dollars) |
| `SEX` | Gender (1 = Male, 2 = Female) |
| `EDUCATION` | Education level (1 = Graduate, 2 = University, ...) |
| `MARRIAGE` | Marital status |
| `AGE` | Age in years |
| `PAY_0` to `PAY_6` | Repayment status (last 6 months) |
| `BILL_AMT1` to `BILL_AMT6` | Monthly bill amounts |
| `PAY_AMT1` to `PAY_AMT6` | Monthly payment amounts |

### Engineered Features:

- `TOTAL_PAY_AMT`: Total amount paid in 6 months
- `TOTAL_BILL_AMT`: Total bill in 6 months
- `NUM_LATE_PAYMENTS`: Number of late payments
- `MAX_DELAY`: Longest delay (in months)
- `LONGEST_LATE_STREAK`: Longest continuous months of delay


## Tech Stack

- **Language:** Python 3.12
- **ML Model:** XGBoost Classifier
- **Tuning:** GridSearchCV
- **Imbalance Handling:** SMOTE
- **Explainability:** SHAP (Waterfall plot)
- **Deployment:** Streamlit

---

## How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/credit-scoring-model.git
   cd credit-scoring-model
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   source venv/bin/activate  # Mac/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```
---
## Model Performance

| Metric         | Score                                        |
| -------------- | -------------------------------------------- |
| ROC-AUC        | 0.7815                                       |
| Accuracy       | 82%                                          |
| Tuning         | GridSearchCV                                 |
| Explainability | SHAP waterfall                               |


## SHAP Visualization

We use SHAP to explain predictions for each customer:
<img width="1568" height="1042" alt="image" src="https://github.com/user-attachments/assets/9d7e8b5a-5367-43b1-8ff6-62fa474b08f7" />

<img width="1696" height="914" alt="image" src="https://github.com/user-attachments/assets/7c0ac2e2-2db6-4cdb-8f71-673a16d705ec" />


## Future Improvements

* Add login/auth for secure access
* Store prediction history to database
* Train model incrementally with real-time data
---
## 🧑‍💻 Author
**Đạt Đình**  
2nd Year @ UET, Data Engineering Track  
Email: dinhdatnguyen0710@example.com

## 📜 License

MIT License — Free to use and modify

## 🙌 Acknowledgments

* UCI Credit Card Dataset
* Streamlit community
* SHAP and XGBoost developers
