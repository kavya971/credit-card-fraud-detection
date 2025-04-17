# Credit Card Fraud Detection using Machine Learning

This project aims to build a machine learning model that detects fraudulent credit card transactions. The dataset is highly imbalanced, so techniques like **SMOTE** and **Random Forest Classifier** are used to improve the model's performance.

---

## 📁 Dataset

- Source: [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets)
- Description: Contains transactions made by credit cards, with a labeled column `IsFraud` indicating whether a transaction is fraudulent (1) or not (0).

---

## 🔧 Technologies Used

- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Google Colab

---

## 📊 Project Workflow

1. **Data Preprocessing**
   - Checked for null values.
   - Scaled the `Amount` feature using `StandardScaler`.
   - Dropped irrelevant features like `Time`, `Amount`, and `TransactionDate`.

2. **Class Imbalance Handling**
   - Used SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset.

3. **Modeling**
   - Applied `RandomForestClassifier`.
   - Evaluated using:
     - Confusion Matrix
     - Classification Report
     - ROC Curve

4. **Evaluation**
   - Compared performance before and after SMOTE.

---

## 📈 Results

- Accuracy remained high due to class imbalance, but recall and F1-score improved slightly after applying SMOTE.
- Still challenging due to the extremely imbalanced nature of the data.

---

## 📌 How to Use

1. Clone this repo:
   ```bash
   git clone https://github.com/kavya971/credit-card-fraud-detection.git
2.Open the notebook (.ipynb) in Google Colab or Jupyter Notebook.

3.Run the cells to see data analysis, model training, and results.
