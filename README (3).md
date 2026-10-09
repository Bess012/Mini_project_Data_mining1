# ⚽ Football Player Loan Prediction Dashboard

Interactive Streamlit dashboard comparing five tuned classifiers
(Logistic Regression, Decision Tree, Random Forest, KNN, XGBoost)
on the task of predicting whether a football player will be loaned.

## Highlights

- Repeated 5×2 stratified CV on ROC-AUC (small-sample robust).
- Leak-free pipeline: imputation, scaling, and one-hot encoding
  (with rare-category grouping) all inside a `ColumnTransformer`.
- Two decision thresholds reported: 0.50 (default) and the
  OOF-F1-selected threshold (per model).
- Interactive confusion matrices, ROC/PR curves, and single-player scoring.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
