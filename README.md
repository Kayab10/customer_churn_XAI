# Customer Churn Prediction AI

A comprehensive Streamlit application for predicting customer churn using XGBoost with explainability powered by SHAP and LIME. Built for business users and data scientists to understand churn patterns and make data-driven retention decisions.

## 🎯 Overview

This application combines machine learning with interpretability to help businesses:
- **Predict** which customers are likely to churn
- **Understand** why predictions are made using SHAP and LIME
- **Act** on data-driven recommendations to reduce churn
- **Analyze** global trends and model performance

The model achieves high accuracy on customer churn data using an XGBoost classifier trained on 10,000+ customer records with 10 features.

---

## ✨ Features

### 1. **Overview Dashboard**
High-level insights into customer churn patterns:
- **4 Key Metrics**: Total customers, churn rate, average tenure, high-risk customer count
- **Churn Distribution**: Bar chart showing stayed vs. churned customers
- **Geographic Analysis**: Churn by geography (France, Germany, Spain)
- **Tenure Impact**: Histogram showing tenure vs. churn correlation
- **Salary Analysis**: Box plot comparing salaries for churned vs. stayed customers
- **Feature Correlation Heatmap**: Correlation matrix for numeric features

**Use case**: Business stakeholders can quickly understand churn trends without diving into details.

---

### 2. **Customer Prediction Page**
Interactive prediction tool with explainability:

#### **Input Form**
- **Customer Details** (10 fields in 2-column layout):
  - Credit Score (300–850)
  - Geography (France, Germany, Spain)
  - Gender (Female, Male)
  - Age (18–100)
  - Tenure in months (0–72)
  - Account Balance ($)
  - Number of Products (1–4)
  - Has Credit Card (Yes/No)
  - Is Active Member (Yes/No)
  - Estimated Salary ($)

#### **Prediction Results**
- **Churn Probability**: Percentage likelihood of churn
- **Risk Level**: Low (< 45%), Medium (45–75%), High (≥ 75%)
- **Predicted Outcome**: "Churn ⚠️" or "Stay ✅"
- **Color-coded Alerts**: Red (high risk), yellow (medium risk), green (low risk)

#### **Recommendations**
Personalized actions based on churn probability:
- **High Risk (≥ 75%)**: Offer loyalty discounts, assign proactive support
- **Medium Risk (45–75%)**: Personalized offers, service bundle recommendations
- **Low Risk (< 45%)**: Maintain relationship, reward loyalty

#### **CSV Download**
Export prediction details including all inputs and outputs for record-keeping.

#### **SHAP & LIME Explanations**
Expandable section with:
- **SHAP Contributions**: Table and bar chart showing feature impact (top 10)
  - Red = increases churn risk
  - Blue = decreases churn risk
- **LIME Explanations**: Local model-agnostic explanations (top 5 features)
- **Natural Language Summary**: Key drivers in human-readable format

**Use case**: Customer service teams can score individual customers and understand why they're at risk, enabling targeted retention efforts.

---

### 3. **Explainability & Model Performance Page**
Global model insights and evaluation:

#### **XGBoost Feature Importance**
- Horizontal bar chart of built-in feature importances
- Shows which features the model relies on most for predictions

#### **SHAP Feature Importance**
- Mean absolute SHAP values across training data
- Reveals average impact of each feature on model decisions
- Expandable table with top 10 features and their impact scores

#### **Model Evaluation Metrics**
- **Accuracy**: Overall correctness (% of correct predictions)
- **Churn Precision**: Of predicted churners, how many actually churned
- **Churn Recall**: Of actual churners, how many were caught

#### **Classification Report**
Full per-class metrics (Stay vs. Churn):
- Precision, Recall, F1-score for each class
- Support (number of samples)

#### **Confusion Matrix**
- Heatmap showing True Positives, True Negatives, False Positives, False Negatives
- Helps understand types of prediction errors

#### **Key Model Insights**
- Model performance summary (accuracy, recall, false alarm rate)
- Top 3 most important features with impact scores

**Use case**: Data scientists and model validators can review model performance and feature importance for model governance.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Kayab10/customer_churn_XAI.git
   cd customer_churn_XAI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model locally** (generates `model.json` and `model_meta.pkl`):
   ```bash
   python train.py
   ```

4. **Ensure data file exists:**
   - Place `Churn_Modelling.csv` in the `data/` directory

### Running the App

Start the Streamlit server:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

---

## 📁 Project Structure

```
customer_churn_XAI/
├── app.py                          # Main Streamlit application
├── train.py                        # Standalone training script (run once locally)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── model.json                      # Trained XGBoost model (native cross-platform format)
├── model_meta.pkl                  # Encoders, feature names, train/test splits
├── .gitignore                      # Git ignore rules
├── data/
│   └── Churn_Modelling.csv        # Customer dataset (10,000 records, 13 features)
├── src/
│   ├── __init__.py                # Package initialization
│   └── model.py                   # ML pipeline, model, and utility functions
├── notebook.ipynb                  # Model training notebook
└── steps.ipynb                     # Project planning & architecture notebook
```

### File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit app with all three pages (Overview, Prediction, Explainability) |
| `train.py` | Run once to train the model and save `model.json` + `model_meta.pkl` |
| `src/model.py` | Core ML pipeline: data loading, encoding, model training, SHAP/LIME setup, predictions |
| `model.json` | XGBoost model saved in native format — cross-platform, no version dependency |
| `model_meta.pkl` | Encoders and data splits saved with standard pickle — safe across environments |
| `data/Churn_Modelling.csv` | Customer dataset with 10,000 records and 13 columns |
| `requirements.txt` | Python package dependencies |

---

## 🔧 Architecture 

### Data Pipeline

1. **Data Loading**: `load_raw_data()` reads CSV from `data/Churn_Modelling.csv`
2. **Feature Engineering**:
   - Drops non-predictive columns (RowNumber, CustomerId, Surname, Exited)
   - Encodes categorical features (Geography, Gender, HasCrCard, IsActiveMember)
3. **Train-Test Split**: 80% train, 20% test with stratification
4. **Model Training**: XGBoost classifier with balanced class weights
5. **Artifact Saving**:
   - `model.json` — XGBoost native format (cross-platform, no version dependency)
   - `model_meta.pkl` — encoders, feature names, train/test splits (pure Python/numpy)
6. **App Startup**: Loads `model.json` + `model_meta.pkl`, rebuilds SHAP/LIME explainers in-memory

### Model Configuration

```python
MODEL_PARAMS = {
    'n_estimators': 300,        # Number of boosting rounds
    'seed': 101,                # Random seed for reproducibility
    'scale_pos_weight': 4,      # Balance for imbalanced classes
    'eval_metric': 'aucpr',     # Use PR-AUC for evaluation
    'use_label_encoder': False, # Avoid deprecation warning
}
```

### Prediction Threshold

- **Default**: 0.4 (not 0.5) to account for class imbalance(can also use SMOTE technique for better accuracy)
- Customers with ≥ 40% churn probability are flagged as "Churn"
- Risk levels calibrated: Low < 45%, Medium 45–75%, High ≥ 75%

### Explainability Methods

| Method | Use | Scope |
|--------|-----|-------|
| **SHAP** | Feature contributions for specific predictions | Local (per-customer) |
| **LIME** | Model-agnostic explanations | Local (per-customer) |
| **XGBoost Importance** | Built-in feature importance | Global (entire model) |

---

## 📊 Data Features

### Input Features (10)

| Feature | Type | Range | Description |
|---------|------|-------|-------------|
| CreditScore | Numeric | 300–850 | Customer credit score |
| Geography | Categorical | France, Germany, Spain | Country |
| Gender | Categorical | Female, Male | Gender |
| Age | Numeric | 18–100 | Age in years |
| Tenure | Numeric | 0–72 | Months as customer |
| Balance | Numeric | 0–250,000 | Account balance ($) |
| NumOfProducts | Numeric | 1–4 | Number of products held |
| HasCrCard | Categorical | Yes, No | Has credit card |
| IsActiveMember | Categorical | Yes, No | Active member status |
| EstimatedSalary | Numeric | 0–200,000 | Estimated annual salary ($) |

### Target Variable

| Feature | Type | Values | Description |
|---------|------|--------|-------------|
| Exited | Binary | 0, 1 | 1 = Churned, 0 = Stayed |

### Dataset Statistics

- **Total Records**: 10,000
- **Churned Customers**: ~2,650 (26.5%)
- **Stayed Customers**: ~7,350 (73.5%)
- **Class Imbalance Ratio**: 1:2.8 (addressed via `scale_pos_weight`)

---

## 💡 Usage Guide

### Scenario 1: Dashboard Exploration
1. Open app and go to **Overview** tab
2. Review churn rate, geographic distribution, tenure patterns
3. Identify high-risk customer segments

### Scenario 2: Individual Customer Scoring
1. Go to **Prediction** tab
2. Enter customer details (or use pre-filled example values)
3. Click **"🔮 Predict Churn"**
4. Review risk level and recommendations
5. Expand **"🔬 View SHAP & LIME Explanations"** to understand why
6. Download CSV report for CRM integration

### Scenario 3: Model Validation
1. Go to **Explainability** tab
2. Review model accuracy and confusion matrix
3. Check feature importance rankings
4. Verify model isn't overfitting to specific features
5. Share insights 

---

## 🔌 API & Function Reference

### Core Functions in `src/model.py`

#### Data Loading
```python
load_raw_data(path: str) -> pd.DataFrame
```
Loads customer data from CSV.

#### Pipeline Setup
```python
build_pipeline(raw: pd.DataFrame) -> dict
```
Builds complete ML pipeline: encoding, training, SHAP/LIME setup.

#### Save Artifacts
```python
save_model_artifacts(pipeline: dict, model_path: str, meta_path: str)
```
Saves XGBoost model as `model.json` (native format) and metadata as `model_meta.pkl`.

#### Load Artifacts
```python
load_model_artifacts(model_path: str, meta_path: str) -> dict
```
Loads saved artifacts and rebuilds SHAP/LIME explainers fresh. Safe across platforms.

#### Feature Encoding
```python
encode_features(record: dict, categorical_encoders: dict) -> pd.DataFrame
```
Encodes user input to match model's expected format.

#### Prediction
```python
predict_customer(model, row: pd.DataFrame) -> Tuple[int, float]
```
Returns (predicted_label, churn_probability).

#### Risk Classification
```python
customer_risk_level(probability: float) -> str
```
Returns 'Low', 'Medium', or 'High' based on probability.

#### Recommendations
```python
customer_recommendations(probability: float) -> List[str]
```
Returns list of personalized retention actions.

#### Feature Importance
```python
get_feature_importance(model, feature_names: List[str]) -> pd.DataFrame
```
Returns DataFrame with feature importance scores.

#### SHAP Values
```python
get_shap_values(shap_explainer, data: pd.DataFrame)
```
Computes SHAP values for a dataset.

#### LIME Explanations
```python
explain_customer_lime(lime_explainer, predict_fn, sample: np.ndarray, num_features: int) -> pd.DataFrame
```
Returns DataFrame with feature contributions.

#### Model Evaluation
```python
evaluate_model(model, test: pd.DataFrame, labels_test: np.ndarray) -> dict
```
Returns dict with accuracy, confusion matrix, classification report, etc.

---

## 📦 Dependencies


```bash
pip install -r requirements.txt
```




## 🎓 Model Performance

### Expected Metrics (on test set)

| Metric | Typical Value |
|--------|---------------|
| Accuracy | ~80% |
| Churn Precision | ~65% |
| Churn Recall | ~70% |
| F1-Score (Churn) | ~0.67 |

These metrics reflect the inherent difficulty of churn prediction due to class imbalance and feature overlap.

### Key Insights

- **Tenure** is typically the #1 churn driver—new customers churn more
- **Age** and **IsActiveMember** are significant secondary drivers
- **Geography** has some effect but less than individual behavior
- **Balance** and **NumOfProducts** show customer engagement patterns

---


### Key Technologies:
- **XGBoost**: Gradient boosting classifier
- **SHAP**: TreeExplainer for feature importance
- **LIME**: Local interpretable model explanations
- **Streamlit**: Rapid ML app development
- **Plotly**: Interactive visualizations

---


