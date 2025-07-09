# HealthCare_Problem_Risks_Of_Malnutrion
# 🏥 Healthcare Child Malnutrition Prediction

This repository presents an end-to-end machine learning workflow for predicting **child malnutrition** using health, demographic, and anthropometric data. Through **data preprocessing**, **exploratory data analysis**, **classification**, **regression**, **clustering**, and **model optimization**, the project aims to assist health professionals and researchers in identifying children at risk of malnutrition.

---

## 📁 Dataset Used

- **Original Dataset Source**: [Multiple Indicator Cluster Surveys (MICS) - UNICEF](https://www.kaggle.com/datasets/usharengaraju/child-malnutrition-unicef-dataset/data)
- **Cleaned Dataset**: [`cleaned_malnutrition_dataset.csv`](https://github.com/Safia-ElSabbagh/Healthcare-Child-Malnutrion-prediction/blob/main/Milestone%201/Cleaned%20Survey%20Data.csv)

---

## 🎯 Objective

The main goals of this project are to:
- **Classify** whether a child is malnourished based on key health and socio-economic factors.
- **Predict** continuous outcomes (e.g., malnutrition scores) using regression.
- **Cluster** children into nutritional groups for exploratory analysis.
- **Interpret model behavior** using SHAP and feature importance.
- **Optimize model performance** through hyperparameter tuning.

---

## 🛠️ Tools & Technologies

| Category | Tools/Libraries |
|---------|-----------------|
| Programming | Python 3 |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, SHAP |
| Machine Learning | Scikit-learn, XGBoost |
| Optimization | GridSearchCV |
| Dimensionality Reduction | PCA |
| Deployment | Joblib |

---

## 📈 Project Pipeline

### 1. 📊 Data Preprocessing & Cleaning

- **Removed missing values** and handled outliers.
- **Encoded categorical features** (e.g., gender, region) using label encoding.
- **Scaled numerical features** (e.g., weight, height, age) using `StandardScaler`.
- Split data into **training** and **testing** sets using stratified sampling.

### 2. 🔎 Exploratory Data Analysis (EDA)

- Box plots, histograms, and heatmaps were used to:
  - Detect class imbalance.
  - Visualize feature distributions.
  - Examine correlations (e.g., between weight and malnutrition).
- Insight: Malnutrition was strongly associated with **underweight**, **height-for-age**, and **socioeconomic status**.

### 3. 🤖 Classification Models

We trained several models to classify whether a child is malnourished:

| Model | Description |
|-------|-------------|
| Logistic Regression | Baseline linear classifier. Simple but effective. |
| Decision Tree | Tree-based model for interpretable rules. |
| Random Forest | Ensemble of trees with better generalization. |
| K-Nearest Neighbors | Distance-based method; sensitive to scaling. |
| XGBoost | High-performing gradient boosting classifier. |

- **Evaluation metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
- **Best model**: XGBoost showed the highest performance and interpretability via SHAP.

### 4. 📉 Regression Models

Used to **predict malnutrition score** (continuous outcome) instead of binary label.

| Model | Performance Metrics (on test set) |
|-------|-----------------------------------|
| Linear Regression | MAE, MSE, RMSE, R² |
| Lasso Regression | Penalized model to reduce overfitting |
| Decision Tree Regressor | Nonlinear regressor, interpretable |

- Visualization of residuals and R² revealed Decision Tree Regressor performed best on non-linear patterns.

### 5. 📊 Unsupervised Learning (Clustering)

Applied **K-Means clustering** to group children based on similar attributes:

- Used **Elbow Method** to determine optimal k (=5).
- Applied **PCA** for 2D visualization of clusters.
- **Silhouette Score**: 0.58 (moderate separation).
- Insights:
  - Cluster 1: Underweight, high risk of malnutrition.
  - Cluster 2: Healthy weight, low risk.
- Plotted malnourished distribution across clusters.

### 6. 🔍 Model Interpretation

#### ✅ SHAP (SHapley Additive exPlanations)
- Applied to XGBoost model to understand feature contributions.
- Features like **weight**, **height**, and **mother's education** had the strongest impact.

#### 📊 Feature Importance
- Visualized importance scores from XGBoost before and after optimization.

### 7. ⚙️ Hyperparameter Tuning

Used **GridSearchCV** to improve XGBoost:

- Tuned `n_estimators`, `max_depth`, `learning_rate`, etc.
- Compared ROC curves before vs. after tuning.
- Reduced **overfitting** with `gamma`, `reg_alpha`, `reg_lambda`.

| Metric | Before | After |
|--------|--------|-------|
| AUC | 0.88 | 0.92 |
| Test Accuracy | 84% | 88% |

### 8. 🧠 Final Model & Deployment

- Final optimized XGBoost model saved with **Joblib**.
- Can be loaded for real-time predictions or deployed in a web application.

---

## 📌 Results Summary

- XGBoost outperformed other models in both classification and interpretability.
- Regression models revealed feature influence on malnutrition scores.
- K-Means clustering helped uncover hidden patterns in the dataset.
- SHAP and PCA enhanced model transparency and stakeholder trust.

---

## 📁 Folder Structure
📦 Healthcare-Child-Malnutrion-Prediction
│
├── data/
│ ├── child_health_raw.csv
│ └── child_health_cleaned.csv
│
├── notebooks/
│ └── malnutrition_prediction_colab.ipynb
│
├── models/
│ └── optimized_xgboost_model.pkl
│
├── images/
│ └── plots, SHAP, PCA visuals
│
└── README.md
