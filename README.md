# 🏠 Predicting Sale Price of Houses (AMES Dataset)

This Streamlit application predicts house sale prices using the **Ames Housing Dataset**.  
It integrates **Exploratory Data Analysis (EDA)**, preprocessing, insights visualization, model training, evaluation, and deployment of ML models.

---

## 🚀 Features
- **Data Loading & Cleaning**
  - Load Ames Housing dataset
  - Handle missing values & duplicates
  - Drop unwanted columns via interactive checkboxes
  - Outlier detection with boxplots  

- **Exploratory Data Analysis (EDA)**
  - Descriptive statistics
  - Correlation analysis with target variable
  - Grouped insights (Year Built, Year Sold, etc.)

- **Visual Insights**
  - Heatmaps (sampled features)
  - Barplots & boxplots for relationships
  - Average Sale Price by year  

- **Machine Learning Models**
  - Linear Regression  
  - ElasticNet (with GridSearchCV for tuning)  
  - Support Vector Regressor (SVR with GridSearchCV)  
  - Cross-validation with MAE, RMSE, and R² metrics  

- **Model Deployment**
  - Save trained model (`joblib`)  
  - Single-click deploy via Streamlit  

---

## 📦 Tech Stack
- **Python** (NumPy, Pandas, Seaborn, Matplotlib)  
- **Machine Learning** (Scikit-Learn, Pipeline, GridSearchCV)  
- **Deployment** (Streamlit, Joblib)  

---

## 🖥️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/house-price-prediction.git
   cd house-price-prediction
