📦 E-Commerce Customer Churn Risk Prediction using Machine Learning
📌 Project Overview

This project predicts customer churn risk for an e-commerce platform using machine learning. By analyzing customer behavior, purchase history, and engagement metrics, the model helps businesses identify high-risk customers early and take proactive retention actions. 💡

🛠️ Libraries Used
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

📂 Dataset

File: synthetic_ecommerce_churn_dataset.csv

Description: Synthetic dataset containing e-commerce customer behavior and churn risk.

Key Columns:

🆔 customer_id – Unique customer identifier

📅 customer_since – Date when the customer joined

⏱️ last_purchase_days – Days since last purchase

💰 avg_order_value – Average value of orders

📧 email_open_rate – Email engagement rate

🛒 total_orders – Total number of orders

📱 preferred_device – Customer’s preferred device

💳 payment_method – Most used payment method

⚠️ churn_risk – Target variable (0–1 scale, risk of churn)

🔍 Data Exploration & Cleaning

🧹 Filled missing values for email_open_rate with median

📆 Converted customer_since to customer_tenure_days

❌ Dropped customer_id and original customer_since

📊 Visualized distributions and correlations using Seaborn

Example plots:

📈 Churn Risk Distribution

🌡️ Feature Correlation Heatmap

🕒 Churn Risk vs Last Purchase Days

📱 Churn Risk by Device & 💳 Payment Method

🏗️ Feature Engineering

Split features (X) and target (y)

Identified numeric and categorical columns

Built preprocessing pipeline using StandardScaler for numeric features and OneHotEncoder for categorical features 🔧

✂️ Train/Test Split

80% training, 20% testing

random_state=42 for reproducibility 🔄

🤖 Machine Learning Models
1️⃣ Linear Regression

Simple regression baseline

Evaluation:

MAE: <value>
RMSE: <value>
R2 Score: <value>

2️⃣ Random Forest Regressor 🌲

Ensemble model capturing non-linear relationships

Evaluation:

MAE: <value>
RMSE: <value>
R2 Score: <value>


Random Forest performed better than Linear Regression ✅

📈 Model Visualization

Actual vs Predicted Churn Risk:

Other visualizations include:

Scatterplots for ⏱️ last_purchase_days and 💰 avg_order_value

Boxplots for 📱 preferred_device and 💳 payment_method

✅ Key Insights

🔑 Top Predictors: Loyalty score, last purchase days, total orders

📧 Marketing Insight: Email engagement affects churn risk

💡 Business Impact: Identify high-risk customers for retention campaigns

🚀 How to Run

Clone the repository:

git clone https://github.com/yourusername/ecommerce-churn-ml.git
cd ecommerce-churn-ml


Install dependencies:

pip install -r requirements.txt


Run Jupyter Notebook:

jupyter notebook


Open E-Commerce_Customer_Churn_Risk_Prediction.ipynb and follow the workflow 📝
