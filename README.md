# 💤 Prediction and Analysis of Sleep Health using ML & LLMs

An end-to-end machine learning project for **predicting sleep quality, identifying sleep behavior patterns, explaining model predictions, and generating natural-language sleep insights using an LLM**.

The project combines regression models, K-Means clustering, cross-validation, SHAP-based explainability, and Google Gemini to build an interpretable sleep and lifestyle analysis pipeline.

> **Note:** This project is intended for data analysis and educational purposes. It does not provide medical diagnosis or clinical advice.

---

## 📌 Overview

Sleep quality is influenced by multiple lifestyle and physiological factors. This project explores these relationships using machine learning and explainable AI.

The pipeline takes sleep and lifestyle data and performs:

* Exploratory data analysis
* Data cleaning and categorical encoding
* Sleep quality prediction
* Comparison of multiple regression models
* Hyperparameter tuning using GridSearchCV
* Cross-validation
* Sleep behavior clustering using K-Means
* Model explainability using SHAP
* LLM-generated natural-language interpretation and recommendations

### Overall Pipeline

```text
Sleep & Lifestyle Dataset
          ↓
Data Cleaning & Encoding
          ↓
Exploratory Data Analysis
          ↓
Feature Selection
          ↓
Train / Test Split
          ↓
Feature Scaling
          ↓
Regression Models
          ↓
Model Comparison & Cross Validation
          ↓
Best Model Selection
          ↓
SHAP Explainability
          ↓
K-Means Clustering
          ↓
Gemini LLM Interpretation
```

---

## 🚀 Key Features

### 1. Sleep Quality Prediction

Predicts **Quality of Sleep** using the following features:

* Sleep Duration
* Physical Activity Level
* Stress Level
* Heart Rate
* Daily Steps
* Age

Multiple regression algorithms are trained and compared.

---

### 2. Multiple Regression Models

The project implements and compares:

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor
* Support Vector Regressor (SVR)
* XGBoost Regressor

Tree-based and SVR models are tuned using **GridSearchCV** with cross-validation.

---

### 3. Model Evaluation

Each regression model is evaluated using:

* MAE — Mean Absolute Error
* MSE — Mean Squared Error
* RMSE — Root Mean Squared Error
* R² Score
* Adjusted R² Score

Actual-vs-predicted plots and residual analysis are also performed.

---

### 4. Model Comparison

The models are compared using their test-set performance and R² scores.

| Model                    |  R² Score |  RMSE |   MAE |
| ------------------------ | --------: | ----: | ----: |
| Decision Tree Regressor  | **1.000** | 0.000 | 0.000 |
| Support Vector Regressor |     0.999 | 0.028 | 0.014 |
| XGBoost Regressor        |     0.998 | 0.059 | 0.019 |
| Random Forest Regressor  |     0.991 | 0.116 | 0.036 |
| Linear Regression        |     0.916 | 0.357 | 0.286 |

The notebook selects the model with the highest test-set R² score as the model used for subsequent prediction interpretation.

**Selected model in the current notebook:** Decision Tree Regressor.

---

## 🔁 Cross-Validation

To evaluate model performance beyond a single train/test split, the project also performs **5-fold cross-validation**.

The models are evaluated using mean R² across the folds.

| Model                    | Mean CV R² |
| ------------------------ | ---------: |
| Linear Regression        |      0.902 |
| Decision Tree Regressor  |      0.980 |
| Random Forest Regressor  |      0.982 |
| Support Vector Regressor |      0.984 |
| XGBoost Regressor        |      0.983 |

Hyperparameter tuning is performed inside the cross-validation workflow for the tree-based and SVR models.

---

## 🔍 Exploratory Data Analysis

The project performs exploratory analysis to understand the relationships between sleep and lifestyle variables.

Visualizations include:

* Sleep duration distribution
* Sleep quality distribution
* Sleep disorder distribution
* Gender distribution
* Correlation heatmap
* Sleep duration vs. sleep quality

These analyses help identify patterns before applying machine learning.

---

## 🧩 Sleep Behavior Clustering

K-Means clustering is used to identify groups of individuals with similar sleep and lifestyle characteristics.

### Features Used

* Sleep Duration
* Physical Activity Level
* Stress Level
* Heart Rate
* Daily Steps
* Age

The features are standardized before clustering.

### Clustering Process

```text
Selected Lifestyle Features
          ↓
StandardScaler
          ↓
Elbow Method
          ↓
K-Means
          ↓
4 Clusters
          ↓
Cluster Analysis
```

The current implementation uses **4 clusters**, selected after examining the elbow curve.

Cluster characteristics are summarized using average values for sleep and lifestyle variables.

### Clustering Evaluation

The current notebook reports a:

**Silhouette Score: 0.49**

Cluster visualizations are also generated using sleep duration and sleep quality.

> The clusters are data-driven groups. They are not automatically labeled as "good", "average", or "poor" sleepers.

---

## 🧠 Explainable AI with SHAP

The project uses **SHAP (SHapley Additive exPlanations)** to understand how individual features contribute to model predictions.

SHAP analysis includes:

* Global feature importance
* Feature importance bar plot
* Individual prediction explanations
* SHAP waterfall plots

This makes the regression model easier to interpret instead of treating it as a black box.

### Example Flow

```text
User Sleep & Lifestyle Data
            ↓
Best Regression Model
            ↓
Sleep Quality Prediction
            ↓
SHAP Analysis
            ↓
Important Contributing Features
```

---

## 🤖 LLM Integration

The project integrates **Google Gemini 2.5 Flash** to convert machine learning outputs into natural-language explanations.

The LLM receives information including:

* Predicted sleep quality
* Assigned sleep cluster
* Regression model used
* Model performance
* Sleep duration
* Physical activity level
* Stress level
* Heart rate
* Daily steps
* SHAP-based feature contributions

The generated output is structured into sections such as:

1. Sleep Health Summary
2. Interpretation of Sleep Cluster
3. Factors Influencing the Prediction
4. Personalized Recommendations
5. Lifestyle Suggestions
6. Conclusion
7. Disclaimer

The prompt also instructs the model not to diagnose diseases or prescribe medication.

---

## 📊 Dataset

The project uses the **Sleep Health and Lifestyle Dataset**.

The dataset is downloaded programmatically using KaggleHub.

### Main Features

| Feature                 | Description                      |
| ----------------------- | -------------------------------- |
| Gender                  | Gender of the individual         |
| Age                     | Age of the individual            |
| Sleep Duration          | Average sleep duration in hours  |
| Quality of Sleep        | Sleep quality score              |
| Physical Activity Level | Physical activity level          |
| Stress Level            | Reported stress level            |
| BMI Category            | BMI category                     |
| Blood Pressure          | Blood pressure information       |
| Heart Rate              | Heart rate                       |
| Daily Steps             | Number of daily steps            |
| Sleep Disorder          | Recorded sleep disorder category |

For the prediction task, the model uses:

```text
Sleep Duration
Physical Activity Level
Stress Level
Heart Rate
Daily Steps
Age
```

Target:

```text
Quality of Sleep
```

---

## 🧹 Data Preparation

The notebook performs several preprocessing steps:

* Loads the dataset using Pandas
* Inspects dataset shape and column types
* Checks for missing values
* Checks for duplicate records
* Removes duplicate rows
* Encodes categorical variables using `LabelEncoder`
* Standardizes features where required for regression/clustering workflows

---

## ⚙️ Machine Learning Methodology

### Step 1 — Data Preparation

The dataset is loaded and inspected for:

* Missing values
* Duplicate records
* Data types
* Statistical characteristics

Duplicate records are removed and categorical columns are encoded.

### Step 2 — Feature Selection

Six sleep/lifestyle variables are selected as predictors:

```python
[
    "Sleep Duration",
    "Physical Activity Level",
    "Stress Level",
    "Heart Rate",
    "Daily Steps",
    "Age"
]
```

### Step 3 — Train/Test Split

The dataset is divided into training and testing sets.

### Step 4 — Feature Scaling

`StandardScaler` is used where feature scaling is required.

### Step 5 — Model Training

Five regression algorithms are trained.

### Step 6 — Hyperparameter Tuning

GridSearchCV is used to search parameter combinations for:

* Decision Tree
* Random Forest
* SVR
* XGBoost

### Step 7 — Evaluation

Models are evaluated using multiple regression metrics.

### Step 8 — Explainability

SHAP is applied to the selected trained model.

### Step 9 — Clustering

K-Means is applied to identify sleep/lifestyle behavior groups.

### Step 10 — LLM Interpretation

Gemini 2.5 Flash generates a human-readable interpretation using the prediction, cluster information, and SHAP contributions.

---

## 📈 Model Diagnostics

Beyond basic performance metrics, the project performs residual analysis on the selected regression model.

This includes:

* Residual vs. predicted plot
* Residual distribution
* Absolute prediction error distribution
* R² comparison across models

These visualizations help inspect prediction behavior and model errors.

---

## 🛠️ Tech Stack

### Programming

* Python 3.x

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost

### Clustering

* K-Means

### Explainable AI

* SHAP

### LLM

* Google Gemini API
* Gemini 2.5 Flash

### Visualization

* Matplotlib
* Seaborn

### Development

* Jupyter Notebook
* Google Colab

### Dataset Access

* KaggleHub

---

## 📂 Project Structure

```text
sleep-health-ai/
│
├── sleep_health_analysis.ipynb
├── requirements.txt
└── README.md
```

If the repository contains additional folders such as `data/`, `models/`, or `visualizations/`, they can be added to this structure accordingly.

---

## 👨‍💻 What I Built

I built the end-to-end machine learning and explainability workflow for analyzing sleep and lifestyle data.

My work includes:

* Loading and preparing the sleep health dataset
* Performing exploratory data analysis
* Handling duplicate records
* Encoding categorical variables
* Selecting relevant prediction features
* Training five regression models
* Performing hyperparameter tuning with GridSearchCV
* Comparing models using MAE, MSE, RMSE, R², and Adjusted R²
* Performing 5-fold cross-validation
* Selecting the best-performing regression model
* Performing residual and prediction-error analysis
* Implementing K-Means clustering
* Evaluating clustering using the Silhouette Score
* Using SHAP to explain model predictions
* Integrating Gemini 2.5 Flash for natural-language interpretation
* Combining model predictions, clustering results, and SHAP explanations into an AI-generated sleep analysis

The project focuses on building an interpretable ML pipeline rather than only training a single predictive model.

---

## 💡 Key Takeaways

The project demonstrates how multiple machine learning techniques can be combined into a single analytical workflow:

```text
Data Analysis
     +
Regression
     +
Hyperparameter Tuning
     +
Cross-Validation
     +
Clustering
     +
Explainable AI
     +
LLM
     ↓
Interpretable Sleep Analysis
```

This approach allows the system to provide not only a prediction, but also additional context about **patterns in the data and factors contributing to individual predictions**.

---

## 🔮 Future Improvements

Potential improvements include:

* Build a dedicated web dashboard for interactive predictions
* Create a FastAPI backend for model inference
* Deploy the application as a containerized service
* Add real-time user input and prediction
* Improve clustering interpretation and automatic cluster labeling
* Experiment with additional regression and ensemble models
* Add systematic feature selection
* Add automated model retraining
* Store trained models for production inference
* Improve LLM grounding and output validation
* Add authentication and user-specific analysis

---

## ⚠️ Disclaimer

This project is intended for **educational and analytical purposes**.

The predictions and LLM-generated explanations are based on the dataset and machine learning models used in the project. They should not be treated as medical diagnoses, professional medical advice, or a substitute for consultation with a qualified healthcare professional.

---

## ⭐ Project Goal

The goal of this project is to demonstrate an end-to-end approach to combining:

**Machine Learning + Clustering + Explainable AI + LLMs**

to transform structured sleep and lifestyle data into **predictions, behavioral patterns, model explanations, and human-readable insights**.

---

## 👤 Author

**NUHA TAMANNA ABDUL**
