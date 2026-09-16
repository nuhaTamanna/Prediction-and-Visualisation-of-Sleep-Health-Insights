# Prediction and Visualization of Sleep Health using ML & LLMs

An AI-powered system for predicting, analyzing, and visualizing sleep health using Machine Learning (ML) and Large Language Models (LLMs).

This project combines regression, clustering, interactive dashboards, and natural language summarization to provide personalized sleep insights and recommendations.

---

## 📌 Overview

Sleep health is a critical factor affecting physical health, mental wellness, and cognitive performance. This project presents an intelligent framework capable of:

- Predicting sleep quality
- Grouping users based on sleep behavior
- Visualizing sleep analytics
- Generating natural-language insights using LLMs

The system leverages:
- Linear Regression for prediction
- K-Means Clustering for behavioral segmentation
- GPT-based LLM integration for explainable insights

---

## 🚀 Features

### ✅ Sleep Quality Prediction
Predicts overall sleep quality based on:
- Sleep Duration
- Deep Sleep Hours
- REM Sleep Hours
- Awake Count

### ✅ Sleep Pattern Clustering
Groups users into:
- Excellent Sleepers
- Average Sleepers
- Poor Sleepers

### ✅ Interactive Dashboard
Provides:
- Correlation heatmaps
- Scatter plots
- Cluster visualizations
- Sleep trend analysis

### ✅ LLM-Generated Recommendations
Converts numerical results into:
- Human-readable summaries
- Sleep improvement recommendations
- Behavioral insights

---

## 🧠 Technologies Used

### Programming Language
- Python 3.x

### Libraries
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Plotly / Recharts

### Machine Learning Models
- Linear Regression
- K-Means Clustering

### AI / LLM
- GPT-5-mini API

### Tools & Platforms
- Jupyter Notebook
- VS Code
- FastAPI
- Docker

---

## 📊 Dataset Description

The dataset contains sleep-related parameters collected from publicly available repositories.

### Features

| Feature | Description |
|---|---|
| SleepDuration | Total sleep duration (hours) |
| DeepSleepHours | Time spent in deep sleep |
| REMHours | REM sleep duration |
| AwakeCount | Number of awakenings |
| SleepQuality | Overall sleep efficiency (%) |

### Data Preprocessing
- Missing value handling
- Z-score normalization
- Outlier removal
- Feature scaling

---

## ⚙️ Methodology

### 1. Data Collection & Preprocessing
- Data cleaning
- Handling missing values
- Outlier removal
- Feature normalization
- Train-validation-test split

### 2. Linear Regression
Used for predicting Sleep Quality.

#### Evaluation Metrics
- R² Score
- Mean Squared Error (MSE)

### 3. K-Means Clustering
Used to classify users into sleep categories.

#### Cluster Categories
- Poor Sleepers
- Average Sleepers
- Excellent Sleepers

### 4. LLM Integration
The Large Language Model:
- Explains analytical outputs
- Generates personalized recommendations
- Summarizes sleep trends

---

## 📈 Results

| Metric | Value |
|---|---|
| R² Score | 0.82 |
| Mean Squared Error | 4.2 |
| Silhouette Score | 0.71 |

### Key Findings
- Deep Sleep Hours significantly influence sleep quality.
- Frequent awakenings reduce sleep efficiency.
- Longer sleep duration correlates with higher sleep quality.

---

## 🖥️ Dashboard Functionalities

The dashboard provides:
- Sleep prediction visualization
- Cluster analysis
- Correlation heatmaps
- LLM-generated insights
- Interactive analytics

Example insight:
> "Your sleep pattern indicates moderate deep sleep but frequent awakenings. Reducing nighttime interruptions may improve overall sleep quality."

---

## 📂 Project Structure

```bash
sleep-health-ai/
│
├── data/
├── notebooks/
├── models/
├── dashboard/
├── api/
├── visualizations/
├── requirements.txt
└── README.md
