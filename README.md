# 🧠 Parkinson’s Disease Prediction using Machine Learning

## Overview

This project uses machine learning to predict Parkinson’s disease based on biomedical voice measurements. Parkinson’s is a progressive neurological disorder that affects movement and speech, making early detection challenging. By analyzing vocal features, this project demonstrates how data-driven approaches can assist in non-invasive diagnosis.

## Live Demo

-> https://cpatte77.github.io/DataMining-Analyses/Parkinson's_Disease_Project

> View the full interactive notebook with visualizations, model outputs, and analysis.

---

## Project Highlights

* Cleaned and processed real-world biomedical voice data
* Performed exploratory data analysis to uncover key patterns
* Identified important vocal features correlated with Parkinson’s
* Built and evaluated classification models to predict disease presence
* Demonstrated how machine learning can support early detection

---

## Dataset

The dataset consists of biomedical voice measurements from individuals, where each row represents a voice recording and each feature captures a specific vocal characteristic (e.g., frequency, jitter, shimmer). The goal is to classify individuals as healthy or having Parkinson’s disease. ([Kaggle][1])

---

## Key Results

* Models successfully distinguished between healthy individuals and Parkinson’s patients
* Certain voice-based features showed strong predictive importance
* Feature selection improved efficiency while maintaining performance
* Results highlight the potential of speech analysis for early disease detection

---

## Tech Stack

* Python
* Pandas & NumPy
* Scikit-learn
* Matplotlib & Seaborn
* Jupyter Notebook

---

## Methodology

1. Data preprocessing and cleaning
2. Exploratory data analysis (EDA)
3. Feature selection and scaling
4. Model training (classification algorithms)
5. Model evaluation using accuracy, precision, and recall

Machine learning models are commonly used in Parkinson’s detection because they can identify subtle patterns in data (like voice signals) that are difficult to detect manually. ([PMC][2])

---

## Sample Code

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)
```

---

## What I Learned!!

* How to apply machine learning to real-world healthcare datasets
* The importance of feature selection in improving model performance
* How data science can uncover patterns not easily visible through traditional analysis
* The potential impact of AI in early disease detection
