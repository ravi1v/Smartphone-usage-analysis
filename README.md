# 📱 SmartSense — Smartphone Addiction Risk Predictor

AI-powered **Smartphone Addiction Detection System** built using a **Stacking Ensemble Machine Learning Model** and deployed with **Streamlit**.

This application analyzes user behavioral patterns such as screen time, social media usage, notifications, and sleep patterns to estimate the **probability of smartphone addiction**.

---

# 🚀 Project Overview

Smartphone addiction is becoming a growing concern in modern society. Excessive smartphone usage can affect productivity, sleep quality, and mental health.

**SmartSense** uses machine learning to predict whether a user is likely to be addicted to smartphone usage based on behavioral metrics.

The system provides:

- 📊 Risk probability score
- ⚠️ Addiction classification
- 📈 Behavioral analytics dashboard
- 💡 Personalized recommendations

---

# 🧠 Machine Learning Model

The prediction model is a **Stacking Ensemble Classifier** combining multiple algorithms.

## Base Models

- GradientBoostingClassifier
- RandomForestClassifier
- ExtraTreesClassifier

## Meta Model

- LogisticRegression

## Model Architecture

```
Input Features
      ↓
GradientBoostingClassifier
RandomForestClassifier
ExtraTreesClassifier
      ↓
Meta Learner (Logistic Regression)
      ↓
Final Prediction
```

---

# 📊 Model Performance

| Metric | Score |
|------|------|
| Accuracy | **93.93%** |
| F1 Score | **0.9567** |
| ROC-AUC | **0.9897** |

## Classification Report

```
              precision    recall  f1-score   support

           0       0.88      0.92      0.90       438
           1       0.97      0.95      0.96      1062

    accuracy                           0.94      1500
   macro avg       0.92      0.93      0.93      1500
weighted avg       0.94      0.94      0.94      1500
```

---

# 📂 Dataset

The dataset was obtained from **Kaggle** and contains behavioral smartphone usage information.

Dataset size:

```
7500 samples
17 total features
```

## Core Features

```
age
gender
daily_screen_time_hours
social_media_hours
gaming_hours
work_study_hours
sleep_hours
notifications_per_day
app_opens_per_day
weekend_screen_time
stress_level
academic_work_impact
```

## Engineered Features

```
social_media_screen_ratio
gaming_screen_ratio
pickups_per_hour
sleep_deprivation_risk
high_social_media
```

---

# ⚙️ Feature Engineering

Several behavioral indicators were derived from raw data.

Example:

```python
df["social_media_screen_ratio"] = df["social_media_hours"] / (df["daily_screen_time_hours"] + 0.01)

df["gaming_screen_ratio"] = df["gaming_hours"] / (df["daily_screen_time_hours"] + 0.01)

df["pickups_per_hour"] = df["app_opens_per_day"] / 16

df["sleep_deprivation_risk"] = (df["sleep_hours"] < 6).astype(int)

df["high_social_media"] = (df["social_media_hours"] > 3).astype(int)
```

---

# 🖥️ Streamlit Web Application

The model is deployed using **Streamlit** to provide an interactive dashboard.

## Features

- Interactive user input panel
- Risk visualization gauge
- Behavioral radar analysis
- Prediction history tracking
- Personalized recommendations

---

# 📦 Project Structure

```
SmartSense/
│
├── app.py
├── smartphone_addiction_model.pkl
├── train_model.py
├── dataset.csv
├── requirements.txt
└── README.md
```

---

# 💾 Saving the Model

The trained stacking model is saved using **Joblib**.

```python
import joblib

joblib.dump(stack_model, "smartphone_addiction_model.pkl")
```

Load the model in the Streamlit app:

```python
model = joblib.load("smartphone_addiction_model.pkl")
```

---

# ▶️ Running the Application

## Clone the repository

```bash
git clone https://github.com/ravi1v/Smartphone-usage-analysis.git

```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run Streamlit

```bash
streamlit run app.py
```

---

# 📈 Risk Score Interpretation

| Risk Score | Interpretation |
|------------|---------------|
| 0-30 | Low Risk |
| 30-50 | Mild Risk |
| 50-75 | Moderate Risk |
| 75-100 | High Risk |

---

# 🛠 Technologies Used

```
Python
Scikit-Learn
Pandas
NumPy
Plotly
Streamlit
Joblib
```

---

# 🔬 Future Improvements

- Deep learning behavioral modeling
- Real-time smartphone usage tracking
- Mobile app integration
- Model explainability with SHAP

---

# 👨‍💻 Author

Ravi Shankar

Machine Learning & Data Science Enthusiast

---

⭐ If you like this project, consider giving the repository a star.
