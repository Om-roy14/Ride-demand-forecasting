# 🚕 Ride Request Demand Prediction

This machine learning project predicts the number of ride requests (e.g., for Ola or Uber) based on factors such as time, date, and other environmental or user-related features.

---

## 📌 Problem Statement

Given a dataset of ride requests with timestamps, the goal is to predict demand volume at a specific hour. This helps ride-sharing platforms allocate drivers more efficiently.

---

## 📊 Dataset Used

- Input CSV: `ola.csv`
- Key features extracted:
  - `datetime` split into `date`, `time`
  - Location-based features
  - Weather/traffic/load (if included)

---

## 🛠️ Tools & Libraries

- Python
- pandas, NumPy
- scikit-learn
- seaborn, matplotlib
- SVM, Linear Regression, Random Forest Regressor

---

## 🔍 Model Training

The project includes:
- Data preprocessing (`LabelEncoder`, `StandardScaler`)
- Feature extraction from timestamps
- Multiple regression models:
  - Support Vector Regressor
  - Linear Regression
  - Random Forest Regressor
- Error evaluation using:
  - Mean Absolute Error (MAE)
  - R² Score

