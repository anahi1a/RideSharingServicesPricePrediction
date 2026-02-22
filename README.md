# Ride Sharing Services Price Prediction
**Anahita Gupta**  
**Sanchita Ghosh**

This project predicts ride-sharing fares using machine learning, with a focus on incorporating weather-related factors. Historical ride data is integrated with meteorological data to build a price prediction model using Random Forest Regression. The solution covers data preprocessing, feature engineering, model training, evaluation, and deployment through a Flask-based web application.

---

## Project Overview

Ride-sharing prices are influenced by multiple dynamic factors such as:

- Distance
- Cab type
- Surge multiplier
- Source and destination
- Weather conditions (temperature, humidity, rain, wind, pressure)

This project combines ride data with weather data to improve prediction accuracy and better capture real-world pricing dynamics.

---

## Methodology

### 1. Data Integration
- Cab ride dataset (`cab_rides.csv`)
- Weather dataset (`weather.csv`)
- Weather features aggregated by location
- Source and destination weather merged into ride data

### 2. Data Preprocessing
- Handling missing values
- Label encoding categorical variables
- Feature scaling using `StandardScaler`
- Train-test split (70-30)

### 3. Model Training
- Random Forest Regressor
- 100 estimators
- Evaluation using:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R² Score

### 4. Model Deployment
- Trained model saved using `joblib`
- Flask backend for prediction API
- HTML + CSS frontend

##  Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Run the Jupyter notebook:

```bash
main.ipynb
```

### 3. Run the Flask Application

```bash
python app.py
```
