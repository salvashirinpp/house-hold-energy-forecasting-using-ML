# 🏠 Household Energy Consumption Forecasting System

A **production-ready machine learning web application** that forecasts **next-day household energy consumption** using historical appliance usage data, lighting patterns, and optional environmental variables.

The system is built with **LightGBM**, served via a **Flask REST API**, and designed to reflect **real-world smart home energy analytics workflows**.

---

## 📌 Project Overview

Accurate energy forecasting is critical for optimizing household electricity usage and reducing operational costs.

This project implements a **regression-based forecasting system** that predicts **next-day energy consumption** using recent historical patterns and contextual features.

### The application integrates:
- Feature-engineered time-series inputs  
- A trained **LightGBM regression model**  
- Scalable preprocessing and inference pipeline  
- Web-based deployment using **Flask**

---

## 🚀 Key Features

- Predicts **next-day energy consumption (Wh)**
- Uses **7 days of historical appliance usage**
- Incorporates **lighting usage** (most influential predictor)
- Supports **optional environmental variables**
- Robust **feature scaling and validation**
- RESTful API for easy integration
- Modular and **production-oriented code structure**

---

## 🤖 Machine Learning Approach

- **Model Type:** Gradient Boosting Regression (LightGBM)  
- **Problem Type:** Supervised Regression  
- **Target Variable:** Household energy consumption (Wh)

### Feature Engineering Techniques:
- Lag features (1, 3, 7 days)
- Rolling statistics (mean, standard deviation)
- Calendar features (weekday, month, quarter, weekend)
- Environmental conditions (optional)

Preprocessing artifacts and model metadata are persisted using **joblib** to ensure consistency during inference.

---

## 🧰 Technology Stack

### Programming Language
- Python

### Backend & Deployment
- Flask
- REST API

### Machine Learning & Data
- LightGBM
- Scikit-learn
- Pandas
- NumPy

### Model Persistence
- Joblib

---

## 📁 Project Structure



```
energy_forecasting/
├── forecast_energy.ipynb      # Complete forecasting script (single file)
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── data/
│   └── energydata.csv     # Dataset (10-minute intervals)
└── models/                # Trained models (generated after running forecast_energy.py)
```

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the dataset is available**:
   - Place `energydata.csv` in the `data/` directory
   - The dataset should have a 'date' column and 'Appliances' column

## Usage

### Step 1: Train the Model

Run the complete forecasting notebook to train models and generate predictions:

This notebook will:
- Load and preprocess the data (resample to daily, handle outliers)
- Perform exploratory data analysis with visualizations
- Engineer features (lags, rolling stats, date features)
- Train multiple models (Random Forest, XGBoost, LightGBM)
- Evaluate models and select the best one
- Save the best model and required files for the Streamlit app

**Generated Files:**
- `energy_forecast_model_*.pkl` - Trained model
- `scaler.pkl` - Feature scaler
- `feature_columns.pkl` - Feature column names
- `model_metadata.pkl` - Model metadata

### Step 2: Run the Streamlit App

Launch the interactive web application:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Streamlit App Features

- **Beautiful UI**: Modern, gradient-styled interface
- **Input Parameters**:
  - Date picker for tomorrow's date
  - Last 7 days of energy consumption (most recent first)
  - Weather conditions (temperature, humidity, windspeed)
  - Optional indoor sensor data
- **Real-time Prediction**: Get instant energy consumption forecasts
- **Visualizations**: Input summary and trend charts
- **Error Handling**: Robust validation and error messages

## Model Details

### Preprocessing
- Resample 10-minute data to daily frequency
- Sum 'Appliances' and 'lights', mean for other features
- Forward fill missing values
- Winsorization (clip at 1% and 99%) for outlier handling
- Optional log1p transform for highly skewed targets

### Feature Engineering
- **Date Features**: day_of_week, month, quarter, is_weekend, day_of_month
- **Lag Features**: 1-day, 3-day, and 7-day lags
- **Rolling Features**: 7-day rolling mean and standard deviation
- **Sequence Features**: Last 7 days as separate columns

### Models
- **Random Forest Regressor**: 100 trees, default parameters
- **XGBoost Regressor**: 100 estimators, default parameters
- **LightGBM Regressor**: 100 estimators, default parameters

### Evaluation Metrics
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)

## Dataset

The UCI Energy Consumption dataset contains:
- 10-minute interval measurements
- Target variable: `Appliances` (energy consumption in Wh)
- Features: Indoor temperatures (T1-T9), Indoor humidity (RH_1-RH_9), Outdoor weather, lights, etc.

## Notes

- The forecasting script follows time-series best practices:
  - No data leakage
  - Temporal train-test split (last 20% as test set)
  - No using future data in features
- All code is in a single file (`forecast_energy.py`) - no custom modules required
- The Streamlit app requires the model files generated by `forecast_energy.py`

## Requirements

- Python 3.8+
- See `requirements.txt` for package versions

## Troubleshooting

**Model files not found error:**
- Make sure you've run `forecast_energy.py` first to generate model files
- Check that all `.pkl` files are in the project root directory

**Dataset not found:**
- Ensure `energydata.csv` is in the `data/` directory
- Check that the CSV has a 'date' column (or 'Date') and 'Appliances' column

**Import errors:**
- Install all dependencies: `pip install -r requirements.txt`
- XGBoost and LightGBM are optional - the script will skip them if not available

## License

This project is for educational and research purposes.

## Author

Built with ❤️ for energy forecasting

## Link

http://127.0.0.1:5000/

## Demo

<img width="1306" height="903" alt="Screenshot 2025-12-30 093757" src="https://github.com/user-attachments/assets/9e32278a-ca20-43ac-9f46-fb4e8e6049bd" />


<img width="1270" height="916" alt="Screenshot 2025-12-30 093906" src="https://github.com/user-attachments/assets/da358fb9-6b5d-426a-8b38-f72943137d63" />


<img width="1336" height="911" alt="Screenshot 2025-12-30 093833" src="https://github.com/user-attachments/assets/27743f27-b6dc-4e8a-8d5b-1835eb842780" />



