# app.py - Energy Forecasting Web App (Optimized from Feature Importance)

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

app = Flask(__name__)

# ============================
# LOAD MODEL ARTIFACTS
# ============================
MODEL_DIR = 'model'
FILES = [
    'energy_forecast_lightgbm_final.pkl',
    'scaler_full.pkl',
    'feature_columns.pkl',
    'model_metadata.pkl'
]

missing = [f for f in FILES if not os.path.exists(os.path.join(MODEL_DIR, f))]
if missing:
    raise FileNotFoundError(f"Missing files in 'model/': {missing}")

model = joblib.load(os.path.join(MODEL_DIR, 'energy_forecast_lightgbm_final.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler_full.pkl'))
feature_columns = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))
metadata = joblib.load(os.path.join(MODEL_DIR, 'model_metadata.pkl'))

model_type = metadata.get('model_type', 'LightGBM')
APPLY_LOG_TRANSFORM = metadata.get('apply_log_transform', False)

print(f"Model loaded: {model_type}")
print(f"Features: {len(feature_columns)}")

# ============================
# PREDICTION FUNCTION
# ============================
def predict_next_day(recent_7_days, lights_today, predict_date_str, advanced=None):
    if len(recent_7_days) != 7:
        raise ValueError("Exactly 7 days required")

    predict_date = pd.to_datetime(predict_date_str)
    features = {}

    # Date features
    features['day_of_week'] = predict_date.dayofweek
    features['month'] = predict_date.month
    features['quarter'] = predict_date.quarter
    features['is_weekend'] = 1 if predict_date.dayofweek >= 5 else 0
    features['day_of_month'] = predict_date.day

    # Critical historical features
    for i in range(1, 8):
        features[f'Appliances_seq_{i}'] = recent_7_days[i-1]
        if i in [1, 3, 7]:
            features[f'Appliances_lag_{i}'] = recent_7_days[i-1]

    arr = np.array(recent_7_days)
    features['Appliances_rolling_mean_7'] = arr.mean()
    features['Appliances_rolling_std_7'] = arr.std(ddof=1) if len(arr) > 1 else 0

    # Lights (most important feature!)
    features['lights'] = lights_today

    # Advanced optional inputs (high importance)
    if advanced:
        if 'bedroom_temp' in advanced:
            features['BEDROOM_TEMP'] = advanced['bedroom_temp']
        if 'windspeed' in advanced:
            features['Windspeed'] = advanced['windspeed']
        if 'pressure' in advanced:
            features['Press_mm_hg'] = advanced['pressure']

    # Fill missing with 0 (safe default)
    df_input = pd.DataFrame([features])
    for col in feature_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[feature_columns]

    scaled = scaler.transform(df_input)
    pred = model.predict(scaled)[0]

    if APPLY_LOG_TRANSFORM:
        pred = np.expm1(pred)

    return round(float(pred), 1)

# ============================
# ROUTES
# ============================
@app.route('/')
def home():
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    return render_template('index.html', tomorrow=tomorrow)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        recent_7_days = data['recent_7_days']           # list of 7 numbers (most recent first)
        lights_today = float(data['lights'])
        predict_date = data['predict_date']

        advanced = {
            'bedroom_temp': data.get('bedroom_temp'),
            'windspeed': data.get('windspeed'),
            'pressure': data.get('pressure')
        }
        # Remove None values
        advanced = {k: v for k, v in advanced.items() if v is not None}

        prediction = predict_next_day(recent_7_days, lights_today, predict_date, advanced)

        return jsonify({
            'success': True,
            'prediction_wh': prediction,
            'date': predict_date,
            'model': model_type
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    print("\n" + "="*70)
    print("HOUSEHOLD ENERGY FORECASTER STARTED")
    print("Open: http://127.0.0.1:5000")
    print("="*70)
    app.run(debug=True, port=5000)