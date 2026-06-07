# 🌤️ Weather Temperature Prediction using SimpleRNN

A deep learning project that uses a **Recurrent Neural Network (SimpleRNN)** built with TensorFlow/Keras to forecast the **next day's temperature** based on historical weather data.

---

## 📁 Project Structure

```
rnn_weather_prediction/
│
├── RNN_Weather_Prediction.ipynb   # Main Jupyter Notebook (all tasks)
├── requirements.txt               # Python dependencies
├── README.md                      # This file
│
└── (generated after running notebook)
    ├── temperature_trend.png      # Task 1 — temperature time series plot
    ├── missing_values.png         # Task 1 — missing value visualisation
    ├── training_curves.png        # Task 5 — loss & MAE curves
    ├── evaluation_plots.png       # Task 6 — predicted vs actual + scatter
    ├── 7day_forecast.png          # Task 7 — 7-day forecast chart
    └── best_rnn_model.keras       # Saved best model weights
```

---

## 📋 Assignment Structure

| Part | Tasks | Description |
|------|-------|-------------|
| **A** | 1–2 | Data Understanding & Preprocessing |
| **B** | 3–5 | RNN Model Development & Training |
| **C** | 6–7 | Model Evaluation & 7-Day Forecasting |

---

## 🗃️ Dataset

**Daily Climate Time Series Data** — Delhi, India  
Source: [Kaggle — Daily Climate Time Series Data](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data)

| Column | Description |
|--------|-------------|
| `date` | Observation date |
| `meantemp` | Mean daily temperature (°C) — **target variable** |
| `humidity` | Mean daily humidity (%) |
| `wind_speed` | Mean daily wind speed (km/h) |
| `meanpressure` | Mean daily pressure (hPa) |

> 📥 Download `DailyDelhiClimateTrain.csv` and place it in the project root before running the notebook.

---

## 🏗️ Model Architecture

```
Input (seq_len=10, features=4)
        ↓
SimpleRNN (64 units, tanh activation)
        ↓
Dropout (0.2)
        ↓
Dense (32 units, ReLU)
        ↓
Dense (1 unit) — regression output
```

| Component | Configuration |
|-----------|---------------|
| Loss | Mean Squared Error (MSE) |
| Optimizer | Adam (lr = 1e-3) |
| Metric | Mean Absolute Error (MAE) |
| Batch Size | 32 |
| Max Epochs | 100 (EarlyStopping with patience=10) |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/rnn_weather_prediction.git
cd rnn_weather_prediction
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `DailyDelhiClimateTrain.csv` from [Kaggle](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data) and place it in the project root.

### 5. Run the notebook

```bash
jupyter notebook RNN_Weather_Prediction.ipynb
```

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error — error in °C |
| **MAE** | Mean Absolute Error — average error in °C |
| **R²** | Coefficient of determination — proportion of variance explained |

---

## 🔬 Key Implementation Details

- **MinMaxScaler** fitted *only* on training data to prevent data leakage
- **Sliding window sequences** of 10 days → predict next day
- **Iterative 7-day forecasting** — each prediction feeds into the next step
- **EarlyStopping** + **ModelCheckpoint** callbacks for robust training
- All predictions inverse-transformed back to °C for interpretability

---

## 📈 Sample Results

After training on the Delhi climate dataset, the model achieves:
- RMSE ≈ 1.5–2.5 °C
- MAE ≈ 1.0–2.0 °C  
- R² ≈ 0.90–0.97

*(Exact values depend on random seed and dataset splits)*

---

## 🔧 Possible Improvements

- Replace `SimpleRNN` with `LSTM` or `GRU` for better long-range memory
- Use multi-step direct forecasting instead of autoregressive iteration
- Add hyperparameter tuning with Keras Tuner
- Incorporate additional features (cloud cover, dew point)

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange)
![Keras](https://img.shields.io/badge/Keras-3.x-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-lightblue)

---

## 📄 License

This project is created for educational purposes as part of a Deep Learning assignment.
