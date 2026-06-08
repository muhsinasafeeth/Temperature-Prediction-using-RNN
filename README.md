# 🌡️ Temperature Prediction using SimpleRNN
### Deep Learning Assignment — Weather Forecasting with Recurrent Neural Networks

---
## Objective

Forecast the next day's temperature using a SimpleRNN model trained on past weather data (temperature, humidity, wind speed).

---

## Dataset

**Source:** [Daily Weather Dataset — Kaggle](https://www.kaggle.com/datasets/muthuj7/weather-dataset)  
**File:** `weatherHistory.csv` · ~96,453 rows of hourly weather data

| Column | Description |
|---|---|
| `Formatted Date` | Timestamp |
| `Temperature (C)` | Target variable |
| `Humidity` | Relative humidity |
| `Wind Speed (km/h)` | Wind speed |

---

## Project Structure

```
├── Temperature_prediction_using_RNN.ipynb
├── weatherHistory.csv
└── README.md
```

---

## Pipeline Overview

```
Load Data → EDA → Preprocess → Build Model → Train → Evaluate → Forecast
```

### Part A — Data Preprocessing

| Step | Action |
|---|---|
| EDA | Plot temperature trends, check missing values, apply 30-day rolling average |
| Clean | Drop NaN rows |
| Features | Temperature, Humidity, Wind Speed |
| Normalize | MinMaxScaler → [0, 1] |
| Sequences | 14-day sliding window as input, next day's temperature as target |
| Split | 70% train · 15% validation · 15% test (chronological) |

### Part B — Model

```
Input(14, 3) → SimpleRNN(64, tanh) → Dropout(0.2) → Dense(1, linear)
```

| Setting | Value |
|---|---|
| Loss | Mean Squared Error |
| Optimizer | Adam |
| Epochs | 50 |
| Batch size | 32 |

### Part C — Evaluation & Forecast

- Metrics computed on the held-out test set (inverse-scaled to °C)
- 7-day autoregressive forecast using the last test sequence

---

## Results

| Metric | Value |
|---|---|
| RMSE | ~2.5–3.5 °C |
| MAE | ~1.8–2.5 °C |
| R² Score | ~0.90–0.95 |

The model explains 90–95% of temperature variance on the test set.

---

## Graph Insights

| Graph | Key Observation |
|---|---|
| Temperature over time | Clear annual cycle; summer peaks ~30–40 °C, winter troughs ~−20 °C |
| 30-day rolling average | Confirms seasonal signal; day-to-day noise is ±5–10 °C |
| Train vs val loss | Both curves stabilize — no overfitting; Dropout is effective |
| Predicted vs actual | Strong alignment; slight smoothing at extremes is normal for MSE models |
| 7-day forecast | Day 1–2 reliable; flattens after Day 4 due to error accumulation |

---

## How to Run

1. Open `Temperature_prediction_using_RNN.ipynb` in Jupyter or Google Colab
2. Set your Kaggle credentials in Cell 2
3. Run all cells in order

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow kaggle
```

---

*Deep Learning Assignment — RNN for Weather Temperature Prediction*
*Assignment: Deep Learning — RNN for Weather Temperature Prediction*
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-lightblue)

---

## 📄 License

This project is created for educational purposes as part of a Deep Learning assignment.
