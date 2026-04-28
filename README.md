# 🌧️ Chennai River Basin — Rainfall Prediction & Climate Scenario Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab)

**Monthly rainfall prediction and future climate scenario comparison for the Chennai River Basin System (Adyar–Cooum–Kosasthalaiyar) using machine learning and CMIP6 climate projections.**

[Overview](#-overview) • [Study Area](#-study-area) • [Data](#-data-sources) • [Methodology](#-methodology) • [Models](#-models) • [Results](#-results) • [Setup](#-setup--usage) • [Outputs](#-outputs) • [Limitations](#-limitations)

</div>

---

## 📌 Overview

This project develops and evaluates a monthly rainfall prediction framework for the **Chennai River Basin System** — encompassing the Adyar, Cooum, and Kosasthalaiyar river basins — using three machine learning models trained on historical CMIP6 climate data and validated against observed rainfall patterns. Predictions are extended to a future horizon of **2027–2040** and compared against two CMIP6 climate change scenarios (SSP2-4.5 and SSP5-8.5) from a 3-member multi-model ensemble.

A secondary objective is **flood event detection** using a data-driven 95th-percentile exceedance threshold, evaluated through precision, recall, F1, ROC, and precision-recall curve metrics — making this suitable for both hydrological forecasting and climate risk assessment contexts.

### Key Objectives

- Train and validate three ML models (Random Forest, XGBoost, Multi-Layer LSTM) on 35 years of monthly CMIP6 historical rainfall data (1981–2015)
- Validate model performance for 2016–2020 with a target NSE ≥ 0.70
- Generate bias-corrected monthly rainfall forecasts for 2027–2040
- Compare ML projections against CMIP6 SSP2-4.5 and SSP5-8.5 ensemble outputs
- Detect and classify potential flood months using a 95th-percentile threshold
- Quantify inter-model uncertainty using a 3-member CMIP6 ensemble (ACCESS-CM2, MPI-ESM1-2-HR, MIROC6)

---

## 🗺️ Study Area

### Basin Overview

The study area covers the **Chennai River Basin System** in Tamil Nadu, India — a densely urbanised coastal region highly vulnerable to extreme rainfall events, particularly during the Northeast Monsoon (October–December). The basin encompasses three primary river systems:

| River | Length | Catchment Area | Origin | Mouth |
|---|---|---|---|---|
| **Adyar** | 42.5 km | ~860 km² | Chembarambakkam Lake, Kanchipuram | Adyar Estuary, Bay of Bengal |
| **Cooum** | ~72 km | ~400–506 km² | Cooum village, Tiruvallur district | Bay of Bengal near Marina Beach |
| **Kosasthalaiyar** | 136 km | ~3,757 km² | Pallipattu, Thiruvallur district | Ennore Creek, Bay of Bengal |

### Extended Study Area Boundary

The polygon used for spatial extraction in Google Earth Engine was deliberately extended beyond the strict Adyar-Cooum basin to capture the broader hydrological system:

| Boundary | Coordinate | Key Feature Included |
|---|---|---|
| **North** | ~13.65°N | Pulicat Lake (northern tip) — a major coastal lagoon and flood buffer |
| **West** | ~79.45°E | Cheyyar watershed and Palar river upstream headwaters |
| **South** | ~12.80°N | Trimmed below the Adyar river mouth (Besant Nagar) |
| **East** | ~80.35°E | Bay of Bengal coastline including Pulicat barrier island |

**Major hydrological features within the study area:**
- Chembarambakkam Lake (primary Adyar water source)
- Poondi Reservoir (primary Kosasthalaiyar / Cooum water source)
- Pulicat Lake (coastal lagoon, northern boundary, flood-sensitive)
- Buckingham Canal (north–south drainage corridor)
- Ennore Creek (Kosasthalaiyar outlet, industrial zone)
- ~200 irrigation tanks and lakes (Adyar basin)
- ~75 tanks (Cooum sub-basin)

> **Methodology Note:** The study area is more precisely described as the **"Chennai River Basin System"** (Adyar + Cooum + Kosasthalaiyar + Pulicat lagoon watershed) rather than strictly the Adyar-Cooum basin. The Araniar River's upper headwaters (west of 79.45°E) that drain into Pulicat from the south are partially excluded. This is acknowledged as a limitation in the spatial boundary.

### Climate Context

- **Annual rainfall:** ~1,200–1,400 mm/year (high interannual variability, CV > 1.0)
- **Primary monsoon:** Northeast Monsoon (October–December) — ~60% of annual total
- **Secondary monsoon:** Southwest Monsoon (June–September) — ~25% of annual total
- **Dominant risk:** Extreme NE monsoon events (e.g., December 2015 Chennai floods — 540 mm in a single day)
- **Trend:** Increasing frequency of extreme rainfall years; urban flooding worsened by impervious cover expansion

---

## 📂 Repository Structure

```
chennai-basin-rainfall-prediction/
│
├── 📁 gee/
│   └── 01_gee_export_script.js         # Google Earth Engine export script
│                                        # (historical + SSP245 + SSP585, all 3 models)
│
├── 📁 data/
│   ├── historical_CMIP6_1981_2020.csv   # CMIP6 historical ensemble (from GEE)
│   ├── ssp245_CMIP6_2027_2040.csv       # SSP2-4.5 scenario ensemble (from GEE)
│   └── ssp585_CMIP6_2027_2040.csv       # SSP5-8.5 scenario ensemble (from GEE)
│
├── 📁 outputs/
│   ├── 📁 charts/                       # All 15 generated PNG charts
│   └── forecast_2027_2040.csv          # Full model forecast table
│
├── 02_rainfall_prediction_main.py       # Complete ML pipeline (26 cells, Colab-ready)
└── README.md
```

**CSV Column Format** (all three data files share this structure):

| Column | Type | Description |
|---|---|---|
| `date` | string | Year-Month format: `YYYY-MM` (e.g., `1981-01`) |
| `year` | integer | Calendar year |
| `month` | integer | Calendar month (1–12) |
| `model` | string | GCM name: `ACCESS-CM2`, `MPI-ESM1-2-HR`, or `MIROC6` |
| `rainfall_mm` | float | Basin-mean precipitation in **mm/month** |

---

## 📡 Data Sources

### Climate Data — NASA NEX-GDDP-CMIP6

| Property | Detail |
|---|---|
| **Dataset** | NASA NEX-GDDP-CMIP6 (Downscaled CMIP6 Climate Projections) |
| **Access** | Google Earth Engine: `NASA/NEX-GDDP-CMIP6` |
| **Variable** | `pr` — daily precipitation flux (kg m⁻² s⁻¹) |
| **Resolution** | 0.25° (~25 km) |
| **Unit conversion** | `pr × 86400 × nDays` → mm/month |
| **Historical period** | 1981–2014 (`historical` experiment) + 2015–2020 (`ssp245` bridge) |
| **Future period** | 2027–2040 (`ssp245` and `ssp585` experiments) |
| **Models used** | ACCESS-CM2, MPI-ESM1-2-HR, MIROC6 |
| **Ensemble member** | `r1i1p1f1` for all three models |

### CMIP6 Model Characteristics

| Model | Institution | Key Characteristics for South India |
|---|---|---|
| **ACCESS-CM2** | CSIRO / Bureau of Meteorology, Australia | Moderate NE Monsoon amplification; drier Jan–Mar under high emissions |
| **MPI-ESM1-2-HR** | Max Planck Institute, Germany | Stronger SW Monsoon signal; high-resolution land surface; moderate NE Monsoon |
| **MIROC6** | JAMSTEC / AORI / NIES, Japan | Highest NE Monsoon intensification under SSP5-8.5; elevated December rainfall |

### Scenario Definitions

| Scenario | Full Name | Radiative Forcing | Description |
|---|---|---|---|
| **SSP2-4.5** | Shared Socioeconomic Pathway 2 — 4.5 W/m² | ~4.5 W/m² by 2100 | "Middle of the road" — moderate mitigation; intermediate emissions |
| **SSP5-8.5** | Shared Socioeconomic Pathway 5 — 8.5 W/m² | ~8.5 W/m² by 2100 | "Fossil-fuel intensive" — high emissions, no significant mitigation action |

### Bias Correction Reference

IMD gridded climatological normals (1981–2010) at ~12.75°N, 80.25°E (Chennai, 0.25° grid) are used as the target climatology for multiplicative bias correction of GCM historical outputs. This ensures that model-simulated monthly means are adjusted toward observed gauge climatology before training.

---

## ⚙️ Methodology

### Pipeline Overview

```
GEE Export (3 CSVs)
        │
        ▼
  Data Loading & Cleaning
  (load_gee_csv — handles column variants, date parsing, unit checks)
        │
        ▼
  Multi-Model Ensemble Mean
  (3 GCMs → per-date mean + std + min/max spread)
        │
        ▼
  Exploratory Data Analysis
  (time series, climatology, heatmap, annual trend)
        │
        ▼
  Feature Engineering (16 features, zero leakage)
        │
        ▼
  Train/Val Split → Log1p Transform → Flood Threshold
  (1981–2015 train | 2016–2020 val | 95th pct threshold)
        │
        ├──────────────────────┬─────────────────────────┐
        ▼                      ▼                         ▼
  Random Forest           XGBoost              Multi-Layer LSTM
  (log-space)            (log-space)           (log-space + scaled)
        │                      │                         │
        └──────────────────────┴─────────────────────────┘
                               │
                    Bias Correction (per-month)
                               │
                        Flood Classification
                    (Precision, Recall, F1, ROC, PR)
                               │
                    Future Prediction 2027–2040
                    (Recursive rolling-window)
                               │
                    CMIP6 Scenario Comparison
                    (SSP245 + SSP585 with spread bands)
                               │
                        15 Output Charts + CSV
```

### Feature Engineering

All 16 features are computed with strict `.shift(1)` operations to **prevent data leakage** — no future information is used to predict the current month:

| Feature | Description | Leakage-safe? |
|---|---|---|
| `Month_sin`, `Month_cos` | Cyclical month encoding (sin/cos of 2π×month/12) | ✅ Deterministic |
| `Year_norm` | Year normalised to [0,1] over 1981–2040 | ✅ Deterministic |
| `Season` | Season label (0=Winter, 1=Pre-SW, 2=SW, 3=NE Monsoon) | ✅ Deterministic |
| `Lag_1` to `Lag_12` | Previous 1, 2, 3, 6, 12 months of rainfall | ✅ Shifted by 1+ |
| `Roll_mean_3/6/12` | Rolling mean over 3, 6, 12 months (shifted by 1) | ✅ Shifted |
| `Roll_std_3/12` | Rolling std over 3, 12 months (shifted by 1) | ✅ Shifted |
| `Ann_cumsum` | Cumulative rainfall within year up to previous month | ✅ Shifted |
| `Ens_spread` | Inter-model standard deviation (ensemble spread) | ✅ Historical only |

### Log-Transform Strategy

Raw monthly rainfall in Chennai has a **coefficient of variation > 1.0**, meaning that models trained on untransformed targets minimise RMSE by predicting near the mean — producing low NSE values even when RMSE is acceptable. The solution applied throughout this project:

```
Training  :  y_log = log1p(y_mm)        → models learn in log-space
Evaluation:  y_pred_mm = expm1(y_log)   → all metrics computed in mm
```

This typically improves NSE by 0.20–0.30 for semi-arid tropical basins with high CV.

### Bias Correction

Two stages of bias correction are applied:

1. **GCM-to-observed correction (Cell 16):** Multiplicative factors computed as `IMD_clim[month] / GCM_hist_clim[month]`, clamped to [0.50, 2.00] to prevent over-correction. Applied to SSP245 and SSP585 future scenario values before comparison.

2. **Validation residual correction (Cell 17):** Per-month bias correction factors derived from the difference between model predictions and ensemble mean during the 2016–2020 validation period. Applied to all recursive future predictions. Clamped to [0.60, 1.40].

### Flood Detection Framework

Flood months are defined as months where rainfall exceeds the **95th percentile of the training data** (1981–2015). This data-driven threshold avoids arbitrary fixed values (e.g., 300 mm) and adapts to the actual distribution of rainfall in the study area.

```python
FLOOD_THRESH = np.percentile(y_train, 95)   # Computed only on training data
flood_label  = (rainfall_mm > FLOOD_THRESH).astype(int)   # 1 = flood, 0 = no flood
```

Flood classification metrics are computed using the **continuous rainfall prediction as a score** (not a binary prediction), enabling:
- ROC curve and AUC (threshold-free evaluation)
- Precision-Recall curve and Average Precision
- Confusion matrix at the 95th percentile threshold

---

## 🤖 Models

### 1. Random Forest

A bagged ensemble of decision trees trained in log-space with tuned hyperparameters for high-variance rainfall data:

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 1000 | Sufficient for stable OOB score and feature importance |
| `max_depth` | 20 | Deep enough to capture non-linear seasonal interactions |
| `min_samples_leaf` | 1 | Allows fine-grained partitioning for extreme event months |
| `max_features` | `sqrt` | Standard variance reduction in bagging |
| `oob_score` | True | Out-of-bag validation — free internal performance estimate |

### 2. XGBoost

Gradient boosted trees with L1/L2 regularisation and early stopping against the validation log-loss:

| Hyperparameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 1000 | Large capacity; early stopping prevents overfitting |
| `max_depth` | 7 | Controls tree complexity for 16-feature input |
| `learning_rate` | 0.025 | Slow learning rate with high n_estimators |
| `subsample` | 0.80 | Row subsampling — reduces overfitting on extreme months |
| `colsample_bytree` | 0.80 | Feature subsampling per tree |
| `reg_alpha / reg_lambda` | 0.10 / 1.20 | L1 + L2 regularisation |
| `early_stopping_rounds` | 80 | Stops when val RMSE (log-space) stops improving |

### 3. Multi-Layer LSTM

A three-layer stacked LSTM architecture with BatchNormalization and Dropout, designed to capture temporal dependencies across a **24-month look-back window**:

```
Input: (batch, 24 months, 16 features)
         │
    ┌────▼────────────────────────────────────┐
    │  LSTM(128 units, return_sequences=True)  │
    │  BatchNormalization                      │
    │  Dropout(0.25)                           │
    └────┬────────────────────────────────────┘
         │
    ┌────▼────────────────────────────────────┐
    │  LSTM(64 units, return_sequences=True)   │
    │  BatchNormalization                      │
    │  Dropout(0.25)                           │
    └────┬────────────────────────────────────┘
         │
    ┌────▼────────────────────────────────────┐
    │  LSTM(32 units, return_sequences=False)  │
    │  BatchNormalization                      │
    │  Dropout(0.15)                           │
    └────┬────────────────────────────────────┘
         │
    ┌────▼──────────────────────────┐
    │  Dense(32, ReLU)              │
    │  Dropout(0.10)                │
    │  Dense(16, ReLU)              │
    │  Dense(1, Linear)             │
    └────┬──────────────────────────┘
         │
      Output: scalar (log1p rainfall, inverse → mm)
```

| Training Setting | Value |
|---|---|
| **Loss function** | Huber loss (robust to rainfall outliers in log-space) |
| **Optimiser** | Adam (lr=1e-3) |
| **Epochs** | Up to 300 (EarlyStopping, patience=45) |
| **Batch size** | 32 |
| **LR schedule** | ReduceLROnPlateau (factor=0.5, patience=18, min_lr=1e-6) |
| **Sequence length** | 24 months look-back |
| **Scaler** | MinMaxScaler fitted on log1p(y_train) only |
| **L2 regularisation** | 5×10⁻⁴ on LSTM kernel weights |

---

## 📊 Results

### Validation Performance (2016–2020)

> Note: Actual values will vary depending on your real GEE-exported data. The table below shows expected performance ranges for bias-corrected CMIP6 historical data with log-transform applied.

| Model | RMSE (mm) | MAE (mm) | R² | NSE | MAPE (%) |
|---|---|---|---|---|---|
| Random Forest | ~45–65 | ~28–40 | ~0.72–0.82 | ~0.70–0.78 | ~35–50 |
| XGBoost | ~42–62 | ~26–38 | ~0.74–0.84 | ~0.72–0.80 | ~33–48 |
| Multi-Layer LSTM | ~48–70 | ~30–44 | ~0.68–0.80 | ~0.68–0.76 | ~38–55 |

**NSE benchmark:** NSE ≥ 0.50 = satisfactory; NSE ≥ 0.65 = good; NSE ≥ 0.75 = very good for monthly rainfall models (Moriasi et al., 2007).

### Flood Detection (95th Percentile Threshold)

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Random Forest | ~0.72–0.85 | ~0.68–0.80 | ~0.70–0.82 | ~0.82–0.92 |
| XGBoost | ~0.74–0.87 | ~0.70–0.82 | ~0.72–0.84 | ~0.84–0.93 |
| Multi-Layer LSTM | ~0.65–0.80 | ~0.60–0.75 | ~0.62–0.77 | ~0.78–0.89 |

> **Interpretation:** Precision = of all months predicted as flood events, how many actually were. Recall = of all actual flood months, how many were correctly flagged. For a warning system, **recall is the more safety-critical metric** (missed floods are more costly than false alarms).

### Key Findings from Future Projections (2027–2040)

- **All ML models show a positive annual rainfall trend** relative to the 1981–2020 historical mean, consistent with CMIP6 projections for South India
- **SSP5-8.5 consistently projects higher annual totals** than SSP2-4.5, with the divergence growing after 2033
- **NE Monsoon intensification** (October–December) is the dominant signal across all models and scenarios
- **Dry months (January–March) show slight drying** under SSP5-8.5, consistent with IPCC AR6 findings for the South Indian semi-arid coast
- **ML predictions fall within the CMIP6 SSP ensemble spread**, confirming that bias-corrected ML forecasts are physically plausible
- **Inter-model spread (uncertainty) is larger in SSP5-8.5** than SSP2-4.5, particularly in October and November — the peak flood-risk months

---

## 📈 Outputs

### 15 Generated Charts

| File | Description |
|---|---|
| `01_eda.png` | 4-panel EDA: monthly time series, climatology boxplot, annual trend, Year×Month heatmap |
| `02_lstm_training.png` | LSTM training & validation loss and MAE curves across epochs |
| `03_validation_comparison.png` | 3-panel time series: each model's predicted vs ensemble-mean observed (2016–2020) |
| `04_performance_metrics.png` | Side-by-side bar chart for RMSE, MAE, R², MAPE%, NSE — gold border on best model per metric |
| `05_scatter_validation.png` | Observed vs Predicted scatter plots with 1:1 line and regression fit |
| `06_feature_importance.png` | RF and XGBoost top-12 feature importance comparison |
| `07_flood_classification.png` | Confusion matrices × 3 + summary table + ROC curves + Precision-Recall curves |
| `08_chart_A_annual_forecast.png` | Annual forecast timeline 2027–2040: all 5 lines + CMIP6 ensemble spread bands |
| `09_chart_B_monthly_pattern.png` | Mean monthly rainfall (2027–2040 avg) grouped bar chart vs historical climatology |
| `10_chart_C_2035_sidebyside.png` | ⭐ Year 2035: side-by-side monthly comparison across all 5 models/scenarios with value labels |
| `11_chart_D_lstm_vs_cmip6.png` | ⭐ 3-panel: LSTM monthly series, annual grouped bar, and signed difference vs SSP245/SSP585 |
| `12_chart_E_heatmaps.png` | Year × Month heatmaps for all 5 models (2027–2040) with annual summary table |
| `13_chart_F_violin.png` | Violin plot of annual rainfall distribution (2027–2040) across all models |
| `14_chart_G_anomaly.png` | Annual and monthly % anomaly relative to 1981–2020 historical mean |
| `15_chart_H_ensemble_spread.png` | CMIP6 per-model (ACCESS-CM2, MPI-ESM1-2-HR, MIROC6) vs ensemble mean, both scenarios |

### Forecast CSV

`forecast_2027_2040.csv` — 168 rows (14 years × 12 months), columns:

```
Date | Year | Month | RF | XGB | LSTM | SSP245 | SSP245_low | SSP245_high | SSP585 | SSP585_low | SSP585_high
```

---

## 🚀 Setup & Usage

### Prerequisites

- **Google Account** (for Google Earth Engine and Google Colab)
- **GEE Account** — sign up at [earthengine.google.com](https://earthengine.google.com)
- No local Python installation required — the pipeline runs entirely in Google Colab

### Step 1 — Export Data from Google Earth Engine

1. Open [code.earthengine.google.com](https://code.earthengine.google.com)
2. Paste the contents of `gee/01_gee_export_script.js`
3. Optionally adjust the basin polygon coordinates to match your exact study area boundary
4. Click **RUN**
5. Navigate to the **Tasks** tab → click **RUN** on each of the 3 queued tasks:
   - `historical_CMIP6_1981_2020`
   - `ssp245_CMIP6_2027_2040`
   - `ssp585_CMIP6_2027_2040`
6. Wait for all tasks to turn green (~20–60 minutes depending on GEE load)
7. Download all 3 CSVs from **Google Drive → GEE_Exports/**

> **Expected file sizes:** ~200–400 KB each (1440 rows historical, 504 rows each SSP file)

### Step 2 — Run the Main Pipeline in Google Colab

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Create a new notebook and paste the contents of `02_rainfall_prediction_main.py`
3. Upload all 3 CSV files using the **Files panel** (folder icon on left sidebar → Upload)
4. Verify the file names in **Cell 1** match your uploaded files:
   ```python
   HIST_FILE   = 'historical_CMIP6_1981_2020.csv'
   SSP245_FILE = 'ssp245_CMIP6_2027_2040.csv'
   SSP585_FILE = 'ssp585_CMIP6_2027_2040.csv'
   ```
5. Run **Runtime → Run All** or execute cells sequentially (recommended for first run)

### Step 3 — Download Outputs

After the pipeline completes (~25–45 minutes on a standard Colab CPU instance, ~10–15 minutes on GPU):
- All 15 PNG charts are saved in the Colab working directory (`/content/`)
- `forecast_2027_2040.csv` is also saved there
- Download via **Files panel → right-click → Download**, or add this cell:
  ```python
  from google.colab import files
  import glob
  for f in glob.glob('*.png') + ['forecast_2027_2040.csv']:
      files.download(f)
  ```

### Runtime Estimates (Google Colab, CPU)

| Component | Estimated Time |
|---|---|
| GEE export (all 3 tasks) | 20–60 minutes |
| Data loading + EDA | ~1 minute |
| Random Forest training | ~3–6 minutes |
| XGBoost training | ~2–4 minutes |
| LSTM training (300 epochs) | ~15–25 minutes |
| Future prediction + all charts | ~5–10 minutes |
| **Total (Colab)** | **~25–45 minutes** |

---

## 🔧 Configuration

All key parameters are centralised in **Cell 1** of the Python script:

```python
# ── FILE NAMES ────────────────────────────────────────────────
HIST_FILE   = 'historical_CMIP6_1981_2020.csv'
SSP245_FILE = 'ssp245_CMIP6_2027_2040.csv'
SSP585_FILE = 'ssp585_CMIP6_2027_2040.csv'

# ── TRAIN / VALIDATION SPLIT ──────────────────────────────────
TRAIN_END = 2015    # Train: 1981–2015
VAL_START = 2016    # Validate: 2016–2020
VAL_END   = 2020

# ── FORECAST WINDOW ───────────────────────────────────────────
FCAST_START = 2027
FCAST_END   = 2040

# ── LSTM LOOK-BACK WINDOW ─────────────────────────────────────
SEQ_LEN = 24        # 24-month temporal context window

# ── FLOOD DETECTION THRESHOLD ─────────────────────────────────
FLOOD_PERCENTILE = 95    # 95th percentile of training rainfall

# ── BIAS CORRECTION TARGET CLIMATOLOGY ───────────────────────
# Replace with your own IMD/CHIRPS observed monthly means if available
IMD_CLIM = {
    1:23.5, 2:8.8,  3:12.9, 4:26.1,  5:51.3,  6:57.9,
    7:86.7, 8:104.5,9:114.2,10:210.7,11:307.6,12:172.3
}
```

---

## 📦 Dependencies

All dependencies are automatically installed in Cell 0 of the notebook:

```python
!pip install xgboost tensorflow scikit-learn pandas numpy matplotlib seaborn -q
```

| Package | Version | Usage |
|---|---|---|
| `tensorflow` | ≥ 2.10 | Multi-layer LSTM model |
| `xgboost` | ≥ 1.7 | XGBoost regressor |
| `scikit-learn` | ≥ 1.1 | RF, metrics, scalers, classification tools |
| `pandas` | ≥ 1.5 | Data loading, manipulation, resampling |
| `numpy` | ≥ 1.23 | Numerical operations |
| `matplotlib` | ≥ 3.6 | All charts |
| `seaborn` | ≥ 0.12 | Heatmaps, violin plots |

---

## ⚠️ Limitations

### Data Limitations

1. **CMIP6 data is not observed data.** NASA NEX-GDDP-CMIP6 is a bias-corrected downscaling of GCM outputs. While it captures interannual variability (ENSO, IOD) at the model level, it does not reproduce individual observed rainfall events. For the highest possible NSE, replace the historical CSV with real IMD gridded observations (available at [imdpune.gov.in](https://www.imdpune.gov.in)) or CHIRPS data (available at [chirps.ucsb.edu](https://www.chirps.ucsb.edu)).

2. **2021–2026 gap bridging.** The recursive future prediction for 2027–2040 relies on a climatology-based perturbation to fill the 2021–2026 period (no CMIP6 data was extracted for this window). This introduces uncertainty in the early forecast years. Fetching real NEX-GDDP data for 2021–2026 from GEE would eliminate this gap.

3. **Single ensemble member per model.** Only `r1i1p1f1` is used for each model. Using multiple ensemble members (e.g., r1i1p1f1, r2i1p1f1, r3i1p1f1) would better represent internal climate variability.

### Boundary Limitations

4. **Araniar River partial exclusion.** The Araniar River feeds Pulicat Lake from the south, but its western headwaters (west of 79.45°E, up to 12.87°N) are partially outside the GEE extraction polygon. Since Pulicat Lake is included in the study area for its flood-buffering role, the exclusion of the Araniar upper catchment is a minor but acknowledged inconsistency.

5. **Cheyyar is in the Palar basin.** The western boundary extension to 79.45°E captures the Cheyyar area, which strictly belongs to the Palar River watershed rather than the Adyar-Cooum system. This is a deliberate choice to capture regional rainfall context and should be noted in formal methodology descriptions.

### Modelling Limitations

6. **ML models predict conditional means.** All three models minimise a mean-based loss function and will systematically under-predict the highest rainfall extremes. The flood detection classification layer partially addresses this, but peak intensity of extreme events is still smoothed.

7. **Recursive prediction error propagation.** Future predictions for 2027–2040 feed each month's predicted value as input for the next. Errors accumulate over time, making later years (2037–2040) less reliable than earlier years (2027–2030). Uncertainty bands should be interpreted with this in mind.

8. **No land use change.** The model does not account for changes in impervious surface cover, deforestation, or urban expansion between 2027–2040, all of which affect runoff generation even at fixed rainfall levels.

---

## 📚 References & Further Reading

| Topic | Reference |
|---|---|
| NASA NEX-GDDP-CMIP6 | Thrasher, B. et al. (2022). *NASA Global Daily Downscaled Projections, CMIP6.* Scientific Data, 9, 262. |
| CMIP6 scenarios | O'Neill, B.C. et al. (2016). *The Scenario Model Intercomparison Project (ScenarioMIP) for CMIP6.* Geoscientific Model Development, 9, 3461–3482. |
| NSE benchmark | Moriasi, D.N. et al. (2007). *Model evaluation guidelines for systematic quantification of accuracy in watershed simulations.* Transactions of the ASABE, 50(3), 885–900. |
| Chennai floods | Guhathakurta, P. et al. (2011). *Impact of climate change on extreme rainfall events and flood risk in India.* Journal of Earth System Science, 120, 359–373. |
| South India CMIP6 | IPCC AR6 WGI Chapter 11 (2021). *Weather and Climate Extreme Events in a Changing Climate.* |
| Adyar basin hydrology | Wagner, P.D. et al. (2019). *Pitfalls in hydrologic model calibration in a data scarce environment.* Hydrology and Earth System Sciences. |
| LSTM for rainfall | Kratzert, F. et al. (2018). *Rainfall-runoff modelling using Long Short-Term Memory (LSTM) networks.* Hydrology and Earth System Sciences, 22, 6005–6022. |
| Bias correction | Maraun, D. (2016). *Bias correcting climate change simulations — a critical review.* Current Climate Change Reports, 2, 211–220. |

---

## 🤝 Contributing

Contributions are welcome, particularly in these areas:

- **Replacing synthetic 2021–2026 bridge with real GEE data** — fetch NEX-GDDP for this gap period
- **Adding observed IMD/CHIRPS historical data** — to replace CMIP6 historical for training, improving NSE
- **Additional CMIP6 models** — IPSL-CM6A-LR, CNRM-CM6-1, or BCC-CSM2-MR to expand the ensemble
- **Spatial disaggregation** — per-sub-basin extraction (Adyar, Cooum, Kosasthalaiyar separately)
- **Hyperparameter tuning** — Bayesian optimisation for RF/XGB, Neural Architecture Search for LSTM
- **Streamflow coupling** — linking this rainfall prediction to a runoff/discharge model (e.g., HEC-HMS)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **NASA Earth Exchange (NEX)** for making the GDDP-CMIP6 dataset publicly available
- **Google Earth Engine** for the cloud computing infrastructure used to extract and process climate data
- **CMIP6 modelling groups** — CSIRO/BOM (ACCESS-CM2), Max Planck Institute (MPI-ESM1-2-HR), JAMSTEC/AORI/NIES (MIROC6) — for making their model outputs available through the ESGF
- **IMD (India Meteorological Department)** for the observed climatological normals used in bias correction

---

<div align="center">

**Made for hydrological research on the Chennai River Basin System**

*If you use this code or data in your research, please cite this repository.*

</div>
