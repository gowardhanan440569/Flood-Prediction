# Flood Early Warning System for Tamil Nadu (CMIP6 Historical Analysis)

## Problem Statement
Predicting flood events in the Tamil Nadu region is a significant challenge due to the extreme rarity of these events. In the historical CMIP6 dataset, flood days (defined by the 95th percentile of rainfall) represent only ~5% of the data. Standard machine learning models often struggle with this 'class imbalance,' frequently ignoring the rare flood class to achieve high overall accuracy, which is dangerous for early warning systems.

## Methodology
1. **Data Preparation**: Loading historical rainfall data, performing unit conversion (kg m-2 s-1 to mm/day), and cleaning missing values.
2. **Feature Engineering**:
    * **Lag Features**: 1, 7, 14, and 30-day rainfall lags to capture temporal memory.
    * **Advanced Indices**: Standardized Precipitation Index (SPI) for drought/wetness context and Rx5day (5-day rolling sum) to capture cumulative saturation.
3. **Sequential Preparation**: Reshaping data into 30-day temporal windows to feed the sequential model.

## Why LSTM?
Long Short-Term Memory (LSTM) networks were chosen because they excel at processing sequential data and capturing long-term dependencies in weather patterns.

### Strategic Trade-off: Recall over Precision
In a life-saving early warning system, the cost of a 'False Negative' (missing a flood) is much higher than a 'False Positive' (a false alarm). Our LSTM was optimized using class weighting to prioritize **Recall**. While this leads to lower Precision (more false alarms), it ensures that the model successfully detects ~92% of actual flood events, providing a critical window for emergency response.

## Visualizations Overview
* **Model Performance Comparison**: A bar chart comparing Accuracy, Precision, Recall, and F1-Score across RF, XGBoost, and Optimized LSTM, highlighting the superior sensitivity of the LSTM.
* **Derived Unit Hydrograph**: A hydrological profile representing the region's typical runoff response based on the top 5 historical extreme rainfall events.
 
