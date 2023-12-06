# Chile Air Quality Predictions

## Overview

This repository contains a series of Jupyter notebooks focused on predicting air quality (PM2.5) levels for areas near air monitoring stations in the Area Metropolitana de Santiago de Chile.

## Getting Started

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/mayo-nesso/chile-air-quality-predictions.git
    cd chile-air-quality-predictions
    ```

2. Install dependencies using Poetry:

    ```bash
    poetry install
    ```

    This will create a virtual environment and install the required dependencies.

## Notebooks

### 1. Download and Explore Data

- File: [1_download_and_explore_data.ipynb](chile_air_quality_predictions/1_download_and_explore_data.ipynb)
- Description: This notebook downloads air quality measurements (PM2.5 and PM10) for the Area Metropolitana de Santiago de Chile. It performs exploratory data analysis to prepare the data for further processing.

### 2. Prepare Data

- File: [2_prepare_data.ipynb](chile_air_quality_predictions/2_prepare_data.ipynb)
- Description: In this notebook, the data is prepared for predictive modeling. It includes merging data, filling missing values, and exploring methods to handle data gaps.

### 3. Predict

- File: [3_predict.ipynb](chile_air_quality_predictions/3_predict.ipynb)
- Description: The final notebook focuses on predicting PM2.5 levels for areas close to air monitoring stations. It involves choosing and tuning a predictive model, visualizing results, and exploring potential improvements.

## Future Improvements

The project suggests several areas for future enhancement:

1. **Better GeoJSON Polygon:** Optimize the size of the current GeoJSON polygon data.
2. **Utilize Other Pollutants:** Explore incorporating additional pollutants if more data becomes available.
3. **Time Series Prediction:** Extend the analysis to predict air quality for future time periods.
4. **Prepare for Productionization:** Consider developing a more complete and scalable process for real-world deployment.

## License

This project is licensed under the [MIT License](LICENSE.md).
