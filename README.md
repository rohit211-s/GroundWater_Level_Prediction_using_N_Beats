# GroundWater_Level_Prediction_using_Machine_Learning_Tools

## Abstract
Forecasting groundwater availability is essential for the future sustainable management of water resources. However, predicting groundwater levels is complicated, given the involvement of var-ious interacting elements such as precipitation, topography, land use, and groundwater extraction. Hydrological process-based prediction methods using numerical models for assessing aquifer conditions can be slow and uncertain. Thus, recent years have seen an increase in application of artificial intelligence (AI) to understand and anticipate changes in the water table. In this research, to compare the machine learning (ML) and deep learning (DL) methods, one of each was selected: the gradient boosting model (XGBoost) and a deep learning model (CNN-LSTM) driven with a 21-year dataset of daily rainfall and groundwater level data at a subtropical ranchland in Florida. Both models performed well during the training and testing phases. The training phase reported Root Mean Square Error (RMSE) values of 0.03m for XGBoost and 0.06m for CNN-LSTM. For the testing phase, the RMSE values were 0.067m for XGBoost and 0.064m for CNN-LSTM. The study found that the XGBoost model delivered the best results. This study contributes to developing AI methods that can be applied to forecasting groundwater levels in areas with limited groundwater data. However, the shallow groundwater level and high fluctuations in the study area could im-pact the modeling results.

## Explanation
Training N-beats model on GroundWater Level data and comparing it with other deep learning models like LSTM, Conv1d Layer and Sequential Deep Learning model. 

Groundwater Level Prediction Readme

1. Introduction:
This project focuses on predicting groundwater levels using two different approaches: univariate data analysis and multivariate data analysis. The objective is to forecast the groundwater level for the current day based on historical data, and additionally, using various environmental features that may influence groundwater dynamics.

2. Univariate Data Analysis:
In the first part of the project, we utilized univariate data, considering only the previous day's well value to predict the well value for today. Three different models were trained for this task:

- N-Beats: A forecasting model designed for time series data that learns to predict future values based on past observations.
- LSTM (Long Short-Term Memory): A type of recurrent neural network known for capturing long-term dependencies in sequential data, making it suitable for time series prediction tasks.
- Conv1d Layer: Utilizing one-dimensional convolutions to capture local patterns and features in the time series data.

3. Multivariate Data Analysis:
In the second part of the project, we extended our approach to incorporate multivariate data. Along with the previous day's well value, we included additional environmental features that could influence groundwater levels:

- Rain: Represents the amount of precipitation in the area, contributing to groundwater recharge.
- Maxt: The maximum temperature, influencing groundwater levels indirectly through evaporation rates and plant transpiration.
- Mint: The minimum temperature, affecting evaporation rates and snow cover persistence, thus impacting groundwater recharge.
- Meant: The mean temperature, similarly impacting groundwater levels to the maximum temperature.
- Vapor: Atmospheric vapor pressure or specific humidity, influencing potential evapotranspiration and groundwater levels.

By using multivariate time series data, we aimed to capture the complex relationships between these features and their combined impact on groundwater levels, leading to more accurate predictions.

4. Conclusion:
This project explores two different approaches for groundwater level prediction, using both univariate and multivariate data analyses. By training N-Beats, LSTM, and Conv1d Layer models on univariate data and incorporating additional environmental features for multivariate analysis, we aim to develop accurate and robust predictive models. The outcome of this project has significant implications for water resource management, environmental preservation, and sustainable planning.

Results:

Univariate Data(RMSE):

Deep Learning Sequential Model: 0.25655594 ||  LSTM Model: 0.2584401 ||  Conv1d Layer: 0.25639158 ||  N-Beats: 0.25508267

Multivariate Data(RMSE):

Deep Learning Sequential Model: 0.24326369  || Conv1d Layer: 0.24235931  ||   LSTM Model: 0.23851517 ||  N-Beats: 0.23459317

6. Usage:
- The code for data preprocessing, model training, and evaluation can be found in the respective directories.
- The dataset used for this project is described in the data directory.
- Additional details about the models, hyperparameters, and evaluation metrics can be found in the corresponding code files.

Please refer to the specific code files and directories for more detailed information and to reproduce the experiments and results presented in this project.

