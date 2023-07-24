# GroundWater_Level_Prediction_using_N_Beats
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

