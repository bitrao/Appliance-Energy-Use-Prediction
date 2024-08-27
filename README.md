

# Introduction

Run [Energy-Used-Prediction.ipynb](../main/Energy-Used-Prediction.ipynb) to reproduce the project

This project aims to predict the energy consumption of lights and appliances in a low-energy building using the UCI Electricity dataset collected in 2017. The dataset includes over 19,735 instances, each representing a 10-minute interval over 4.5 months, capturing variables such as home appliance usage, weather patterns, and environmental factors (temperature and humidity). The primary objective is to leverage machine learning techniques to accurately estimate energy usage, providing insights that could optimize energy utilization and contribute to more efficient energy management systems.

This notebook uses multiple different libraries to study and process the data, as well as implying various machine learning algorithms to achieve the goal. Most of the tools are from scikit-learn and there are three algorithms that are used: Linear Regression, LightGBM and RandomForestRegressor.

# Dataset Information

The dataset was retrieved from the UCI Machine Learning Repository.
Dataset source: Appliance Energy Prediction Dataset

**Number of Instances:** 19375  

**Number of Attributes:**	28 (covering temperature and humidity in different rooms as well as external environmental factors)

For this project, the provided dataset was initially modified and required minor adjustments for further usage. To elaborate, the entries are not in a chronological order and needed to be sorted so that the analysis and utilization for training models can be correctly conducted. 


# Approach
Multiples preprocessing methods are used to make the data ready for model training

### Alternative Target Variable
TARGET_energy is heavily right skewed; therefore, log transformation was used to address this and ‘log_energy’ variable was created. 
As training the models, the chosen target value was ‘log_energy’ and certainly, the prediction is the log energy that can be calculated to the true energy assumption later. 
Log2 is used since the value range is small so that the transformed variable range is reasonable. 
Additionally, due to there was no 0 value in this figure, the chosen transform formula was log⁡2(x).

### Lag Features
Lag features are essential in time series analysis, allowing models to incorporate past data to predict future outcomes. In the context of predicting energy use in low-energy buildings, lag features help capture temporal patterns, such as daily or seasonal cycles, that influence energy consumption. 
By shifting the time series data, these features enable more accurate forecasting by leveraging historical energy usage. 
The interval of each data variances is 10 minutes, 5 lag features are created: 10 minutes before, 30 minutes before, an hour before, 3 hours before and 1 day before respectively. 
Certainly, by introducing the lag features, there would be data points that had missing lag features due to the absence of previous time data. The missing rows are reasonably removed.

### Time-based features
On the other hand, time-based features are considered significant for time-series analysis as they can capture the temporal patterns and trends within the dataset that can impact the prediction ability of machine learning models. 
Added features included: ‘minute’, ‘hour’, ‘day_of_week’, ‘day_of_month’, ‘month’, ‘is_weekday’, and ‘is_daytime’.

### Average features
There are 9 temperature and 9 humidity measurements corresponding to different rooms in the building. 
Average variables for these 2 factors were also created.

### Principal Component Analysis
Overall, there were roughly 44 features that were used for the machine learning models, therefore, Principal Component Analysis (PCA) was used to reduce the dimensionality of data while retaining most of its variance. 
It transforms correlated features into a set of uncorrelated principal components, making it easier to visualize, analyze, and model high-dimensional data. 
It helps in reducing multicollinearity, simplifying the data and certainly, expectedly improving model performance. To choose the number of components we pick the value that retain a significant portion of the total variance (>=95%). 
For this problem, 8 components were enough to capture more than 95% of the total variance of the dataset.


# Evaluation


##### Linear Regression Models Actual and Predictions Plot (from left to right: *without lag features*, *with lag features*, *with PCA*)

<p align="center">
  <img src="https://github.com/tringuyenbao/Appliance-Energy-Use-Prediction/blob/main/images/linear-regression-models-actual-predict-plot.png?raw=true" alt="linear-regression-models-actual-predict-plot"/>
</p>

##### LGBM Models Actual and Predictions Plot (from left to right: *without lag features*, *with lag features*, *with PCA*)

<p align="center">
  <img src="https://github.com/tringuyenbao/Appliance-Energy-Use-Prediction/blob/main/images/lgbm-models-actual-predict-plot.png?raw=true" alt="lgbm-models-actual-predict-plot"/>
</p>

##### LGBM Feature Importance (from left to right: *without lag features*, *with lag features*

<p align="center">
  <img src="https://github.com/tringuyenbao/Appliance-Energy-Use-Prediction/blob/main/images/lgbm-features-importance.png?raw=true" alt="lgbm-models-actual-predict-plot"/>
</p>

##### Random Forest Regressor Models Actual and Prediction Plot (from left to right: *without lag features*, *with lag features*, *with PCA*)

<p align="center">
  <img src="https://github.com/tringuyenbao/Appliance-Energy-Use-Prediction/blob/main/images/random-forest-regressor-models-actual-predict-plot.png?raw=true" alt="random-forest-regressor-models-actual-predict-plot"/>
</p>

### Evaluation metrics of models
- *cross validation for train score*
- *predict on test set for test score*

| Models                                       | Train RMSE | Train R2 | Test RMSE | Test R2 |
|----------------------------------------------|------------|----------|-----------|---------|
|Linear Regression without lag features        | 100.348    | 0.04     | 85.176    | 0.078   |
|**Linear Regresssion with lag features**      | 68.794     | 0.696    | 59.624    | 0.548   |
|Linear Regresssion with PCA                   | 87.671     | 0.475    | 75.453    | 0.276   |
|LightGBM without lag features                 | 98.22      | 0.202    | 95.871    | -0.169  |
|LightGBM with lag features                    | 71.069     | 0.686    | 59.449    | 0.551   |
|LightGBM with PCA                             | 89.608     | 0.421    | 79.024    | 0.206   |
|RandomForestRegressor without lag features    | 97.649     | 0.129    | 98.937    | -0.244  |
|RandomForestRegressor with lag features       | 73.717     | 0.625    | 73.75     | 0.308   |
|RandomForestRegressor with PCA                | 88.921     | 0.401    | 78.614    | 0.214   |

- Models that was trained without lag features has high average errors and does not fit the data well as can be seen through the extremely low R2
- With the lag features, the performance significantly increased with the lower average RMSE score from cross validating, and even lower with unseen test set. These models also fit the data better with decent R2 scores in both set.
- Models with PCA performs better than the default one, but still worse than the model with lag features.
- With feature importances plot, it can be clearly seen that the **lag features** are crucial for the estimator.

=> Overall, **Linear Regression with lag features** has the highest performance.

*Assignment 2 of COSC3013-Computational Machine Learning RMIT 2024*
