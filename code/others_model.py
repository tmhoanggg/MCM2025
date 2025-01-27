import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


df = pd.read_csv('./data/fixed_olympic_ranking.csv', index_col=False)


encoder = LabelEncoder()
df['NOC_CODE'] = encoder.fit_transform(df['NOC_CODE'])


target_variable = 'Bronze'
year = 2020
df_train = df[df['Year'] < year]
df_test = df[df['Year'] >= year]
X_train, y_train = df_train.drop(columns=['Rank', 'Gold', 'Silver', 'Bronze', 'Total']), df_train[target_variable]
X_test, y_test = df_test.drop(columns=['Rank', 'Gold', 'Silver', 'Bronze', 'Total']), df_test[target_variable]


# Đảm bảo dataframe có cột 'ds' và 'y'
df_train['ds'] = pd.to_datetime(df_train['Year'], format='%Y')  # Chuyển đổi cột 'Year' thành cột 'ds'
df_train['y'] = df_train[target_variable]  # Là cột bạn muốn dự đoán
df_test = df_test.drop(columns=['Rank', 'Gold', 'Silver', 'Bronze', 'Total'])
df_test['ds'] = pd.to_datetime(df_test['Year'], format='%Y')  # Chuyển đổi cột 'Year' thành cột 'ds'
#df_test['y'] = df_test[target_variable]  # Là cột bạn muốn dự đoán

# Initialize the Prophet model
model = Prophet(
    interval_width=0.95,
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.15,
)

# Add special seasonality for the Olympic cycle
model.add_seasonality(
    name='4_yearly', 
    period=4*365.25,  # 4 years in days
    fourier_order=10
)

# Add regressors
regressors = df_train.drop(columns=['y', 'ds', 'Year', 'Rank', 'Gold', 'Silver', 'Bronze', 'Total']).columns
for regressor in regressors:
    model.add_regressor(regressor)


df_train = df_train.drop(columns=['Year'])
df_test = df_test.drop(columns=['Year'])

# Fit the model
model.fit(df_train)

# Predict
forecast = model.predict(df_test)
pred = forecast['yhat']

rmse = round(np.sqrt(mean_squared_error(y_test, pred)), 3)
mae = round(mean_absolute_error(y_test, pred), 3)
r2 = round(r2_score(y_test, pred), 3)
print(f"{rmse} & {mae} & {r2}")