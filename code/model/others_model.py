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
from scipy.interpolate import interp1d


df = pd.read_csv('./data/summerOly_medal_counts.csv', index_col=False)
# Lọc dữ liệu trước năm 2024
df_before_2024 = df[df['Year'] < 2024]
countries_before_2024 = df_before_2024['NOC'].unique()

df_2024 = df[df['Year'] == 2024]
countries_win_first_medal = set(df_2024['NOC'].unique()) - set(countries_before_2024)

# Mã hóa cột NOC_CODE
df_NOC = pd.read_csv('./data/summerOly_athletes.csv')
df_NOC = df_NOC[df_NOC['Team'].isin(countries_win_first_medal)]
first_medal_list = df_NOC['NOC'].unique()
print(first_medal_list)


df = pd.read_csv('./data/fixed_olympic_ranking.csv', index_col=False)

encoder = LabelEncoder()
df['NOC_CODE'] = encoder.fit_transform(df['NOC_CODE'])

target_variable = 'Gold'
year = 2024
df_train = df[df['Year'] < year]
df_test = df[df['Year'] >= year]
X_train, y_train = df_train.drop(columns=['Rank', 'Gold', 'Silver', 'Bronze', 'Total']), df_train[target_variable]
X_test, y_test = df_test.drop(columns=['Rank', 'Gold', 'Silver', 'Bronze', 'Total']), df_test[target_variable]

y = df_test[target_variable]

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
noc = list(df_test['NOC_CODE'])

# Fit the model
model.fit(df_train)

# Predict
forecast = model.predict(df_test)
forecast = forecast[['NOC_CODE', 'yhat', 'yhat_lower', 'yhat_upper']]
forecast['NOC_CODE'] = noc
forecast['NOC_CODE'] = encoder.inverse_transform(forecast['NOC_CODE'])
print(forecast)

low_country = forecast[forecast['NOC_CODE'].isin(first_medal_list)]
print(low_country)

# # Sắp xếp và in ra top 10 quốc gia theo yhat
# sorted_forecast = forecast.sort_values(by='yhat', ascending=False)
# print(sorted_forecast.head(10))
print(forecast[83:].sort_values(by='yhat', ascending=False).head(10))


# mae = mean_absolute_error(y, forecast['yhat'])
# mse = mean_squared_error(y, forecast['yhat'])
# r2 = r2_score(y, forecast['yhat'])
# print(f"Mean Absolute Error: {mae}")
# print(f"Mean Squared Error: {mse}")
# print(f"R2 Score: {r2}")


# number_of_countries = 200
# forecast = forecast[:number_of_countries]
# y = y[:number_of_countries]

# # Vẽ hình
# x = np.arange(len(forecast['NOC_CODE']))
# y_actual = y
# y_predicted = forecast['yhat']

# # Làm mượt các đường bằng interp1d
# f_actual = interp1d(x, y_actual, kind='cubic')
# f_predicted = interp1d(x, y_predicted, kind='cubic')
# x_new = np.linspace(0, len(forecast['NOC_CODE']) - 1, num=500)
# y_actual_smooth = f_actual(x_new)
# y_predicted_smooth = f_predicted(x_new)

# x_new = x
# y_actual_smooth = y_actual
# y_predicted_smooth = y_predicted

# plt.figure(figsize=(10, 6))
# plt.plot(x_new, y_actual_smooth, label="Actual Total medals", color="#44AA99")
# plt.plot(x_new, y_predicted_smooth, label="Predicted Total medals", color="#CC6677")
# plt.fill_between(
#     x_new,
#     y_actual_smooth,
#     y_predicted_smooth,
#     color="gray",
#     alpha=0.1,
#     label="Residuals"
# )
# plt.title("Actual vs Predicted Total Medals")
# plt.xlabel("Countries")
# plt.ylabel("Number of Total Medals")
# plt.legend()
# plt.tight_layout()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.show()