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

# Đọc dữ liệu
df = pd.read_csv('./data/fixed_olympic_ranking.csv', index_col=False)

# Mã hóa cột NOC_CODE
encoder = LabelEncoder()
df['NOC_CODE'] = encoder.fit_transform(df['NOC_CODE'])

# Chia dữ liệu thành train và test
target_variable = 'Total'
year = 2020
df_train = df[df['Year'] < year]
df_test = df[df['Year'] >= year]
X_train, y_train = df_train.drop(columns=['Rank', 'Gold', 'Silver', 'Bronze', 'Total']), df_train[target_variable]
X_test, y_test = df_test.drop(columns=['Rank', 'Gold', 'Silver', 'Bronze', 'Total']), df_test[target_variable]

# Đảm bảo dataframe có cột 'ds' và 'y'
df_train['ds'] = pd.to_datetime(df_train['Year'], format='%Y')  # Chuyển đổi cột 'Year' thành cột 'ds'
df_train['y'] = df_train[target_variable]  # Là cột bạn muốn dự đoán
#df_test = df_test.drop(columns=['Rank', 'Gold', 'Silver', 'Bronze', 'Total'])
df_test['ds'] = pd.to_datetime(df_test['Year'], format='%Y')  # Chuyển đổi cột 'Year' thành cột 'ds'
df_test['y'] = df_test[target_variable]

# Khởi tạo mô hình Prophet
model = Prophet(
    interval_width=0.95,
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.15,
)

# Thêm seasonality đặc biệt cho chu kỳ Olympic
model.add_seasonality(
    name='4_yearly', 
    period=4*365.25,  # 4 năm tính bằng ngày
    fourier_order=10
)

# Thêm các regressor
regressors = df_train.drop(columns=['y', 'ds', 'Year', 'Rank', 'Gold', 'Silver', 'Bronze', 'Total']).columns
for regressor in regressors:
    model.add_regressor(regressor)

df_train = df_train.drop(columns=['Year'])
df_test = df_test.drop(columns=['Year'])

# Fit mô hình
model.fit(df_train)

# Dự đoán
forecast = model.predict(df_test)

# Tính toán residuals
df_test['yhat'] = forecast['yhat']
df_test['residual'] = df_test['y'] - df_test['yhat']
print(df_test)

# Vẽ biểu đồ residuals
plt.figure(figsize=(12, 6))
plt.plot(df_test['ds'], df_test['residual'], marker='o', linestyle='-', color='b', label='Residuals')
plt.fill_between(df_test['ds'], df_test['residual'], 0, where=(df_test['residual'] > 0), interpolate=True, color='green', alpha=0.3, label='Positive Residuals')
plt.fill_between(df_test['ds'], df_test['residual'], 0, where=(df_test['residual'] < 0), interpolate=True, color='red', alpha=0.3, label='Negative Residuals')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Residual', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()