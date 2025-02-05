import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prophet import Prophet

# Đọc dữ liệu
df = pd.read_csv('./data/fixed_olympic_ranking.csv', index_col=False)

# Mã hóa cột NOC_CODE
encoder = LabelEncoder()
df['NOC_CODE'] = encoder.fit_transform(df['NOC_CODE'])

# Chia dữ liệu thành train và test
target_variable = 'Gold'
year = 2020
df_train = df[df['Year'] < year]
df_test = df[df['Year'] >= year]
X_train, y_train = df_train.drop(columns=['Rank', 'Gold', 'Silver', 'Bronze', 'Total']), df_train[target_variable]
X_test, y_test = df_test.drop(columns=['Rank', 'Gold', 'Silver', 'Bronze', 'Total']), df_test[target_variable]

# Đảm bảo dataframe có cột 'ds' và 'y'
df_train['ds'] = pd.to_datetime(df_train['Year'], format='%Y')  # Chuyển đổi cột 'Year' thành cột 'ds'
df_train['y'] = df_train[target_variable]  # Là cột bạn muốn dự đoán
df_test['ds'] = pd.to_datetime(df_test['Year'], format='%Y')  # Chuyển đổi cột 'Year' thành cột 'ds'
df_test['y'] = df_test[target_variable]  # Là cột bạn muốn dự đoán

# Tách cột ngày tháng
ds_train = df_train['ds']
ds_test = df_test['ds']

# Loại bỏ cột ngày tháng trước khi áp dụng fit_transform
df_train = df_train.drop(columns=['ds', 'Year'])
df_test = df_test.drop(columns=['ds', 'Year'])

# Áp dụng fit_transform
scaler = StandardScaler()
df_train_scaled = scaler.fit_transform(df_train)
df_test_scaled = scaler.transform(df_test)

# Chuyển đổi lại thành DataFrame và ghép lại cột ngày tháng
df_train_scaled = pd.DataFrame(df_train_scaled, columns=df_train.columns)
df_test_scaled = pd.DataFrame(df_test_scaled, columns=df_test.columns)
df_train_scaled['ds'] = ds_train.reset_index(drop=True)
df_test_scaled['ds'] = ds_test.reset_index(drop=True)

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
regressors = df_train.columns.drop(['y'])
for regressor in regressors:
    model.add_regressor(regressor)

# Fit the model
model.fit(df_train_scaled)

# Predict
forecast = model.predict(df_test_scaled)
forecast = forecast[forecast['NOC_CODE', 'yhat', 'yhat_lower', 'yhat_upper']]
print(forecast)