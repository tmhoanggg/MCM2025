import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet
from sklearn.preprocessing import LabelEncoder
from prophet.plot import add_changepoints_to_plot
import matplotlib.pyplot as plt

df = pd.read_csv('./data/fixed_olympic_ranking.csv')


encoder = LabelEncoder()
df['NOC_CODE'] = encoder.fit_transform(df['NOC_CODE'])


# Đảm bảo dataframe có cột 'ds' và 'y'
df['ds'] = pd.to_datetime(df['Year'], format='%Y')  # Chuyển đổi cột 'Year' thành cột 'ds'
df['y'] = df['Total']  # Là cột bạn muốn dự đoán

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
regressors = df.drop(columns=['y', 'ds', 'Year', 'Rank', 'Gold', 'Silver', 'Bronze', 'Total']).columns
for regressor in regressors:
    model.add_regressor(regressor)

# Make future dataframe for Olympics 2028 for all countries
df_2024 = df[df['Year']==2024]
countries = df_2024['NOC_CODE'].unique()

future = pd.DataFrame({
    'ds': pd.to_datetime(['2028-01-01'] * len(countries)),
    'NOC_CODE': countries,
    'Host': 0,
})

# Add regressors into dataframe future
for regressor in regressors:
    if regressor not in ['NOC_CODE', 'Host']:
        future[regressor] = df_2024[regressor].values

# Set Host for USA in 2028
future.loc[future['NOC_CODE'] == encoder.transform(['USA'])[0], 'Host'] = 1

df = df.drop(columns=['Year'])

# Fit the model
model.fit(df)


# Predict
forecast = model.predict(future)
# Decode NOC_CODE back to original labels
forecast['NOC_CODE'] = encoder.inverse_transform(countries.astype(int))

# Làm tròn các giá trị dự đoán thành số nguyên
#forecast[['yhat', 'yhat_lower', 'yhat_upper']] = forecast[['yhat', 'yhat_lower', 'yhat_upper']].round().astype(int)

# Print forecast
prediction = forecast[['NOC_CODE', 'yhat', 'yhat_lower', 'yhat_upper']]
prediction.to_csv('./result_data/total_2028.csv', index=False)
#prediction.to_csv('./result_data/bronze_2028.csv', index=False)

# fig = model.plot(forecast)
# a = add_changepoints_to_plot(fig.gca(), model, forecast)
# # plt.savefig('./figures/forecast_plot_2028.png')
# plt.show()

#model.predictive_samples(future)


# from scipy.interpolate import interp1d
# # Chuyển đổi NOC_CODE thành số nguyên để nội suy

# prediction.sort_values(by='yhat', ascending=False, inplace=True)
# prediction = prediction[:20]
# x = np.arange(len(prediction['NOC_CODE'])) + 1

# # Tạo các hàm nội suy
# f_yhat = interp1d(x, prediction['yhat'], kind='cubic')
# f_yhat_upper = interp1d(x, prediction['yhat_upper'], kind='cubic')
# f_yhat_lower = interp1d(x, prediction['yhat_lower'], kind='cubic')

# # Tạo các điểm dữ liệu mượt mà hơn
# x_new = np.linspace(1, len(prediction['NOC_CODE']), num=500)
# yhat_smooth = f_yhat(x_new)
# yhat_upper_smooth = f_yhat_upper(x_new)
# yhat_lower_smooth = f_yhat_lower(x_new)

# # Vẽ biểu đồ
# fig, ax = plt.subplots(figsize=(12, 8))

# # Vẽ đường chính yhat
# ax.plot(x_new, yhat_smooth, '-', color='b', alpha=0.4, label='Dự đoán')

# # Vẽ các đường phụ yhat_upper và yhat_lower
# ax.plot(x_new, yhat_upper_smooth, '-', color='b', alpha=0.15, label='Dự đoán cao nhất')
# ax.plot(x_new, yhat_lower_smooth, '-', color='b', alpha=0.15, label='Dự đoán thấp nhất')

# # Thêm phần diện tích giữa yhat_upper và yhat_lower
# ax.fill_between(x_new, yhat_lower_smooth, yhat_upper_smooth, color='b', alpha=0.05)

# # Thêm tiêu đề và nhãn
# #title = r'$\bf{\textcolor{gold}{Gold}}$ Medals Prediction for Olympic 2028'
# ax.set_title(r'Gold Medals Prediction for Olympic 2028', fontsize=18, fontweight='bold')
# #ax.title.set_text('Gold Medals Prediction for Olympic 2028', fontsize=14, color='gold')
# ax.set_xlabel('Ranking', fontsize=14)
# ax.set_ylabel('Number of medals', fontsize=14)
# # Bỏ đi các số ở trục x
# ax.set_xticks([0, 5, 10, 15])

# # Hiển thị biểu đồ
# plt.grid(True)
# plt.show()