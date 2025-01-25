import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from prophet import Prophet
from sklearn.preprocessing import LabelEncoder
from prophet.plot import add_changepoints_to_plot

df = pd.read_csv('./data/final_medal_counts_with_sports.csv')


encoder = LabelEncoder()
df['NOC_CODE'] = encoder.fit_transform(df['NOC_CODE'])


# Đảm bảo dataframe có cột 'ds' và 'y'
df['ds'] = pd.to_datetime(df['Year'], format='%Y')  # Chuyển đổi cột 'Year' thành cột 'ds'
df['y'] = df['Gold']  # Là cột bạn muốn dự đoán

# Initialize the Prophet model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.15
)

# Add special seasonality for the Olympic cycle
model.add_seasonality(
    name='4_yearly', 
    period=4*365.25,  # 4 years in days
    fourier_order=8
)

# Add regressors
regressors = df.drop(columns=['y', 'ds', 'Year', 'Rank', 'Gold', 'Silver', 'Bronze', 'Total']).columns
for regressor in regressors:
    model.add_regressor(regressor)

# Fit the model
model.fit(df)

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

# Print future dataframe to check Host values
print(future)

# Predict
forecast = model.predict(future)
# Decode NOC_CODE back to original labels
forecast['NOC_CODE'] = encoder.inverse_transform(countries.astype(int))

# Print forecast
prediction = forecast[['NOC_CODE', 'yhat', 'yhat_lower', 'yhat_upper']]
print(prediction)

import matplotlib.pyplot as plt


import matplotlib.pyplot as plt

# plot the time series 
forecast_plot = model.plot(forecast)

# add a vertical line at the end of the training period
axes = forecast_plot.gca()

# # plot true test data for the period after the red line
# df['Year'] = pd.to_datetime(df['Year'])
# plt.plot(df['Year'], df['Gold'],'ro', markersize=3, label='True Test Data')
# plt.legend()
# plt.show()

fig = model.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), model, forecast)
# plt.savefig('./figures/forecast_plot_2028.png')
plt.show()
#prediction.to_csv('./data/medal_table_2028.csv', index=False)