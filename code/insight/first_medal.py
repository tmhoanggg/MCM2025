import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import norm
import scipy.stats as stats

def prob(yhat, yhat_lower, yhat_upper, z=1.96):
    sigma = (yhat_upper - yhat_lower) / (2 * z)
    return 1 - norm.cdf((1 - yhat) / sigma)

df = pd.read_csv('./result_data/total_2028.csv', index_col=False)
# Trung bình
mean = np.mean(df['yhat'])

# Phương sai
variance = np.var(df['yhat'], ddof=1)  # ddof=1 để tính phương sai không chệch

print(f"Trung bình: {mean}")
print(f"Phương sai: {variance}")
#df['prob'] = prob(df['yhat'], df['yhat_lower'], df['yhat_upper'])
#df['prob'] = 1 - np.exp(-df['yhat'])
#print(no_medal_df[19:30])


df_athletes = pd.read_csv('./data/summerOly_athletes.csv', index_col=False)

# Lọc các nước chưa từng có huy chương
countries_with_medals = df_athletes[df_athletes['Medal'] != 'No medal']['NOC'].unique()
all_countries = df_athletes['NOC'].unique()
countries_no_medals = list(set(all_countries) - set(countries_with_medals))


no_medal_df = df[df['NOC_CODE'].isin(countries_no_medals)].sort_values(by='yhat', ascending=False)
print(no_medal_df.head(10))


# std_dev = variance**0.5
# # Prior
# prior = no_medal_df['yhat']  # Giả định không thông tin

# # Likelihood P(B|A)
# threshold = 1
# p_b_given_a = 1 - stats.norm.cdf(threshold, loc=mean, scale=std_dev)

# # Likelihood P(B|~A)
# p_b_given_not_a = stats.norm.cdf(threshold, loc=mean, scale=std_dev)

# # Xác suất dữ liệu P(B)
# p_b = p_b_given_a * prior + p_b_given_not_a * (1 - prior)

# # Posterior P(A|B)
# posterior = (p_b_given_a * prior) / p_b

# no_medal_df['posterior'] = posterior
# print(no_medal_df)
#print(df_athletes[(df_athletes['NOC']=='TLS') & (df_athletes['Medal'] != 'No medal')])

# # Vẽ biểu đồ cột
# plt.figure(figsize=(12, 8))
# plt.barh(first_medal_df['NOC_CODE'], first_medal_df['prob'], color='skyblue')
# plt.xlabel('Probability', fontsize=14)
# plt.ylabel('Country', fontsize=14)
# plt.title('Top 10 Countries with Highest Probability of Winning First Medal in 2028', fontsize=16)
# plt.gca().invert_yaxis()  # Đảo ngược trục y để nước có xác suất cao nhất ở trên cùng
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()

# # Lấy top 20 nước có khả năng giành huy chương đầu tiên
# prediction = no_medal_df[:20]

# # Chuyển đổi NOC_CODE thành số nguyên để nội suy
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
# ax.set_title('Dự đoán nước có khả năng giành huy chương đầu tiên', fontsize=16)
# ax.set_xlabel('Quốc gia', fontsize=14)
# ax.set_ylabel('Số lượng huy chương dự đoán', fontsize=14)

# # Đặt nhãn trục x là tên các nước
# ax.set_xticks(np.arange(1, len(prediction['NOC_CODE']) + 1))
# ax.set_xticklabels(prediction['NOC_CODE'], rotation=45, ha='right')

# # Hiển thị biểu đồ
# plt.grid(True)
# plt.legend()
# plt.show()