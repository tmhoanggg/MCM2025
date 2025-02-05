import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Đọc dữ liệu từ tệp CSV
df = pd.read_csv('./data/summerOly_athletes.csv')

# Lọc dữ liệu cho môn Volleyball, đội tuyển USA và China, và giới tính nữ
df_volleyball_usa_f = df[(df['Year'].between(1980, 2024)) & (df['Sport'] == 'Volleyball') & \
                         (df['Team'] == 'United States') & (df['Sex'] == 'F')]
df_volleyball_china_f = df[(df['Year'].between(1980, 2024)) & (df['Sport'] == 'Volleyball') & \
                           (df['Team'] == 'China') & (df['Sex'] == 'F')]

# Loại bỏ các bản ghi trùng lặp dựa trên các cột Year, Event, và Medal
df_volleyball_usa_unique = df_volleyball_usa_f.drop_duplicates(subset=['Year', 'Event'])
df_volleyball_china_unique = df_volleyball_china_f.drop_duplicates(subset=['Year', 'Event'])

# Tạo cột điểm cho mỗi loại huy chương
medal_points = {'No medal': 0, 'Bronze': 1, 'Silver': 3, 'Gold': 6}
df_volleyball_usa_unique['Points'] = df_volleyball_usa_unique['Medal'].map(medal_points).fillna(0)
df_volleyball_china_unique['Points'] = df_volleyball_china_unique['Medal'].map(medal_points).fillna(0)

# Tính tổng điểm cho mỗi năm
points_per_year_usa = df_volleyball_usa_unique.groupby('Year')['Points'].sum()
points_per_year_china = df_volleyball_china_unique.groupby('Year')['Points'].sum()

# Danh sách các năm bà Lang Ping làm HLV
langping_years_usa = [2008]
langping_years_china = [1996, 2016, 2020]

# Làm mượt các điểm cho USA
x_usa = np.arange(len(points_per_year_usa))
f_points_usa = interp1d(x_usa, points_per_year_usa, kind='cubic')
x_new_usa = np.linspace(0, len(points_per_year_usa) - 1, num=500)
points_smooth_usa = f_points_usa(x_new_usa)

# Làm mượt các điểm cho China
x_china = np.arange(len(points_per_year_china))
f_points_china = interp1d(x_china, points_per_year_china, kind='cubic')
x_new_china = np.linspace(0, len(points_per_year_china) - 1, num=500)
points_smooth_china = f_points_china(x_new_china)

# Vẽ biểu đồ
fig, ax = plt.subplots(figsize=(12, 8))

# Vẽ các điểm làm mượt cho USA
ax.plot(x_new_usa, points_smooth_usa, '-', color='blue', alpha=0.7, label='USA')

# Vẽ các điểm làm mượt cho China
ax.plot(x_new_china, points_smooth_china, '-', color='green', alpha=0.7, label='China')

# Biến cờ để kiểm soát việc thêm label
langping_label_added = False

# Vẽ các điểm gốc cho USA
for i, year in enumerate(points_per_year_usa.index):
    if year in langping_years_usa:
        if not langping_label_added:
            ax.plot(i, points_per_year_usa[year], 'o', color='red', label='Coached by Lang Ping')
            langping_label_added = True
        else:
            ax.plot(i, points_per_year_usa[year], 'o', color='red')
    else:
        ax.plot(i, points_per_year_usa[year], 'o', color='blue')

# Vẽ các điểm gốc cho China
for i, year in enumerate(points_per_year_china.index):
    if year in langping_years_china:
        if not langping_label_added:
            ax.plot(i, points_per_year_china[year], 'o', color='red', label='Coached by Lang Ping')
            langping_label_added = True
        else:
            ax.plot(i, points_per_year_china[year], 'o', color='red')
    else:
        ax.plot(i, points_per_year_china[year], 'o', color='green')

# Thêm tiêu đề và nhãn
ax.set_title('Số điểm của đội tuyển Volleyball nữ USA và China qua các năm', fontsize=16)
ax.set_xlabel('Năm', fontsize=14)
ax.set_ylabel('Số điểm', fontsize=14)
ax.set_xticks(np.arange(len(points_per_year_usa)))
ax.set_xticklabels(points_per_year_usa.index, rotation=45)

# Thêm chú thích
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), fontsize=12)

# Hiển thị biểu đồ
plt.grid(True)
plt.show()