import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('./data/summerOly_athletes.csv')

# Lọc dữ liệu cho môn Volleyball, đội tuyển China và giới tính nữ
df_volleyball_china_f = df[(df['Sport'] == 'Volleyball') & (df['Team'] == 'China') & (df['Sex'] == 'F')]
#df_volleyball_china_f = df[(df['Sport'] == 'Volleyball') & (df['Team'] == 'United States') & (df['Sex'] == 'F')]

# Loại bỏ các bản ghi trùng lặp dựa trên các cột Year, Event, và Medal
df_volleyball_china_unique = df_volleyball_china_f.drop_duplicates(subset=['Year', 'Event'])


# Nhóm dữ liệu theo năm và đếm số lượng huy chương
#medal_counts = df_volleyball_china_unique.groupby('Year')['Medal'].value_counts().unstack().fillna(0)


# Tạo cột điểm cho mỗi loại huy chương
medal_points = {'No medal': 1, 'Bronze': 2, 'Silver': 3, 'Gold': 4}
df_volleyball_china_unique['Points'] = df_volleyball_china_unique['Medal'].map(medal_points).fillna(1)

# Tính tổng điểm cho mỗi năm
points_per_year = df_volleyball_china_unique.groupby('Year')['Points'].sum()

# Danh sách các năm đặc biệt
langping_years = [1996, 2016, 2020]

# Vẽ biểu đồ điểm theo thời gian
fig, ax = plt.subplots(figsize=(12, 8))

# Vẽ các điểm cho tổng điểm qua các năm
ax.plot(points_per_year.index, points_per_year.values, marker='o', markersize=8, linestyle='--', color='b', alpha=0.7, label='Tổng điểm')

# Đổi màu các điểm tương ứng với các năm trong danh sách langping_years
for year in langping_years:
    if year in points_per_year.index:
        ax.plot(year, points_per_year[year], marker='o', color='r', markersize=8, label=f'Năm đặc biệt {year}')

# Thêm tiêu đề và nhãn
ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Medal', fontsize=14)
ax.grid(True)

# Đặt các nhãn cho trục dọc
ax.set_yticks([1, 2, 3, 4])
ax.set_yticklabels(['No medal', 'Bronze', 'Silver', 'Gold'])

# Thêm chú thích
handles, labels = ax.get_legend_handles_labels()
by_label = {'Coached by Lang Ping': handles[1], 'Others': handles[0]}
ax.legend(by_label.values(), by_label.keys(), fontsize=12)

# Hiển thị biểu đồ
plt.show()