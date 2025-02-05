import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d


df = pd.read_csv('./data/summerOly_athletes.csv')


# df_gymnastics_usa = df[(df['Sport'] == 'Gymnastics') & (df['NOC'] == 'USA') & (df['Year'] == 1984) & (df['Sex'] == 'F')]
# coach_year = [1980]

# print(df_gymnastics_usa)
# # # Loại bỏ các bản ghi trùng lặp dựa trên các cột Year, Event, và Medal
# df_gymnastics_usa_unique = df_gymnastics_usa.drop_duplicates(subset=['Year', 'Event', 'Medal'])

# print(df_gymnastics_usa_unique)

# # # Nhóm dữ liệu theo năm và đếm số lượng huy chương
# medal_counts = df_gymnastics_usa_unique.groupby('Year')['Medal'].value_counts().unstack().fillna(0)

# # # Hiển thị kết quả tổng hợp cho mỗi năm
# print(medal_counts)
#(df['Name'].str.contains('Julianne', case=False, na=False))

athletes_df = df[(df['Sport'] == 'Gymnastics') & (df['NOC'] == 'USA') & (df['Year'] == 1984) & (df['Sex'] == 'F')]

# Nhóm dữ liệu theo tên vận động viên và đếm số lượng huy chương
medal_counts_per_athlete = athletes_df.groupby('Name')['Medal'].value_counts().unstack().fillna(0)

# Tính điểm cho mỗi vận động viên
medal_counts_per_athlete['Points'] = (
    medal_counts_per_athlete.get('Gold', 0) * 6 +
    medal_counts_per_athlete.get('Silver', 0) * 3 +
    medal_counts_per_athlete.get('Bronze', 0)
)

# Sắp xếp các vận động viên theo điểm số
medal_counts_per_athlete = medal_counts_per_athlete.sort_values(by='Points', ascending=False)

# Làm mượt các điểm
x = np.arange(len(medal_counts_per_athlete))
f_points = interp1d(x, medal_counts_per_athlete['Points'], kind='cubic')
x_new = np.linspace(0, len(medal_counts_per_athlete) - 1, num=500)
points_smooth = f_points(x_new)

# Danh sách các vận động viên được huấn luyện bởi Karolyi
karolyi_athletes = ['Mary', 'Julianne']

# Vẽ biểu đồ
fig, ax = plt.subplots(figsize=(12, 8))

# Vẽ các điểm làm mượt
ax.plot(x_new, points_smooth, '-', color='b', alpha=0.4)

# Biến cờ để kiểm soát việc thêm label
karolyi_label_added = False
other_label_added = False

# Vẽ các điểm gốc
for i, name in enumerate(medal_counts_per_athlete.index):
    if any(athlete in name for athlete in karolyi_athletes):
        if not karolyi_label_added:
            ax.plot(i, medal_counts_per_athlete['Points'].iloc[i], 'o', color='red', label='Athletes coached by Karolyi')
            karolyi_label_added = True
        else:
            ax.plot(i, medal_counts_per_athlete['Points'].iloc[i], 'o', color='red')
    else:
        if not other_label_added:
            ax.plot(i, medal_counts_per_athlete['Points'].iloc[i], 'o', color='blue', label='Other athletes')
            other_label_added = True
        else:
            ax.plot(i, medal_counts_per_athlete['Points'].iloc[i], 'o', color='blue')

# Lấy tên vận động viên không có dấu ngoặc đơn
athlete_names = [name.split('(')[0].strip() for name in medal_counts_per_athlete.index]

# Đặt nhãn trục x
ax.set_xlabel('Athlete', fontsize=14)
ax.set_ylabel('Points', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(athlete_names, rotation=45)

# Thêm chú thích
ax.legend(fontsize=12)

# Hiển thị biểu đồ
plt.grid(True)
plt.show()