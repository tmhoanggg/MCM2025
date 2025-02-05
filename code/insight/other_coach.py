import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

# Đọc dữ liệu
df = pd.read_csv('./data/summerOly_athletes.csv', index_col=False)

sports = ['Gymnastics', 'Judo', 'Swimming', 'Volleyball', 'Wrestling']

# Tạo một DataFrame để lưu trữ kết quả
results = pd.DataFrame()

# Lọc dữ liệu và tính toán điểm cho từng môn thể thao
for sport in sports:
    df_sport = df[(df['Year'].between(1970, 2024)) & (df['Sport'] == sport) & (df['Team'] == 'Japan')]
    medal_counts = df_sport.groupby('Year').agg(
        gold=('Medal', lambda x: (x == 'Gold').sum()),
        silver=('Medal', lambda x: (x == 'Silver').sum()),
        bronze=('Medal', lambda x: (x == 'Bronze').sum())
    ).reset_index()
    medal_counts['points'] = medal_counts['gold'] * 6 + medal_counts['silver'] * 3 + medal_counts['bronze']
    medal_counts['Sport'] = sport
    results = pd.concat([results, medal_counts])

# Làm mượt dữ liệu
results['points_smooth'] = results.groupby('Sport')['points'].transform(lambda x: gaussian_filter1d(x, sigma=2))

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
sns.lineplot(data=results, x='Year', y='points_smooth', hue='Sport', marker='o')
#plt.title('Points Scored by Japan Teams in Various Sports (1970-2024)', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Points', fontsize=14)
plt.xticks(rotation=0)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(title='Sport')
plt.show()