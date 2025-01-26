import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('./data/summerOly_athletes.csv')


df_gymnastics_usa = df[(df['Sport'] == 'Gymnastics') & (df['NOC'] == 'USA') & (df['Year'].between(1980, 2004))]

# Loại bỏ các bản ghi trùng lặp dựa trên các cột Year, Event, và Medal
df_gymnastics_usa_unique = df_gymnastics_usa.drop_duplicates(subset=['Year', 'Event'])

# Nhóm dữ liệu theo năm và đếm số lượng huy chương
medal_counts = df_gymnastics_usa_unique.groupby('Year')['Medal'].value_counts().unstack().fillna(0)

# Hiển thị kết quả tổng hợp cho mỗi năm
print(medal_counts)