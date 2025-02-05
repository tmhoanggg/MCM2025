import pandas as pd
import numpy as np


gold_df = pd.read_csv('./result_data/gold_2028.csv')
silver_df = pd.read_csv('./result_data/silver_2028.csv')
bronze_df = pd.read_csv('./result_data/bronze_2028.csv')

gold_df.rename(columns={'yhat': 'Gold', 'yhat_lower': 'Gold_lower', 'yhat_upper': 'Gold_upper'}, inplace=True)
silver_df.rename(columns={'yhat': 'Silver', 'yhat_lower': 'Silver_lower', 'yhat_upper': 'Silver_upper'}, inplace=True)
bronze_df.rename(columns={'yhat': 'Bronze', 'yhat_lower': 'Bronze_lower', 'yhat_upper': 'Bronze_upper'}, inplace=True)

final_df = pd.merge(gold_df, silver_df, on='NOC_CODE', how='inner')
final_df = pd.merge(final_df, bronze_df, on='NOC_CODE', how='inner')
final_df['Total'] = final_df['Gold'] + final_df['Silver'] + final_df['Bronze']
final_df['Total_lower'] = final_df['Gold_lower'] + final_df['Silver_lower'] + final_df['Bronze_lower']
final_df['Total_upper'] = final_df['Gold_upper'] + final_df['Silver_upper'] + final_df['Bronze_upper']


# Sắp xếp dataframe theo số lượng huy chương vàng, bạc, và đồng
final_df = final_df.sort_values(by=['Gold', 'Silver', 'Bronze'], ascending=False)

# Tính toán thứ hạng (Rank)
final_df['Rank'] = final_df[['Gold', 'Silver', 'Bronze']].apply(tuple, axis=1).rank(method='min', ascending=False).astype(int)

# Lấy 10 đội đầu tiên
top_10_df = final_df.head(10)
print(top_10_df[['Rank', 'NOC_CODE', 'Gold_lower', 'Gold_upper', 'Silver_lower', 'Silver_upper',\
                 'Bronze_lower', 'Bronze_upper', 'Total_lower', 'Total_upper']])

# # Hàm chuyển đổi dataframe thành chuỗi định dạng LaTeX
# def to_latex_format(df):
#     latex_str = ""
#     for index, row in df.iterrows():
#         latex_str += f"{row['Rank']} & {row['NOC_CODE']} & {row['Gold']} & {row['Silver']} & {row['Bronze']} & {row['Total']} \\\\\n"
#     return latex_str

# # In ra kết quả theo định dạng LaTeX
# latex_output = to_latex_format(top_10_df)
# print(latex_output)