import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split

# 加载数据
df = pd.read_csv("C:\\Users\\code\\DataScience\\user_personalized_features.csv")

# 数值变量相关系数矩阵(保留两位小数)
numeric_df = df[['Age', 'Income', 'Last_Login_Days_Ago', 'Purchase_Frequency', 'Average_Order_Value', 'Total_Spending', 'Time_Spent_on_Site_Minutes', 'Pages_Viewed']]
correlation_matrix = numeric_df.corr().round(2)
print("数值变量相关系数矩阵: ")
print(correlation_matrix)

# 分类变量卡方检验
categorical_cols = ['Gender', 'Location', 'Interests', 'Product_Category_Preference', 'Newsletter_Subscription']
chi2_results = {}
for i in range(len(categorical_cols)):
    for j in range(i + 1, len(categorical_cols)):
        contingency_table = pd.crosstab(df[categorical_cols[i]], df[categorical_cols[j]])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        chi2_results[(categorical_cols[i], categorical_cols[j])] = (chi2.round(2), p.round(2))

print("\n分类变量卡方检验结果: ")
for pair, (chi2_value, p_value) in chi2_results.items():
    print(f'{pair}: Chi2 = {chi2_value}, p = {p_value}')

# 对Age列分箱, 分为5箱
df['Age_bins'] = pd.cut(df['Age'], bins=5)

# 对Income列分箱, 分为5箱
df['Income_bins'] = pd.cut(df['Income'], bins=5)

# 查看分箱结果
print("Age列分箱结果: ")
print(df['Age_bins'].value_counts())

print("\nIncome列分箱结果: ")
print(df['Income_bins'].value_counts())


# 按Location分层抽样
stratified_sample = pd.DataFrame()
for location in df['Location'].unique():
    subset = df[df['Location'] == location]
    sample_subset = subset.sample(frac=0.2, random_state=42)
    stratified_sample = pd.concat([stratified_sample, sample_subset])

# 将结果保存为csv文件
csv_path = "C:\\Users\\code\\DataScience\\user_personalized_features_reduced.csv"
stratified_sample.to_csv(csv_path)