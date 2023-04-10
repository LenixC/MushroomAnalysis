import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori

mushrooms = pd.read_csv('secondary_data_shuffled.csv', sep=';')

percent_missing = mushrooms.isnull().sum() * 100 / len(mushrooms)
missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
missing_value_df = missing_value_df.sort_values('percent_missing')
bar = plt.bar(missing_value_df.index, missing_value_df['percent_missing'])
plt.title("Missing Values by Percent")
plt.bar_label(bar)
plt.xticks(rotation=-30)
plt.show()

mush_nums = mushrooms.select_dtypes(include='number')
num_cols = mush_nums.columns
scaler = StandardScaler()
standard_mush_nums = pd.DataFrame(scaler.fit_transform(mush_nums),
                                  columns=num_cols)

encoded_names = ["sm", "smm", "m", "ml", "l", "ll"]

dummy_numeric = pd.DataFrame()

for column in standard_mush_nums.columns:
    cut_mush_nums = pd.cut(standard_mush_nums[column], 
                           6,
                           labels=encoded_names)
    cut_mush_nums = pd.DataFrame(cut_mush_nums)
    mush_dum = pd.get_dummies(cut_mush_nums)
    dummy_numeric = pd.concat([dummy_numeric, mush_dum],
                             axis='columns')
print(dummy_numeric)

mush_cat = mushrooms.drop(mush_nums.columns, axis='columns')
dummies = pd.get_dummies(mush_cat)
dummies = pd.concat([dummies, dummy_numeric], axis='columns')
dummies = dummies.astype(bool)
print(dummies)

apri = apriori(dummies, min_support=0.1, use_colnames=True)


frequent_itemsets = apri.copy()
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets = frequent_itemsets[frequent_itemsets['length'] > 1]

poison = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: 'class_p' in x)]
edible = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: 'class_e' in x)]

print(poison)
print(edible)



