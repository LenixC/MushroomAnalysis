import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import hmine
from scipy.stats import chi2_contingency

MIN_SUPPORT = .1

mushrooms = pd.read_csv('secondary_data_shuffled.csv', sep=';')


### Graphs missing Values
percent_missing = mushrooms.isnull().sum() * 100 / len(mushrooms)
missing_value_df = pd.DataFrame({'percent_missing': percent_missing})
missing_value_df = missing_value_df.sort_values('percent_missing')
bar = plt.bar(missing_value_df.index, missing_value_df['percent_missing'])
plt.title("Missing Values by Percent")
plt.bar_label(bar)
plt.xticks(rotation=-90)
plt.savefig("MushPrcntMissing.png", bbox_inches="tight")


### Fetch numerical elements to make dummies
mush_nums = mushrooms.select_dtypes(include='number')
num_cols = mush_nums.columns

print(mush_nums.describe().to_latex())

scaler = StandardScaler()
standard_mush_nums = pd.DataFrame(scaler.fit_transform(mush_nums),
                                  columns=num_cols)

encoded_names = ["sm", "smm", "m", "ml", "l", "ll"]
dummy_numeric = pd.DataFrame()

# Converts continuous variable to categorical
for column in standard_mush_nums.columns:
    cut_mush_nums = pd.cut(standard_mush_nums[column], 
                           6,
                           labels=encoded_names)
    cut_mush_nums = pd.DataFrame(cut_mush_nums)
    mush_dum = pd.get_dummies(cut_mush_nums)
    dummy_numeric = pd.concat([dummy_numeric, mush_dum],
                             axis='columns')


### Gets categorical variables and makes dummies
mush_cat = mushrooms.drop(mush_nums.columns, axis='columns')
print(mush_cat.describe().T.to_latex())
#target = "class"
#results = []
#for col in mush_cat.columns:
#    if col == target:
#        continue
#    contingency_table = pd.crosstab(mush_cat[col], mush_cat[target])
#    chi2, p, _, _ = chi2_contingency(contingency_table)
#    results.append((col, chi2, p))
#chi2s = pd.DataFrame(results, columns=["feature", "chi2", "p-value"])
#chi2s = chi2s.set_index('feature')
#print(chi2s)

dummies = pd.get_dummies(mush_cat)
dummies = pd.concat([dummies, dummy_numeric], axis='columns')
dummies = dummies.astype(bool)

### Pattern mining algorithms
#
time_apri = time.time()
apri = apriori(dummies, min_support=MIN_SUPPORT, use_colnames=True)
time_apri = time.time() - time_apri

time_fp = time.time()
fp = fpgrowth(dummies, min_support=MIN_SUPPORT, use_colnames=True)
time_fp = time.time() - time_fp

# H-Mine is an algorithm that minimize space.
time_hmine = time.time()
hpattern = hmine(dummies, min_support=MIN_SUPPORT, use_colnames=True)
time_hmine = time.time() - time_hmine

timings = {'Apriori': time_apri,
           'FPGrowth': time_fp,
           'Hmine': time_hmine}
timing_series = pd.Series(timings)
timing_series.name = "Time (s) for completion"
print(timing_series.to_latex())

### Takes the datasets and filters them by length
### and edibility.
frequent_itemsets = apri.copy()
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets = frequent_itemsets[frequent_itemsets['length'] > 2]

poison = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: 'class_p' in x)]
edible = frequent_itemsets[frequent_itemsets['itemsets'].apply(lambda x: 'class_e' in x)]

#print(poison)
#print(edible)

### Calculates and stores probability of pattern 
### that are poisonous. 
poison_percents = {}
for item_set in poison['itemsets']:
    list_items = list(item_set)
    list_items.remove('class_p')
    mask = dummies[list_items].all(axis=1) & (dummies['class_p'] | dummies['class_e'])
    df = dummies.loc[mask]
    pct_poison = (df['class_p']).sum() / len(df)
    pct_edible = 1-pct_poison
    poison_percents[str(list_items)] = pct_poison
poison_mushes = pd.Series(poison_percents)
poison_mushes = poison_mushes.sort_values(ascending=False)
print(poison_mushes)

