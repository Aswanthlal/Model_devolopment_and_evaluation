 import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#load dataset
filepath="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv"
df=pd.read_csv(filepath,header=0)
df.head(5)

#Visualizing induvidual feature patterns 
sns.regplot(x="CPU_frequency", y="Price", data=df)
plt.ylim(0,)

sns.regplot(x="Screen_Size_inch", y="Price", data=df)
plt.ylim(0,)

sns.regplot(x="Weight_pounds", y="Price", data=df)
plt.ylim(0,)

# Correlation values of the three attributes with Price
for param in ["CPU_frequency", "Screen_Size_inch","Weight_pounds"]:
    print(f"Correlation of Price and {param} is ", df[[param,"Price"]].corr())

#categorical features
#Generating Box plots for the different feature that hold categorical values
#category box plot
sns.boxplot(x="Category", y="Price", data=df)
#GPU boxplot
sns.boxplot(x="GPU", y="Price", data=df)
#OS box plot
sns.boxplot(x="OS", y="Price", data=df)
#CPU_core box plot
sns.boxplot(x="CPU_core", y="Price", data=df)
#RAM_GB box plot
sns.boxplot(x="RAM_GB", y="Price", data=df)
# Storage_GB_SSD Box plot
sns.boxplot(x="Storage_GB_SSD", y="Price", data=df)


#descriptive statistical analysis
print(df.describe())
print(df.describe(include=['object']))

#groupby and pivot tables
#create group
df_gptest = df[['GPU','CPU_core','Price']]
grouped_test1 = df_gptest.groupby(['GPU','CPU_core'],as_index=False).mean()
print(grouped_test1)

#create pivot table
grouped_pivot = grouped_test1.pivot(index='GPU',columns='CPU_core')
print(grouped_pivot)

# Create the Plot
fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

fig.colorbar(im)

#pearson corelation and p values
for param in ['RAM_GB','CPU_frequency','Storage_GB_SSD','Screen_Size_inch','Weight_pounds','CPU_core','OS','GPU','Category']:
    pearson_coef, p_value = stats.pearsonr(df[param], df['Price'])
    print(param)
    print("The Pearson Correlation Coefficient for ",param," is", pearson_coef, " with a P-value of P =", p_value)
