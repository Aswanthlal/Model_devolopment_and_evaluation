import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#load dataset
filepath= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod1.csv"
df=pd.read_csv(filepath)

#get dataframe summary
print(df.info())
#view first 5 rows
df.head(5)

#round all value of the screen size colunm to nearest 2 decimals 
df[['Screen_Size_cm']] = np.round(df[['Screen_Size_cm']],2)

#evaluate missing data
missing_data=df.isnull()
print(missing_data.head())
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")

#replacing missing value with mean    
avg_weight=df['Weight_kg'].astype("float").mean(axis=0)
df['Weight_kg'].replace(np.nan,avg_weight,inplace=True)

#replacing with most frequent value
common_screen_size = df['Screen_Size_cm'].value_counts().idxmax()
df["Screen_Size_cm"].replace(np.nan, common_screen_size, inplace=True)

#fixing datatypes
df[['Weight_kg', 'Screen_Size_cm']] = df[['Weight_kg', 'Screen_Size_cm']].astype('float')

#data standardization: convert weight to pounds
#rename colunm
df['Weight_kg']=df['Weight_kg']*2.205
df.rename(columns={'Weight_kg':'Weight_pounds'},inplace=True)

#convert screen size to inch
#rename colunm
df['Screen_Size_cm']=df['Screen_Size_cm']/2.54
df.rename(columns={'Screen_Size_cm':'Screen_Size_inch'},inplace=True)

#data normalization
df['CPU_frequency']=df['CPU_frequency']/df['CPU_frequency'].max()

#binning(creating a categorical attribute which splits the values of a continuous data into a specified number of groups)
bins=np.linspace(min(df['Price']),max(df['Price']),4)
group_names=['Low','Medium','High']

#create it as a new attribute 
df['Price_binned']=pd.cut(df['Price'],bins,labels=group_names,include_lowest=True)

#ploting graphs of the bins
plt.bar(group_names,df['Price_binned'].value_counts())
plt.xlabel("Price")
plt.ylabel("count")
plt.title("price bins")
plt.show()

#creating dummy variable for screen
dummy_variable_1=pd.get_dummies(df['Screen'])
dummy_variable_1.rename(columns={'IPS Panel':'Screen-ips_Panneel','Full HD':'Screen-Full_HD'},inplace=True)
df=pd.concat([df,dummy_variable_1],axis=1)

#drop original colunm
df.drop('Screen',axis=1,inplace=True)
print(df.head())