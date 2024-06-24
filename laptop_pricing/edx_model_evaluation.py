from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.preprocessing import PolynomialFeatures

#load dataset
filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv"
df=pd.read_csv(filepath,header=0)
df.head()

#drop unnecessary colunms
df.drop(['Unnamed: 0.1','Unnamed: 0'],axis=1,inplace=True)
df.head()

#divide the dataset
y_data=df['Price']
x_data=df.drop(['Price'],axis=1)

#split data into training and testing subset with 10% for testing
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.10,random_state=1)
print('number of test samples:',x_test.shape[0])
print('number of train samples:',x_train.shape[0])


#create LR model
#print R^2 value and cross val score
lre=LinearRegression()
lre.fit(x_train[['CPU_frequency']],y_train)
print(lre.score(x_test[['CPU_frequency']],y_test))
print(lre.score(x_train[['CPU_frequency']],y_train))
Rcross = cross_val_score(lre, x_data[['CPU_frequency']], y_data, cv=4)
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())

#spli data into training and testing reserving 50% data for testing
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.5, random_state=0)

#evaluate the R^2 scores of the model created using different degrees of polynomial features, ranging from 1 to 5.
# Save this set of values of R^2 score as a list.
le = LinearRegression()
Rsqu_test = []
order = [1, 2, 3, 4, 5]
for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train[['CPU_frequency']])
    x_test_pr = pr.fit_transform(x_test[['CPU_frequency']])    
    lre.fit(x_train_pr, y_train)
    Rsqu_test.append(lre.score(x_test_pr, y_test))


#plot the valur of R^2 score against the order 
plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.show()


#Ridge regression
#Create a polynomial feature model that uses all these parameters with degree=2. 
#also create the training and testing attribute sets.
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']])
x_test_pr=pr.fit_transform(x_test[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']])


#Create a Ridge Regression model and evaluate it using values of the hyperparameter alpha ranging from 0.001 to 1 with increments of 0.001. 
#Create a list of all Ridge Regression R^2 scores for training and testing data
Rsqu_test = []
Rsqu_train = []
Alpha = np.arange(0.001,1,0.001)
pbar = tqdm(Alpha)

for alpha in pbar:
    RigeModel = Ridge(alpha=alpha) 
    RigeModel.fit(x_train_pr, y_train)
    test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})
    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)


#Ploting the R^2 values for training and testing sets with respect to the value of alpha
plt.figure(figsize=(10, 6))  
plt.plot(Alpha, Rsqu_test, label='validation data')
plt.plot(Alpha, Rsqu_train, 'r', label='training Data')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.ylim(0, 1)
plt.legend()
plt.show()

#Grid search
parameters1=[{'alpha': [0.0001,0.001,0.01,1,10]}]

#Creating a Ridge instance and run Grid Search using a 4 fold cross validation.
RR=Ridge()
Grid1 = GridSearchCV(RR, parameters1,cv=4)
#fit the data
Grid1.fit(x_train[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']], y_train)
#R^2 score
BestRR=Grid1.best_estimator_
print(BestRR.score(x_test[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core','OS','GPU','Category']],y_test))

