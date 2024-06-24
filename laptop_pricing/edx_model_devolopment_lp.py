import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,r2_score
import warnings
warnings.filterwarnings('ignore',category=UserWarning)

#load dataset
filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv"
df=pd.read_csv(filepath)
print('The forst five rows of the dataframe')
df.head(5)

#single linear regression
lm=LinearRegression()

X=df[['CPU_frequency']]
Y=df['Price']
lm.fit(X,Y)
Yhat=lm.predict(X)

#Generate the Distribution plot for the predicted values and that of the actual values
ax1 = sns.distplot(df['Price'], hist=False, color="r", label="Actual Value")
sns.distplot(Yhat, hist=False, color="b", label="Fitted Values" , ax=ax1)
plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of laptops')
plt.legend(['Actual Value', 'Predicted Value'])
plt.show()

#Evaluate the Mean Squared Error and R^2 score values for the model.
mse_slr=mean_squared_error(df['Price'],Yhat)
r2_score_slr=lm.score(X,Y)
print('The R-square for Linear regression is:', r2_score_slr)
print('The Mean square error of price and predictive value is:', mse_slr)

#multiple linear regression
lm1=LinearRegression()
Z=df[['CPU_frequency','RAM_GB','Storage_GB_SSD','CPU_core','OS','GPU','Category']]
lm1.fit(Z,Y)
Y_hat=lm1.predict(Z)

#Ploting the Distribution graph of the predicted values as well as the Actual values
ax1=sns.distplot(df['Price'],hist=False,color='r',label='Actual Value')
sns.distplot(Y_hat,hist=False,color='b',label='Fitted Values',ax=ax1)
plt.title('Actual vs Fitted value for price')
plt.xlabel('Price')
plt.ylabel('proportion of laptops')
plt.show()

#R^2 score and the MSE value for this fit
ax2 = sns.distplot(df['Price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax2)

plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price')
plt.ylabel('Proportion of laptops')

#polynomial regression
X = X.to_numpy().flatten()
f1 = np.polyfit(X, Y, 1)
p1 = np.poly1d(f1)

f3 = np.polyfit(X, Y, 3)
p3 = np.poly1d(f3)

f5 = np.polyfit(X, Y, 5)
p5 = np.poly1d(f5)


#Ploting the regression output against the actual data points
def plotpolly(model,independent_variable,dependent_variable,name):
    x_new=np.linspace(independent_variable.min(),dependent_variable.max(),100)
    y_new=model(x_new)
    plt.plot(independent_variable,dependent_variable,'.',x_new,y_new,'-')
    plt.title(f'polynomial Fit for price~{name}')
    ax=plt.gcf()
    plt.xlabel(name)
    plt.ylabel('price of laptops')
    plt.show()


#Call this function for the 3 models created and get the required graphs
plotpolly(p1,X,Y,'CPU_frequency')
plotpolly(p3,X,Y,'CPU_frequency')
plotpolly(p5,X,Y,'CPU_frequency')

#calculate the R^2 and MSE values for these fits
r_squared_1=r2_score(Y,p1(X))
print('The R-square value for 1st dergee polynomial is :',r_squared_1)
print('The MSE value for 1st degree polynomial is :',mean_squared_error(Y,p1(X)))
r_squared_3=r2_score(Y,p3(X))
print('The R-square value for 3rd dergee polynomial is :',r_squared_3)
print('The MSE value for 3rd degree polynomial is :',mean_squared_error(Y,p3(X)))
r_squared_5=r2_score(Y,p5(X))
print('The R-square value for 5th dergee polynomial is :',r_squared_5)
print('The MSE value for 5th degree polynomial is :',mean_squared_error(Y,p5(X)))

#pipeline
#Create a pipeline that performs parameter scaling, Polynomial Feature generation and Linear regression
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
Z = Z.astype(float)
pipe.fit(Z,Y)
ypipe=pipe.predict(Z)

#Evaluate the MSE and R^2 values for the this predicted output
print('MSE for multi-variable polynomial pipeline is: ', mean_squared_error(Y, ypipe))
print('R^2 for multi-variable polynomial pipeline is: ', r2_score(Y, ypipe))

#the values of R^2 increase as we go from Single Linear Regression to Multiple Linear Regression. 
#Further, if we go for multiple linear regression extended with polynomial features, we get an even better R^2 value.