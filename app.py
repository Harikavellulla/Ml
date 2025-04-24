import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"C:\Users\LENOVO\Downloads\Salary_Data.csv")

x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary vs experience(Test set)')
plt.xlabel('Years of experience')
plt.ylabel('salary')
plt.show()

m_slope = regressor.coef_
print(m_slope)
      
c_intercept = regressor.intercept_
print(c_intercept)

pred_12yr_emp_exp = m_slope * 12 + c_intercept
print(pred_12yr_emp_exp)

pred_20yr_emp_exp = m_slope * 20+ c_intercept
print(pred_20yr_emp_exp)


bias=regressor.score(x_train,y_train)
print(bias)

varience=regressor.score(x_test,y_test)
print(varience)

# stats for ml
dataset.mean()

dataset['Salary'].mean()

dataset['Salary'].median()

dataset.median()

dataset.var()
dataset['Salary'].var()

dataset.mode()
dataset['Salary'].mode()

dataset.std()
dataset['Salary'].std()

from scipy.stats import variation
variation(dataset.values)

variation(dataset['Salary'])

dataset.corr()

# ssr
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)
# sse
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

# sst
mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

#r2
r_squre = 1-SSR/SST
print(r_squre)

import pickle
# save the trained model to disk
filename = 'linear_regression_model.pkl'
with open(filename,'wb')as file:
    pickle.dump(regressor,file)
print("Model has been pickled and saved as linear_regression_model.pkl")
import os
os.getcwd()

