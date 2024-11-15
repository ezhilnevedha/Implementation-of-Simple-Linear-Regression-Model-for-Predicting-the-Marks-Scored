# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
 
## Algorithm:

1.Import necessary libraries: pandas, numpy, matplotlib, and scikit-learn.
2.Load the dataset student_scores.csv into a DataFrame and print it to verify contents.
3.Display the first and last few rows of the DataFrame to inspect the data structure.
4.Extract the independent variable (x) and dependent variable (y) as arrays from the DataFrame.
5.Split the data into training and testing sets, with one-third used for testing and a fixed random_state for reproducibility.
6.Create and train a linear regression model using the training data.
7.Make predictions on the test data and print both the predicted and actual values for comparison.
8.Plot the training data as a scatter plot and overlay the fitted regression line to visualize the model's fit.
9.Plot the test data as a scatter plot with the regression line to show model performance on unseen data.
10.Calculate and print error metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) for evaluating model accuracy.
11.Display the plots to visually assess the regression results.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: EZHIL NEVEDHA.K
RegisterNumber:  212223230055
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
![image](https://github.com/user-attachments/assets/c62c59ae-fe64-4804-9e67-ec5f35e5b8bf)

![image](https://github.com/user-attachments/assets/838a6cd6-fc7d-494d-af40-492449afd37a)

![image](https://github.com/user-attachments/assets/5f21f657-7628-4822-b7bf-a4191051e276)

![image](https://github.com/user-attachments/assets/8c6c2d8f-bf03-4e24-acac-0ae67d64171c)

![image](https://github.com/user-attachments/assets/38c04ac6-e5b2-4727-9dd7-e674a1f2cecb)

![image](https://github.com/user-attachments/assets/ea2c69fb-710a-4ca5-8aaa-4e746869621e)

![image](https://github.com/user-attachments/assets/73fab738-730a-46d7-821d-c5fee05263bf)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
