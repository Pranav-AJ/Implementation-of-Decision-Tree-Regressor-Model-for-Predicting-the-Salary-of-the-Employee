# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload the csv file and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree inport DecisionTreeRegressor.
5. Import metrics and calculate the Mean squared error.
6. Apply metrics to the dataset, and predict the output.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: A.J.PRANAV
RegisterNumber: 212222230107 
*/
```
```
import pandas as pd
df=pd.read_csv("/Salary.csv")
df
```

```
df.head()
```

```
df.info()
```

```
df.isnull().sum()
```
```
df['Salary'].value_counts()
```

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Position']=le.fit_transform(df['Position'])
df.head()
```
```
X=df[['Position','Level']]
Y=df[['Salary']]
```
```
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(X,Y,test_size=0.2,random_state=2)
```
```
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```
```
y_pred
```

```
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
```
```
r2=metrics.r2_score(y_test,y_pred)
r2
```
```
dt.predict([[5,6]])
```
## Output:

![image](https://github.com/Pranav-AJ/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118904526/e8364e4b-d2b1-4c26-9089-037d038259f5)

### data.head()

![image](https://github.com/Pranav-AJ/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118904526/9a1eac4a-400d-4c34-8bdb-153d6100c644)

### data.info()

![image](https://github.com/Pranav-AJ/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118904526/448472e6-dfc0-4954-b084-974c4f48dec6)

### isnull() & sum() function

![image](https://github.com/Pranav-AJ/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118904526/95bd14ac-6a5d-46c1-9d25-0e6b301742e8)

![image](https://github.com/Pranav-AJ/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118904526/2caf1b8e-c767-47da-9755-bf0f586c9771)

### data.head() for position 

![image](https://github.com/Pranav-AJ/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118904526/ce646133-730e-4980-a198-d1345340987c)

![image](https://github.com/Pranav-AJ/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118904526/4115e155-6b9e-4727-b596-088a1806cc57)

### MSE value

![image](https://github.com/Pranav-AJ/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118904526/0de5def1-8028-480c-b6e1-7081d0d1b12c)

### R2 value

![image](https://github.com/Pranav-AJ/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118904526/f3305303-32c2-4ed0-949c-43c4a61a74ed)

### Prediction value

![image](https://github.com/Pranav-AJ/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118904526/9e033278-dfc0-4ae5-a57d-9486145e22ea)

## Result:

Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
