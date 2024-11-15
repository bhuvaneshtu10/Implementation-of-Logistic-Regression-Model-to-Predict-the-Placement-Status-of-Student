# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: BHUVANESHWARAN TU
RegisterNumber:  24009351
*/
 import pandas as pd
 data=pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
 data.head()
 data1=data.copy()
 data1=data1.drop(["sl_no","salary"],axis=1)
 data1.head()
 data1.isnull().sum()
 data1.duplicated().sum()
 from sklearn.preprocessing import LabelEncoder
 le=LabelEncoder()
 data1["gender"]=le.fit_transform(data1["gender"])
 data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
 data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
 data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
 data1["degree_t"]=le.fit_transform(data1["degree_t"])
 data1["workex"]=le.fit_transform(data1["workex"])
 data1["specialisation"]=le.fit_transform(data1["specialisation"])
 data1["status"]=le.fit_transform(data1["status"])
 data1
 x=data1.iloc[:,:-1]
 x
 y=data1["status"]
 y
 from sklearn.model_selection import train_test_split
 x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
 from sklearn.linear_model import LogisticRegression
 lr=LogisticRegression(solver="liblinear")
 lr.fit(x_train,y_train)
 y_pred=lr.predict(x_test)
 y_pred
 from sklearn.metrics import accuracy_score
 accuracy=accuracy_score(y_test,y_pred)
 accuracy
 from sklearn.metrics import confusion_matrix
 confusion=confusion_matrix(y_test,y_pred)
 confusion
 from sklearn.metrics import classification_report
 classification_report1=classification_report(y_test,y_pred)
 print(classification_report1)
 lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
![Screenshot 2024-11-15 173648](https://github.com/user-attachments/assets/0c8fb106-3618-4f60-a2ed-a996c960bae5)
![Screenshot 2024-11-15 173706](https://github.com/user-attachments/assets/a927aa2c-7290-4c9e-87f4-40cb59305a09)
![Screenshot 2024-11-15 173826](https://github.com/user-attachments/assets/43415ecb-b1b5-458d-a09e-ef0fae38a923)
![Screenshot 2024-11-15 174021](https://github.com/user-attachments/assets/66e715ea-c487-4255-a7ee-1052f96cd47c)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
