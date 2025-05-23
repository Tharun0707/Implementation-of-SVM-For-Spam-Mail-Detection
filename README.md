# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the Program.
2. Import the necessary packages.
3. Read the given csv file and display the few contents of the data.
4. Assign the features for x and y respectively.
5. Split the x and y sets into train and test sets.
6. Convert the Alphabetical data to numeric using CountVectorizer.
7. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
8. Find the accuracy of the model.
9. End the Program.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: Tharun Sridhar
RegisterNumber: 212223230230 
```
```
import chardet
file='spam.csv'
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![SVM For Spam Mail Detection](sam.png)

**data.head():**

![image](https://github.com/user-attachments/assets/8e67f8da-09f1-4c9d-96ba-56cd14626289)

**data.info():**

![image](https://github.com/user-attachments/assets/d6b97a25-55eb-42a2-9bec-cc9c371bf435)

**data.isnull().sum():**

![image](https://github.com/user-attachments/assets/bc0a0a4d-0153-4d87-87c9-32bdf0fde543)

**Y Prediction:**

![image](https://github.com/user-attachments/assets/ae9c957d-a075-46e5-b6fd-24e6d4907971)

**Accuracy:**

![image](https://github.com/user-attachments/assets/71cbb774-dc15-43ae-ab6b-149b2d212df4)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
