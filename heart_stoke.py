# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 08:40:11 2022

@author: venki
"""

#################### Stoke_Rate Health industry ############
import pandas as pd
stroke = pd.read_csv("C:/Users/venki/OneDrive/Desktop/Datascience360/Stroke_Rate/healthcare-dataset-stroke-data.csv")

## Understanding data
stroke.columns
""" ['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status', 'stroke']"""
stroke.head()
stroke.shape ## (5110, 12)
stroke.info()
stroke.describe()

### check the value counts for all categorical columns
cols = ['gender', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'smoking_status', 'stroke']

for i in cols:
    print(stroke[i].value_counts())

# Drop id column(no information)
stroke.drop(['id'], axis = 1, inplace = True)
stroke.shape # (5110, 11)

## 1. Type casting
# no need yo convert

## 2. Hnadling Duplicates
# Check duplicates
stroke.duplicated().sum() ## no duplicates

## 6. Missing values
# Check null/Nan values
stroke.isna().sum()
# or
stroke.isnull().sum() ## nan values exist in 'bmi' column
##Use simple imputator to fill nan values
import numpy as np
from sklearn.impute import SimpleImputer
median_imputer = SimpleImputer(missing_values = np.nan, strategy ='median')
stroke["bmi"] = pd.DataFrame(median_imputer.fit_transform(stroke[["bmi"]]))
stroke.isna().sum()  ## no nan values

## 3. Outlier Analysis
## Check outliers
import matplotlib.pyplot as plt
import seaborn as sns
cols1 = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi','stroke']

for i in cols1:
    sns.boxplot(stroke[i]);plt.show()

# outlier columns
cols2 = ['avg_glucose_level', 'bmi']
#Out lier treatment with winsorization
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method ='iqr', # choose IQR rule boundaries or gaussian for mean and std
                   tail = 'both', # cap left, right or both tails
                   fold = 1.5,
                  # variables = ['']
                  )

for i in cols2:
    stroke[i] = winsor.fit_transform(stroke[[i]])
    
for i in cols2:
    sns.boxplot(stroke[i]);plt.show()
    
## One graph all the boxplots in horizantal
bx = sns.boxplot(data = stroke, orient = "h", palette = "Set2")

## 4. Variance
stroke.var() ## no zero variance

## 5. Discritizatin/Binning
# no need to do binning


## 7. Dummy variable creation
stroke.info()
## Label encoder (Converting categorical into numeric)
cols3 = ['gender', 'ever_married','work_type', 'Residence_type', 'smoking_status']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Instantiate the encoders
encoders = {column: le for column in cols3}

for column in cols3:
    stroke[column] = encoders[column].fit_transform(stroke[column])

stroke.info()

## 8. Transformation
# check normal distribution or not
sns.displot(stroke.stroke, kde = True) # Normally distributed

## Balanceing the data set by SMOTE 

from imblearn.over_sampling import SMOTE
sm = SMOTE()
x = stroke.drop(['stroke'], axis = 1)
y = stroke['stroke']
X, Y = sm.fit_resample(x, y)

Y['stroke'].value_counts()

# Split the data set into train(80% of the data) and test(20% of the data)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

## Feature Scling (Standardization)

from sklearn.preprocessing import StandardScaler
# scaling the variables and store it in different data
sc = StandardScaler()
cols4 = ['age','avg_glucose_level', 'bmi']
x_train[cols4] = sc.fit_transform(x_train[cols4])
x_test[cols4] = sc.fit_transform(x_test[cols4])

x_train.head()

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay,RocCurveDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_lr_pred = lr.predict(x_test)
score_lr=accuracy_score(y_test,y_lr_pred)*100
print("accuracy score: ",accuracy_score(y_train,lr.predict(x_train))*100)
print("accuracy score: ",score_lr)

print(f"Confusion Matrix :- \n {confusion_matrix(y_test,y_lr_pred)}")
print(f"Classiication Report : -\n {classification_report(y_test, y_lr_pred)}")



from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train,y_train)
from sklearn.metrics import confusion_matrix
confusion_knn=confusion_matrix(y_test,knn.predict(x_test))
plt.figure(figsize=(8,8))
sns.heatmap(confusion_knn,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
from sklearn.metrics import classification_report
print(classification_report(y_test,knn.predict(x_test)))


from sklearn.ensemble import RandomForestClassifier
rd_clf = RandomForestClassifier()
rd_clf.fit(x_train, y_train)

# accuracy score, confusion matrix and classification report of random forest

rd_clf_acc = accuracy_score(y_test, rd_clf.predict(x_test))

print(f"Training Accuracy of Decision Tree Classifier is {accuracy_score(y_train, rd_clf.predict(x_train))}")
print(f"Test Accuracy of Decision Tree Classifier is {rd_clf_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, rd_clf.predict(x_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, rd_clf.predict(x_test))}")

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)

dtc_acc = accuracy_score(y_test, dtc.predict(x_test))

print(f"Training Accuracy of Decision Tree Classifier is {accuracy_score(y_train, dtc.predict(x_train))}")
print(f"Test Accuracy of Decision Tree Classifier is {dtc_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, dtc.predict(x_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, dtc.predict(x_test))}")

from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(base_estimator = dtc)
ada.fit(x_train, y_train)

ada_acc = accuracy_score(y_test, ada.predict(x_test))

print(f"Training Accuracy of Decision Tree Classifier is {accuracy_score(y_train, ada.predict(x_train))}")
print(f"Test Accuracy of Decision Tree Classifier is {ada_acc} \n")

print(f"Confusion Matrix :- \n{confusion_matrix(y_test, ada.predict(x_test))}\n")
print(f"Classification Report :- \n {classification_report(y_test, ada.predict(x_test))}")


from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
param_grid={'C':[0.001,0.01,0.1,1,10,100], 'gamma':[0.001,0.01,0.1,1,10,100]}
rcv=RandomizedSearchCV(SVC(),param_grid,cv=5)
rcv.fit(x_train,y_train)
y_pred_svc=rcv.predict(x_test)
confusion_svc=confusion_matrix(y_test,rcv.predict(x_test))
plt.figure(figsize=(8,8))
sns.heatmap(confusion_svc,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
print(classification_report(y_test,y_pred_svc))













