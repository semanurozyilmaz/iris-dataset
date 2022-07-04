import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

veriler = pd.read_csv('Iris.csv')
x = veriler.iloc[:,1:5].values #bağımsız değişkenler
y = veriler.iloc[:,5:].values #bağımlı değişken

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
flower_names=["Iris-setosa","Iris-versicolor","Iris-virginica"]
values = list(veriler["Species"].value_counts())
axs[0].bar(flower_names,values)
axs[1].pie(values,labels=flower_names)
fig.suptitle('Flowers')
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle('Box-Plots')
sns.boxplot(ax=axes[0, 0], x='Species',y='SepalLengthCm', data=veriler)
sns.boxplot(ax=axes[0, 1], x='Species',y='SepalWidthCm', data=veriler)
sns.boxplot(ax=axes[1, 0], x='Species',y='PetalLengthCm', data=veriler)
sns.boxplot(ax=axes[1, 1], x='Species',y='PetalWidthCm', data=veriler)
plt.show()

Q1 = veriler.SepalLengthCm.quantile(0.25)
Q3 = veriler.SepalLengthCm.quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
veriler = veriler[(veriler.SepalLengthCm>lower_limit)&(veriler.SepalLengthCm<upper_limit)]

Q1 = veriler.SepalWidthCm.quantile(0.25)
Q3 = veriler.SepalWidthCm.quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
lower_limit, upper_limit
veriler = veriler[(veriler.SepalWidthCm>lower_limit)&(veriler.SepalWidthCm<upper_limit)]

Q1 = veriler.PetalLengthCm.quantile(0.25)
Q3 = veriler.PetalLengthCm.quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
lower_limit, upper_limit
veriler = veriler[(veriler.PetalLengthCm>lower_limit)&(veriler.PetalLengthCm<upper_limit)]

Q1 = veriler.PetalWidthCm.quantile(0.25)
Q3 = veriler.PetalWidthCm.quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5*IQR
upper_limit = Q3 + 1.5*IQR
lower_limit, upper_limit
veriler = veriler[(veriler.PetalWidthCm>lower_limit)&(veriler.PetalWidthCm<upper_limit)]

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#knn
knn = KNeighborsClassifier(n_neighbors=3, metric='minkowski')
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('KNN')
print(cm)
print(classification_report(y_test, knn.predict(X_test)))

#svc
svc = SVC(kernel='linear')
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('SVC')
print(cm)
print(classification_report(y_test, svc.predict(X_test)))

#gnb
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('GNB')
print(cm)
print(classification_report(y_test, gnb.predict(X_test)))

#dtc
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
y_pred = dtc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('DTC')
print(cm)
print(classification_report(y_test, dtc.predict(X_test)))

#rfc
rfc = RandomForestClassifier(n_estimators=15, criterion = 'entropy')
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)
cm = confusion_matrix(y_test,y_pred)
print('RFC')
print(cm)
print(classification_report(y_test, rfc.predict(X_test)))
