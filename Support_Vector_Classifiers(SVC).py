import numpy as np
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = load_wine()

x=data.data
y=data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1)

clf=SVC(kernel="linear",C=3)
clf.fit(x_train,y_train)

clf2=KNeighborsClassifier(n_neighbors=3)
clf2.fit(x_train,y_train)

print(f'SVC:{clf.score(x_test,y_test)}')
print(f'KNN:{clf2.score(x_test,y_test)}')
