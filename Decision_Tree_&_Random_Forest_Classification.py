import numpy as np
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data = load_wine()

x=data.data
y=data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

clf = SVC(kernel="linear",C=2)
clf.fit(x_train,y_train)


clf2 = KNeighborsClassifier(n_neighbors=2)
clf2.fit(x_train,y_train)

clf3 = DecisionTreeClassifier()
clf3.fit(x_train,y_train)

clf4 = RandomForestClassifier()
clf4.fit(x_train,y_train)

print(f'SVC:{clf.score(x_test,y_test)}')
print(f'KNN:{clf2.score(x_test,y_test)}')
print(f'DIC:{clf3.score(x_test,y_test)}')
print(f'RFC:{clf4.score(x_test,y_test)}')