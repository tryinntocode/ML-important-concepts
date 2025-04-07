import numpy as np
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = load_wine()
print(data.DESCR)
print(data.target_names)

x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.4,random_state=42)

classifiers = KNeighborsClassifier(n_neighbors=5)
classifiers.fit(x_train,y_train)

print(classifiers.score(x_test,y_test))


# import numpy as np
# from sklearn.datasets import load_wine
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import train_test_split
#
# # Load dataset correctly
# data = load_wine()
#
# # Print feature names and target names
# print(data.feature_names)
# print(data.target_names)
#
# # Splitting data correctly
# X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.4, random_state=42)
#
# # Correct parameter name: `n_neighbors`
# classifier = KNeighborsClassifier(n_neighbors=5)
# classifier.fit(X_train, y_train)
#
# # Print the accuracy score
# print(classifier.score(X_test, y_test))
