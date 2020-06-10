from sklearn.datasets import load_iris
iris = load_iris()


# Randomly splitting data into train and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(iris['data'],iris['target'],random_state = 42, test_size = 0.33)

# ML Algorithm
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train,y_train)

knn.score(X_test,y_test)