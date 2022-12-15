import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

iris_dataset = load_iris()
x_train,x_test,y_train,y_test = train_test_split(iris_dataset.data,iris_dataset.target,random_state=0)
model = DecisionTreeClassifier()
test = model.fit(x_train,y_train)
x_new = np.array([[6.0 , 3.1 , 4.6 , 2.0]])
print(test.predict(x_new))#predict result
print("score: " +str((test.score(x_test,y_test))*100),end=" %")