import sklearn 
from sklearn import tree
import numpy as np
my_tree = tree.DecisionTreeClassifier()
data_training = np.array([
               [1,3,3,7],
               [5,2,4,6],
               [1,2,4,6],
               [5,4,4,3],
               [1,4,4,7],
               [3,2,3,7],
               [3,3,3,6],
               [5,2,2,7]])
predicted_results =[0,1,1,0,0,0,0,1]
result = my_tree.fit(data_training,predicted_results)
person1 = [[1,4,3,3]]
person2 = [[1,4,3,7]]
person3 = [[5,2,4,6]]
check = result.predict(person3)
if check == 1 :
    check = "Yes"
else:
    check = "No"
print(check)


