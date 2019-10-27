#sklearn has the iris data inbuilt
#importing it
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#making an object of this
iris  = load_iris()

'''

print(iris) #this prints all info
print(iris.data) #this prints only the data samples(rows (150)) features(columns(4))
print(iris.feature_naames) #prints feature names
print(iris.target) #the value of the excepted prediction , cuz sklearn expects only numbers
print(iris.target_names)# what each value represents


'''

X = iris.data #store the dataset
y = iris.target #store the labels  (since labels present supervised learning)

X_train, X_test ,y_train ,y_test = train_test_split(X,y,test_size = 0.8,random_state = 20)
accuracy = [] # sum of squared errors

for i in range(1,11):
    #using KNN
    #creating its object
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train ,y_train) #training

    y_pred = knn.predict(X_test)

    # to check our models accuracy
    accuracy.append(metrics.accuracy_score(y_pred,y_test))

plt.figure()
plt.plot([i for i in range(1,11)],accuracy)
plt.xlabel("k")
plt.ylabel("accuracy")
plt.show()

#from this if we plot the prediction on the training data,
# we get k=1 most efficient,but k should not be one 
#this means that the model is just remembering the training data
#using the test set as a cross-validation set we find k=3 to be most effective
