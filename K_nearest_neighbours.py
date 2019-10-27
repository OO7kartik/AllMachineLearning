# training data = training_set
# training labels = training_labels ( either 1 or 0 )
# validation_set
# validation_labels


# THE MANUAL METHOD	

#euclidean distance
def distance(obj1,obj2):
	squared_difference = 0
	for i in range(len(obj1)):
		squared_difference += (obj1[i] - obj2[i])**2

	return squared_difference**0.5	

#classify(unknown data,training dataset, traning labels, nearest neighbours )
def classify(unknown,dataset,labels,k):
	distances = []
	#to find the k nearest ( to find similar ojb )
	for value in dataset:
		obj = dataset[value]
		distance_to_point = distance(obj,unknown)
		#noting this value down
		distances.append([distance_to_point,value])
		#sorting the distances accending order
	distances.sort()
	#extracting only the K nearest
	neighbours = distances[0:k]

	num_good = 0
	num_bad = 0

	for neighbour in neighbours:
		value = neighbour[1]
		if(labels[value] == 1):
			num_good += 1
		elif(labels[value] == 0):
			num_bad += 1

	if num_good > num_bad:
		return 1
	else:
		return 0	

#function to check the accuracy of our model
def find_validation_accuracy(training_set, training_labels,validation_set,validation_labels,k):
  
  num_correct = 0.0
  for value in validation_set:
    guess = classify(validation_set[movie],training_set,training_labels,k)
    if(validation_labels[value] == guess):
      num_correct += 1
  return  num_correct / len(validation_set)


#low value of k -> overfitting
#high value of k -> underfitting

#checking the accuracy with k = 3
print(find_validation_accuracy(training_set, training_labels, validation_set, validation_labels, 3))


	
#USING INBUILT FUNCTION METHOD

#after extracting all our data
from sklearn.neighbours import KNeighborsClassifier

#we enter the number of neighbors, to create the KNeighborsClassifier object
classifier = KNeighborsClassifier( n_neighbors = 5 )
#traning the classifier
classifier.fit(dataset_set,dataset_set_labels)
#to predict 
classifier.predict(values_to_predict)
