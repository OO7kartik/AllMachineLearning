#MANUAL METHOD
#we can use the dataset that sklearn has " iris "
#NOTE : ( iris is our dataset)
	# for discription of dataset -> print(iris.DESCR)
	# for labels of dataset -> print(iris.target)
	# for the flowers -> print(iris.data) , each line a flower with its features

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

x = samples[:,0] 
y = samples[:,1] 

# Number of clusters
k = 3
# Create x coordinates of k random centroids
centroids_x = np.random.uniform(min(y),max(y),size = k)
# Create y coordinates of k random centroids
centroids_y = np.random.uniform(min(y), max(y), size=k)
# Create centroids array
centroids = np.array(list(zip(centroids_x,centroids_y)))
sepal_length_width = np.array(list(zip(x, y))) # we will need this
# a scatter plot of x, y
#plt.scatter(x,y)

# a scatter plot of the centroids
#plt.scatter(centroids_x,centroids_y)

# Display which plot you wish to see
#plt.show()


#Assign samples to nearest centroid
# create our euclidean distance function
def distance(a, b):
  one = (a[0] - b[0]) **2
  two = (a[1] - b[1]) **2
  distance = (one+two) ** 0.5
  return distance

# labels filled with zeros, we are going to fill this with 0,1,2 based on the least ditances from centroids
labels = np.zeros(len(samples))

# Distances to each centroid
distances = np.zeros(k)

# Initialize error:
error = np.zeros(3)
for p in range(k):
  error[p] = distance(centroids[p], centroids_old[p])

# this loop loops till the centroids dont change , ie they are at the optimal position
while(error.all() != 0):  

  for i in range(len(samples)):
    distances[0] = distance(sepal_length_width[i], centroids[0]) #distance of centroid_0 from current point
    distances[1] = distance(sepal_length_width[i], centroids[1]) #distance of centroid_1 from current point
    distances[2] = distance(sepal_length_width[i], centroids[2]) #distance of centroid_2 from current point
    cluster = np.argmin(distances) # from these 3 (0,1,2) pick the one with least distance from our point ( we obtain the index )
    labels[i] = cluster # store the index here ie either 0,1,2

    # Step 3: Update centroids
    centroids_old = deepcopy(centroids) 

	#for loop to average the points which gave 0,1,2 resp and set the new centroid_0,centroid_1,centroid_2 resp
	#note centroid_0 will store the average value of all points which gave 0 in labels
	for j in range(k):
 	 points = [sepal_length_width[i] for i in range(len(labels)) if labels[i]==j]
 	 centroids[j] = np.mean(points,axis = 0)
  
	print(centroids_old,centroids)

	#update the error

	for p in range(k):
  error[p] = distance(centroids[p], centroids_old[p])


  #For visualization to see what has happened
  colors = ['r', 'g', 'b']
#for loop that shows the 3 different regions in different colors
for i in range(k):
  points = np.array([sepal_length_width[j] for j in range(len(samples)) if labels[j] == i])
  plt.scatter(points[:, 0], points[:, 1], c=colors[i], alpha=0.5)
  
#to display the centroids clearly  
plt.scatter(centroids[:, 0], centroids[:, 1], marker='D', s=150) 
plt.xlabel('sepal length (cm)') 
plt.ylabel('petal length (cm)') 
plt.show()
  



#THE INBUILT WAY, saves time and is efficient

import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn import datasets
#import this for inbuilt functions
from sklearn.cluster import KMeans

iris = datasets.load_iris()

samples = iris.data

# creating a object of KMeans and using it to make 3 cluster
k = 3
model = KMeans(n_clusters = k)
# now fit the data so the model can arrange it into 3 clusters
model.fit(samples)
# predicting on our training set
labels = model.predict(samples)
# Printing to verify
print(labels)

#for visualization
species = np.chararray(target.shape, itemsize=150) 
for i in range(len(samples)): 
  if target[i] == 0: 
    species[i] = 'setosa' 
  elif target[i] == 1:
    species[i] = 'versicolor'
  elif target[i] == 2: 
    species[i] = 'virginica'
    
df = pd.DataFrame({'labels': labels, 'species': species}) 
print(df)    

ct = pd.crosstab(df['labels'], df['species'])
print(ct)  
