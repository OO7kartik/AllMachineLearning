#recognizing handwritten digits using KMeans-Clustering
#note accuracy is comprimized, training set isnt very large
import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
#print(digits)
#print(digits.DESCR)
#print(digits.data)
#print(digits.target)

#to visualize we need to use matplotlib
plt.gray() 
plt.matshow(digits.images[51])
plt.show()
#check this image
#guess it,.... seems like '2' lets check
print(digits.target[51]) #yep its '2'

#since 10 digits 0,1,....9 ( k = 10 )
model = KMeans(n_clusters=10,random_state=42)
#training... (clustering it)
model.fit(digits.data)

#Visualization
fig = plt.figure(figsize=(8, 3))#figure size of 8x3
fig.suptitle('Cluster Center Images', fontsize=14,fontweight='bold')

#to Visualize our different results ( centroids ) we will use this clustering labels later on.
for i in range(10):
  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)
  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()

#this is an array that captures 4 images
# img_1 = 1 img_2 = 2 img_3 = 3 img_4 = 9
new_samples = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,4.48,3.65,0.00,0.00,0.00,0.00,0.00,3.11,7.61,7.45,0.38,0.00,0.00,0.00,0.30,7.07,7.53,7.62,1.60,0.00,0.00,0.00,2.20,7.62,4.18,7.62,2.73,0.00,0.00,0.00,1.98,5.64,1.67,7.62,4.63,0.00,0.00,0.00,0.00,0.00,0.08,6.93,5.32,0.00,0.00,0.00,0.00,0.00,0.00,6.02,4.95,0.00,0.00,0.00],
[0.00,0.00,2.96,6.99,7.62,3.04,0.00,0.00,0.00,0.00,5.03,5.77,6.01,6.86,0.00,0.00,0.00,0.00,0.00,0.00,3.34,7.62,0.00,0.00,0.00,0.00,0.00,0.00,3.04,7.61,0.00,0.00,0.00,0.00,0.00,0.45,4.94,7.38,0.00,0.00,0.00,0.22,3.19,7.30,7.61,7.23,3.80,2.74,0.00,4.87,7.62,7.62,7.61,6.86,7.08,7.55,0.00,1.67,3.04,2.29,0.83,0.00,0.15,0.60],
[0.00,0.00,4.71,7.62,7.09,7.16,7.54,1.06,0.00,0.00,6.86,5.09,0.22,3.87,7.62,1.29,0.00,0.00,1.60,0.68,2.65,7.62,6.77,0.53,0.00,0.00,0.00,0.45,7.31,7.62,7.61,5.71,0.00,0.00,0.00,0.15,3.57,1.35,4.41,6.85,0.00,0.00,0.00,0.00,0.00,0.00,5.16,6.77,0.00,0.00,0.00,1.82,0.69,2.80,7.46,4.49,0.00,0.00,0.00,7.39,7.31,7.62,6.31,0.61],
[0.00,0.61,6.77,7.61,7.61,3.58,0.00,0.00,0.00,3.12,7.62,5.09,7.62,2.97,0.00,0.00,0.00,5.56,6.48,3.04,7.61,5.10,0.00,0.00,0.00,6.85,4.79,6.70,7.61,6.09,0.00,0.00,0.00,6.85,7.46,7.62,6.62,6.09,0.00,0.00,0.22,3.26,5.33,3.34,6.77,5.40,0.00,0.00,2.97,7.55,2.58,3.79,7.60,3.34,0.00,0.00,1.37,7.15,7.61,7.61,5.98,0.53,0.00,0.00]
])

new_labels = model.predict(new_samples)
#this will give us the clusters, being unsupervised the #model cant recognise whether 1 is 'one' .So we have to manually check its centroid to check ( or assume our model did the job right cuz we programmed it like a pro :p)

#looking at the clustering labels we displayed earlier But wait, because this is a clustering algorithm, we don't know which label is which.

#By looking at the cluster centers, let's map out each of the labels with the digits we think it represents:

for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')
    




