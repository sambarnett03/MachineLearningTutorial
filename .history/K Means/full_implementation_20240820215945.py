import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

import sys
sys.path.append(r'C:\Users\samba\Documents\Machine Learning Tutorial\MachineLearningTutorial\utilities') 

from utility_functions import *





class KMeans():
    def __init__(self, data, n):
        self.data = data
        self.n = n
        self.dims = self.data.shape[1]
        
        self.centroids = np.zeros((self.n, self.dims))
        
        
        
    def _distance(self, x, y):
        return np.sqrt(np.sum([(x[i] - y[i]) ** 2 for i in range(len(x))]))
        
        
        
    def initialise(self, method='kmeans++'):
        if method == 'kmeans++':
            distances = np.zeros(self.data.shape[0])
            
            # Randomly select the first 
            ind = np.random.choice(self.data.shape[0], replace=False)
            self.centroids[0] = self.data[ind]
            
            # Select following from weighted distribution
            for i in range(self.n):
                distances += np.array([self._distance(x, self.centroids[i]) ** 2 for x in self.data])
                probabilities = distances / np.sum(distances)
                index = np.random.choice(self.data.shape[0], p=probabilities)
                self.centroids[i] = self.data[index]
                
        
        else:  # Choose at random
            ind = np.random.choice(self.data.shape[0], self.n, replace=False)
            self.centroids = [self.data[i] for i in ind]
            return self.centroids
        
        return
    
    
    
    def update_allocations(self):
        allocation = []
        for x in self.data:
            distances = [self.distance(x, centroid) for centroid in self.centroids]
            allocation.append(np.argmin(distances))
        
        self.allocation = np.array(allocation)
        return
        
    


    
num_points = 1000
data, target_array = generate_data(num_points, 2)

data = data.tolist()
[x.append(0) for x in data]
data = np.array(data)

kmeans = KMeans(data, 2)
kmeans.initialise()



# fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
# ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2)

# [ax.scatter(kmeans.centroids[i, 0], kmeans.centroids[i, 1], kmeans.centroids[i, 2], marker='x', s=200, c='b') for i in range(kmeans.n)]
# plt.show()
