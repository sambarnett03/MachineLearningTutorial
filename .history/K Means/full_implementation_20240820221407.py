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
            
        
        
    def auto_run(self):
        # Initialise clusters (creates self.centroids attribute)
        centroids = self.initialise()
        
        # Iteratively update allocations and centroids until convergence
        allocations_old = []
        while np.array_equal(allocations, allocations_old) == False:
            allocations_old = allocations
            
            allocations = update_allocations(centroids)
            centroids = update_centroids(allocations)
    
    
        self.plot(centroids, allocations)
            
        return allocations
            
        
    
    
    
    def update_allocations(self, centroids):
        allocations = []
        # Loop through each data point and check distance to each centroid. 
        for x in self.data:
            distances = [self._distance(x, centroid) for centroid in centroids]
            allocations.append(np.argmin(distances))
        
        return np.array(allocations)
    
    
    
    
    def update_centroids(self, allocations):
        clusters = [self.data[allocations == i] for i in range(self.n)]
        centroids = [np.mean(clusters[i], axis=0) for i in range(self.n)]
        return np.array(centroids)
    
    
    

    def initialise(self, method='kmeans++'):
        centroids = np.zeros((self.n, self.dims))
        
        if method == 'kmeans++':
            distances = np.zeros(self.data.shape[0])
            
            # Randomly select the first 
            ind = np.random.choice(self.data.shape[0], replace=False)
            centroids[0] = self.data[ind]
            
            # Select following from weighted distribution
            for i in range(self.n):
                distances += np.array([self._distance(x, centroids[i]) ** 2 for x in self.data])
                probabilities = distances / np.sum(distances)
                index = np.random.choice(self.data.shape[0], p=probabilities)
                centroids[i] = self.data[index]
                
                
        
        else:  # Choose at random
            ind = np.random.choice(self.data.shape[0], self.n, replace=False)
            centroids = [self.data[i] for i in ind]
            
        return centroids
            
    
    def plot(self, centroids, allocations):
        fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
        ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2], s=2)

        [ax.scatter(centroids[i, 0], centroids[i, 1], centroids[i, 2], marker='x', s=200, c='b') for i in range(n)]
        plt.show()
    
    
        
        
        
    def _distance(self, x, y):
        return np.sqrt(np.sum([(x[i] - y[i]) ** 2 for i in range(len(x))]))


    
num_points = 1000
data, target_array = generate_data(num_points, 2)

data = data.tolist()
[x.append(0) for x in data]
data = np.array(data)

kmeans = KMeans(data, 2)
kmeans.auto_run()




