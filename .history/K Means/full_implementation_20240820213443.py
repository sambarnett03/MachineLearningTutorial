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
        return np.sqrt(np.sum([(x[i] - y[i]) ** 2 for i in len(x)]))
        
    def initialise(method='kmeans++'):
        
        return
    


    
num_points = 1000
data, target_array = generate_data(num_points, 2)
data = data.tolist()
[x.append(0) for x in data]
data = np.array(data)
print(data)

fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'equal'})
ax.scatter(data[:, 0], data[:, 1], data[:, 2])
plt.show()
