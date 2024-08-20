import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

import os
os.chdir(r'c:\Users\samba\Documents\Machine Learning Tutorial\MachineLearningTutorial') 

from utilities.utility_functions import *





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
data_1 = np.stack(data[target_array == 0], np.zeros(num_points))
data_2 = np.stack(data[target_array == 1], np.ones(num_points))

print(data_1)

