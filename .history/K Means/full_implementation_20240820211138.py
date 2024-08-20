import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm


class KMeans():
    def __init__(self, data, n):
        self.data = data
        self.n = n
        self.dims = self.data.shape[1]
        
        self.centroids = np.zeros((self.n, self.dims))
        
        
    def _distance(self, x, y):
        dist = [(x[i] - y[i]) ** 2 for i in len(x)]
        
    def initialise(method='kmeans++'):
        
        return
    
    
    
    
data = [[1, 2, 3], [0, 1, 1]]
x = data[0]
y = data[1]

dist = [(x[i] - y[i]) ** 2 for i in range(len(x))]
print(dist)
    
