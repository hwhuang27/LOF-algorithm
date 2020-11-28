import sys
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
sns.set()

# Calculate distances between every 2 data points
def distances(data):
    return cdist(data, data, 'euclid')

# Get kth closest neighbors of each data point
def knn_dist(data, dist, k):
    knn = np.zeros(shape=(len(data), k))
    for row in range(len(dist)):
        temp = dist[row]
        temp = np.sort(temp)
        knn[row] = temp[1:k+1]
    return knn

def avg_reachability()

# Returns outliers from input
def local_outlier_factor(data, k):
    outliers = data.copy()
    dist = distances(data)
    print(dist)
    knn = knn_dist(data, dist, k)
    print(knn)
    
    return outliers

def main():
    data = pd.read_csv("outliers-3.csv")
    outliers = local_outlier_factor(data, 3)
    
    plt.scatter(data['X1'], data['X2'], s=5, c='b')
    plt.scatter(outliers['X1'], outliers['X2'], s=5, c='r')
    plt.show()
    
if __name__ == '__main__':
    main()
    

    