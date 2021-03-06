import sys
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Calculate euclidean distances between every 2 data points
def distances(data):
    dist = np.zeros(shape = (len(data), len(data)))
    coords = data.to_numpy()
    for i in range(len(data)):
        for j in range(len(data)):
            dist[i][j] = math.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
    return dist

# Get kth closest neighbors (and their indices) of each data point
def knn_distance(data, dist, k):
    knn = np.zeros(shape = (len(data), k))
    knn_index = np.zeros(shape = (len(data), k))
    for row in range(len(dist)):
        temp = dist[row]
        temp2 = np.sort(temp)
        temp3 = np.argsort(temp)
        knn[row] = temp2[1:k+1]
        knn_index[row] = temp3[1:k+1]
    return knn, knn_index

# Calculate average reachability for each point
def avg_reachability(dist, knn, knn_idx, k):
    reach = np.zeros(len(dist))
    for i in range(len(knn)):
        avgRD = 0
        for j in range(k):
                eucl_dist = knn[i][j]
                knn_dist = knn[int(knn_idx[i][j])][k-1]
                avgRD += max(knn_dist, eucl_dist)
        reach[i] = avgRD / k
    return reach

# Returns outliers from input
def local_outlier_factor(data, k):  
    dist = distances(data)
    knn, knn_idx = knn_distance(data, dist, k)
    reach = avg_reachability(dist, knn, knn_idx, k) 
    LRD = 1/reach
    LOF = np.zeros(len(LRD))
    
    for point in range(len(LRD)):
        avgLRD = 0
        for neighbor in range(k):
            avgLRD += LRD[int(knn_idx[point][neighbor])]
        
        LOF[point] = (avgLRD/k) / LRD[point]
        
    # check the maximum LOF to tune hyperparameter k    
    #print('max LOF value: ', max(LOF))
    #print('max LOF index: ', np.where(LOF == max(LOF))[0])
    
    # filter outliers from original data
    dnp = data.to_numpy()
    outliers_idx = np.where(LOF > 1.95)[0]
    outliers = []
    for i in range(len(outliers_idx)):
        index = outliers_idx[i]
        outliers.append(list(dnp[index]))
        
    return pd.DataFrame(outliers, columns=['X1', 'X2'])

def main():
    data = pd.read_csv("outliers-3.csv")
    outliers = local_outlier_factor(data, 22)
    plt.scatter(data['X1'], data['X2'], s=8, c='b')
    plt.scatter(outliers['X1'], outliers['X2'], s=8, c='r')
    plt.savefig('outliers-3-output.png')
    
if __name__ == '__main__':
    main()
    