import sys
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Calculate euclidean distances between every 2 data points
def distances(data):
    dist = data.apply(lambda row: [np.linalg.norm(row.values - data.loc[[elem], :].values, 2) 
                                   for elem in data.index.values], axis=1)
    dist = pd.DataFrame(
        data=dist.values.tolist(),
        columns=data.index.tolist(),
        index=data.index.tolist())
    return dist.to_numpy()

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
    outliers = data.copy()
    dist = distances(data)
    print(dist)
    knn, knn_idx = knn_distance(data, dist, k)
    print(knn)
    print(knn_idx)
    reach = avg_reachability(dist, knn, knn_idx, k)
    print(reach)
    LRD = 1/reach
    print(LRD)
    
    return outliers

def main():
    data = pd.read_csv("outliers-3.csv")
    outliers = local_outlier_factor(data, 3)
    
    
    
    #plt.scatter(data['X1'], data['X2'], s=5, c='b')
    #plt.scatter(outliers['X1'], outliers['X2'], s=5, c='r')
    #plt.show()
    
if __name__ == '__main__':
    main()
    
    '''
    for i in range(len(data)):
        for j in range(i, len(data)):
            print()
            #arr[i][j] = math.sqrt(data[i]['X1']**2 + 
    '''      
    