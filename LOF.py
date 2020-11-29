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
    print('Calculating euclidean distances..')
    dist = distances(data)
    print('Done.')
    print('Calculating knn distances..')
    knn, knn_idx = knn_distance(data, dist, k)
    print('Done.')
    print('Calculating average reachability and LRD..')
    reach = avg_reachability(dist, knn, knn_idx, k) 
    LRD = 1/reach
    print('Done.')
    
    print('Calculating LOF..')
    LOF = np.zeros(len(LRD))
    for point in range(len(LRD)):
        avgLRD = 0
        for neighbor in range(k):
            avgLRD += LRD[int(knn_idx[point][neighbor])]
        
        LOF[point] = (avgLRD/k) / LRD[point]
    print('Done.')
    
    max_LOF = max(LOF)
    max_LOF_idx = np.where(LOF == max_LOF)
    
    print(max_LOF)
    print(max_LOF_idx)
    
    return data

def main():
    data = pd.read_csv("outliers-3.csv")
    outliers = local_outlier_factor(data, 2)
    #data = pd.read_csv("outliers-3 - Copy.csv")
    #outliers = local_outlier_factor(data, 22)
    
    plt.scatter(data['X1'], data['X2'], s=5, c='b')
    plt.scatter(outliers['X1'], outliers['X2'], s=5, c='r')
    plt.show()
    
if __name__ == '__main__':
    main()
    
    '''
    for i in range(len(data)):
        for j in range(i, len(data)):
            print()
            #arr[i][j] = math.sqrt(data[i]['X1']**2 + 
    '''      
    