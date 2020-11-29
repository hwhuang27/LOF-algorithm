import sys
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Calculate distances between every 2 data points
def distances(data):
    dist = data.apply(lambda row: [np.linalg.norm(row.values - data.loc[[elem], :].values, 2) 
                                   for elem in data.index.values], axis=1)
    dist = pd.DataFrame(
        data=dist.values.tolist(),
        columns=data.index.tolist(),
        index=data.index.tolist())
    return dist.to_numpy()

# Get kth closest neighbors of each data point
def knn_dist(data, dist, k):
    knn = np.zeros(shape=(len(data), k))
    for row in range(len(dist)):
        temp = dist[row]
        temp = np.sort(temp)
        knn[row] = temp[1:k+1]
    return knn

# Calculate average reachability for each point
def avg_reachability(data, knn, k):
    reach = np.zeros(len(data))
    for i in range(len(reach)):
        avgRD = (1/k)
    return reach

# Returns outliers from input
def local_outlier_factor(data, k):
    outliers = data.copy()
    dist = distances(data)
    knn = knn_dist(data, dist, k)
    print(knn)
    reach = avg_reachability(data, knn, k)
    #print(reach)
    
    return outliers

def main():
    data = pd.read_csv("outliers-3.csv")
    outliers = local_outlier_factor(data, 3)
    
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
    