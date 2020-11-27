import sys
import numpy as np
import pandas as pd
import math
import random
from scipy import stats
from pprint import pprint

# 1) pick random starting centroids (without replacement) 
def init_centroids(data, k):
    return data[np.random.choice(data.shape[0], k, replace=False)]

# 2) create 2d array of hamming distances, compares every centroid to every mushroom (row)
    # doesn't count first column (label) in calculation
def calc_hlist(data, k, centroids):
    hlist = np.zeros((len(data), len(centroids)), dtype=np.uint8)
    for centroid in range(len(centroids)):
        hscore = 0
        for row in range(len(data)):
            hscore = 0
            for feature in range(len(centroids[0])):
                if centroids[centroid][feature] != data[row][feature] and feature != 0:
                    hscore = hscore+1
            hlist[row][centroid] = hscore   
    return hlist
        
# 3) assigns each mushroom (row) to the closest centroid
    # input: clusters (empty dict), hlist(2d numpy array)
def assign_cluster(data, k, clusters, hlist):
    for row in range(len(hlist)):
        index = np.argmin(hlist[row])
        clusters[index].append(data[row]) 
    return clusters

# 4) update centroid with mode of every feature in respective clusters
def update_centroid(data, k, centroids, clusters):
    for i in range(len(centroids)):
        new_centroid = stats.mode(clusters[i])
        centroids[i] = new_centroid[0]
    return centroids

# output: dictionary with clusters (keys) and mushroom entries (values)
def kmodes(data, k):
    count = 1
    print('Iteration', count, '..')    
    # 1) init centroids
    centroids = init_centroids(data, k)
    old_centroids = centroids.copy()
    # 2) calculate hlist (hamming distances)
    hlist = calc_hlist(data, k, centroids)
    # 3) populate clusters
    clusters = {n: [] for n in range(k)}
    clusters = assign_cluster(data, k, clusters, hlist)
    # 4) update centroids
    centroids = update_centroid(data, k, centroids, clusters)
    new_centroids = centroids.copy()
    print('Done')
    count = count+1
    # 5) repeat 2,3,4 until centroids are the same 
    while not (np.array_equal(old_centroids, new_centroids)):
        print('Iteration', count, '..')
        old_centroids = new_centroids.copy()
        hlist = calc_hlist(data, k, centroids)
        clusters = clusters.clear()
        clusters = {n: [] for n in range(k)}
        clusters = assign_cluster(data, k, clusters, hlist)
        centroids = update_centroid(data, k, centroids, clusters)
        new_centroids = centroids.copy()
        print('Done')
        count = count+1
    print('Total iterations:', count-1)
    return clusters

# change format to more easily export csv
def dict_to_df(clusters, features):
    clusters_arr = []
    keys = list(clusters.keys())
    for cluster in range(len(clusters)):
        for entry in range(len(clusters[cluster])):
            row = clusters[cluster][entry]
            row = np.append(row, keys[cluster])
            clusters_arr.append(row)
    clusters_np = np.array(clusters_arr)
    output = pd.DataFrame(clusters_np, columns = features)
    cols = list(output.columns)
    cols = [cols[-1]] + cols[:-1]
    output = output[cols]
    return output

def main():
    features = ['class-label', 'cap-shape', 'cap-surface', 'cap-color', 'bruises?',  
                'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
                'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring',
                'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
                'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat', 'cluster']
    
    txtfile = 'agaricus-lepiota.data'
    data = np.loadtxt(txtfile, dtype = str, delimiter = ',')
    
    # impute missing data in the 'stalk-root' column
    size = np.shape(data)
    for i in range(size[0]):
        if data[i][11] == '?':
            data[i][11] = 'b'
    
    # run k-modes algorithm on data with k number of clusters
    numClusters = 20    
    print('Starting k-modes on', txtfile, 'with', numClusters, 'clusters.')
    clusters = kmodes(data, numClusters)
    
    # change format and export to csv
    out_csv = 'kmodes-output.csv'
    print('\nExporting file to', out_csv)
    output = dict_to_df(clusters, features)
    output.to_csv(out_csv, index=False)
    print('Finished')
  
    
if __name__ == '__main__':
    main()
    

    