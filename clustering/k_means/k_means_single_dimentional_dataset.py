# Implementation of K means algorithm for single dimentional array

import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt

dataset = pd.DataFrame({'x': [2, 4, 10, 12, 3, 20, 30, 11, 25]})
np.random.seed(200)
k = 2
centroids = {
	i + 1: [np.random.randint(0, 80)]
	for i in range(k)
}
old_centroids = []
colmap = {1: 'r', 2: 'g', 3: 'b'}


def assign(ds, centroids):
	for i in centroids.keys():
		ds['distance_from_{}'.format(i)] = \
			np.sqrt((ds['x'] - centroids[i]) ** 2)
		centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
		ds['cluster'] = ds.loc[:, centroid_distance_cols].idxmin(axis=1)
		ds['cluster'] = ds['cluster'].map(lambda x: int(x.lstrip('distance_from_')))
		ds['y'] = ds['cluster'].map(lambda x: centroids[x][0])
		dataset['color'] = ds['cluster'].map(lambda x: colmap[x])
	return ds


def update_centroids(centroids):
	old_centroids.append(copy.deepcopy(centroids))
	for i in centroids.keys():
		centroids[i][0] = np.mean(
			dataset[dataset['cluster'] == i]['x']
		)
	return centroids


dataset = assign(dataset, centroids)
centroids = update_centroids(centroids)

while True:
	cluster_centroids = dataset['cluster'].copy(deep=True)
	centroids = update_centroids(centroids)
	dataset = assign(dataset, centroids)
	if cluster_centroids.equals(dataset['cluster']):
		break

print(dataset[['x', 'cluster', 'color', 'y']])
#
# plt.plot(dataset['x'])
# plt.show()

plt.scatter(dataset['x'], dataset['y'], color=dataset['color'], alpha=0.5, edgecolor='k')
# for idx, centroid in enumerate(centroids):
# 	plt.scatter(centroids[centroid], centroids[centroid], color=colmap[idx + 1])
plt.xlim(0, 40)
plt.ylim(0, 30)
plt.show()
