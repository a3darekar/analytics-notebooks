import pandas as pd
from sklearn.cluster import KMeans

df = pd.DataFrame({
	'x': [2, 4, 10, 12, 3, 20, 30, 11, 25]
})

kmeans = KMeans(n_clusters=2)
kmeans.fit(df)
