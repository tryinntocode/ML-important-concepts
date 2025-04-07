from sklearn.datasets import load_wine
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

# Load the Wine dataset
wine = load_wine()
X, y = wine.data, wine.target

# Scale the features
X_scaled = scale(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# Create and train the KMeans model
kmeans = KMeans(n_clusters=3, init='random', n_init=10, random_state=42)
kmeans.fit(X_train)

# Print cluster centers and labels
print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Predicted Labels:\n", kmeans.labels_)
