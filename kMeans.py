import numpy as np
from scipy.io import loadmat
from scipy import stats as st
import sklearn.metrics
import pandas as pd

class kMeans:
    def __init__(self, k = 10, threshold=0, iterations=100):
        self.threshold = threshold
        self.num_centers = k
        self.iterations = iterations
    
    def vote(self, cluster):
        return st.mode(self.labels[cluster])
    
    def get_cluster_labels(self):
        labels = [0] * self.num_centers
        for i in range(self.num_centers):
            labels[i] = st.mode(self.labels[self.clusters[i]])[0][0][0]
        print(labels) 
    
    def size_of_each_cluster(self):
        cluster_counts = []
        for cluster in self.clusters:
            cluster_counts.append(len(cluster))
        return cluster_counts

    def rand_score(self):
        label_pred = [0] * len(self.labels)
        for cluster in self.clusters:
            cluster_vote = np.nan
            if len(cluster) > 0:
                cluster_vote = st.mode(self.labels[cluster])[0][0][0]
            for idx in cluster:
                label_pred[idx] = cluster_vote
        return sklearn.metrics.rand_score(self.labels.flatten(), np.array(label_pred))

    def accuracy(self):
        accuracy = 0
        for cluster in self.clusters:
            cluster_labels = self.labels[cluster]
            # find what percent of the labels in the cluster are the vote
            label_homogeneity = (cluster_labels == st.mode(cluster_labels)).sum() / len(cluster_labels)
            accuracy += label_homogeneity / self.num_centers
        return accuracy

    def train(self, data, labels):
        self.data = data
        self.labels = labels
        self.__initialize_centroids()
        while True:
            self.centers, points_changed = self.__update_centroids()
            if points_changed <= self.threshold: 
                break
            # print("Updating. {} points changed.".format(points_changed))
        self.clusters = self.get_clusters(self.centers)

    
    def __initialize_centroids(self):
        """
        initialize 10 random data points to be 
        the first 10 centers
        """
        init_centers = np.random.randint(low=0, high=self.data.shape[0], size=self.num_centers)
        self.centers = self.data[init_centers]
      
    
    def get_clusters(self, centers):
        """
        sorts indices of training data into clusters, returns a 2D array of clusters
        """
        clusters = []
        for _ in range(self.num_centers):
            clusters.append([])  # initialize list of clusters
        for i in range(len(self.data)):
            closest_center = 0
            # NOTE: updated to cosine distance
            min_distance = 1 - np.dot(self.data[i], centers[0])/(np.linalg.norm(self.data[i]) * np.linalg.norm(centers[0]))
            for j in range(1, self.num_centers):
                # NOTE: updated to cosine distance
                distance = 1 - np.dot(self.data[i], centers[j])/(np.linalg.norm(self.data[i]) * np.linalg.norm(centers[j]))
                if distance < min_distance:
                    min_distance = distance
                    closest_center = j
            clusters[closest_center].append(i) 
        clusters = np.array([np.array(cluster) for cluster in clusters], dtype='object')  # convert to numpy array
        return clusters 
    
    def __update_centroids(self):
        """ 
        returns new centers for the clusters
        and the number of points that changed clusters
        in the update
        """
        clusters = self.get_clusters(self.centers)
        new_centers = [0] * self.num_centers
        for i in range(self.num_centers):
            if len(clusters[i] > 0):
                cluster = self.data[clusters[i]]
            else:
                cluster = []
            new_centers[i] = np.average(cluster, axis=0)
        # find how many points changed
        new_clusters = self.get_clusters(new_centers)
        points_changed = 0
        for cluster in range(len(new_clusters)):
            for point in new_clusters[cluster]:
                if point not in clusters[cluster]:
                    points_changed += 1
        return np.array(new_centers), points_changed
    
    def __is_outside_of_threshold(self, new_centers):
        # check that update change is greater than threshold
        delta = np.subtract(new_centers, self.centers)
        if np.average(delta) > self.threshold:
            return True
        return False


if __name__ == "__main__":
    # get sentence embeddings
    flat_embeddings = np.load("data/flattened_embeddings.npy")
    num_cols = int(flat_embeddings.size / 11507)
    sentence_embeddings = flat_embeddings.reshape(11507, int(num_cols))

    # get embedding labels
    data = pd.read_csv("data/liar_embedded.csv")
    labels = np.array(data.label)

    # run kMeans
    model = kMeans(k=6)
    model.train(sentence_embeddings, labels)
    print(model.rand_score())
