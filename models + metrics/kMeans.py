import numpy as np
from scipy.io import loadmat
from scipy import stats as st
import sklearn.metrics
import pandas as pd
import copy

class kMeansInstance:
    def __init__(self, k = 10):
        self.num_centers = k
    
    def vote(self, cluster):
        return st.mode(self.ground_truth[cluster])
    
    def get_cluster_labels(self):
        labels = [0] * self.num_centers
        for i in range(self.num_centers):
            labels[i] = st.mode(self.ground_truth[self.label == i])[0][0]
        print(labels) 
    
    def size_of_each_cluster(self):
        cluster_counts = []
        for i in range(len(self.num_centers)):
            cluster_counts.append(np.sum(self.label == i))
        return cluster_counts


    def train(self, data, ground_truths):
        self.data = data
        self.ground_truth = ground_truths
        self.labels = np.zeros(data.shape[0]) # initalize predictions to 0s
        self.__initialize_centroids()  # get random centers
        self.get_clusters(self.centers)  
        while True:
            self.centers, points_changed = self.__update_centroids()
            if points_changed == 0: 
                break
            # print("Updating. {} points changed.".format(points_changed))

    
    def __initialize_centroids(self):
        """
        initialize 10 random data points to be 
        the first 10 centers
        """
        init_centers = np.random.randint(low=0, high=self.data.shape[0], size=self.num_centers)
        self.centers = self.data[init_centers]
      
    
    def get_clusters(self, centers):
        """
        sorts indices of training data into clusters, returns a list where
        each idx corresponds to that data entry and each value corresponds to the cluster
        """
        for i in range(len(self.data)):
            closest_center = 0
            # NOTE: updated distances to cosine distance: 1 - <u, v> / (||u|| ||v||)
            min_distance = 1 - np.dot(self.data[i], centers[0])/(np.linalg.norm(self.data[i]) * np.linalg.norm(centers[0]))
            for j in range(1, self.num_centers):
                distance = 1 - np.dot(self.data[i], centers[j])/(np.linalg.norm(self.data[i]) * np.linalg.norm(centers[j]))
                if distance < min_distance:
                    min_distance = distance
                    closest_center = j
            self.labels[i] = closest_center
    
    def __update_centroids(self):
        """ 
        returns new centers for the clusters
        and the number of points that changed clusters
        in the update
        """
        old_labels = copy.deepcopy(self.labels)
        new_centers = [0] * self.num_centers
        for i in range(self.num_centers):
            cluster = self.data[self.labels == i]
            if cluster.size > 0:
                new_centers[i] = np.average(self.data[self.labels == i], axis=0)
            else:
                # if cluster is empty, initialize a new centroid
                new_idx = np.random.randint(low=0, high=self.data.shape[0])
                new_center = self.data[new_idx]
                new_centers[i] = new_center 
        # find how many points changed
        self.get_clusters(new_centers)
        return np.array(new_centers), np.sum(old_labels != self.labels)


class kMeans(kMeansInstance):
    """ class for running kMeans multiple times and picking
    the best run, to account for randomness"""

    def __init__(self, k = 10):
        self.num_centers = k
    
    def train(self, data, ground_truths):
        best_rand_score = 0
        best_labels = 0
        for _ in range(10):
            instance = kMeansInstance(self.num_centers)
            instance.train(data, ground_truths)
            rand_score = sklearn.metrics.rand_score(ground_truths, instance.labels)
            if rand_score > best_rand_score:
                best_rand_score = rand_score
                best_labels = instance.labels
        self.labels = best_labels
        



if __name__ == "__main__":
    # get embedding labels
    data = pd.read_csv("data/politifact/politifact_all.csv")
    labels = np.array(data.label)

    # get sentence embeddings
    flat_embeddings = np.load("data/politifact/politifact_embeddings.npy")
    num_cols = int(flat_embeddings.size / len(data))
    sentence_embeddings = flat_embeddings.reshape(len(data), int(num_cols))

    # run kMeans
    model = kMeans(k=10)
    model.train(sentence_embeddings, labels)
    print(sklearn.metrics.rand_score(labels.flatten(), model.labels))
