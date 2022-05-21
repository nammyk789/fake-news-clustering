import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
import kmeans

class spectralClustering:
    def __init__(self, sigma=5, neighbors=10):
        self.sigma = sigma
        self.neighbors = neighbors
    
    def cluster_counts(self):
        cluster_counts = []
        for cluster in self.model.get_clusters():
            cluster_counts.append(len(cluster))
        return cluster_counts

    def rand_score(self):
        return self.model.rand_score()
    
    def accuracy(self):
        return self.model.accuracy()
    
    def cluster(self, data, labels):
        self.data = data

        # get weight and degree of data
        self.__create_knn_weight_matrix()
        self.degree = np.diag(self.weight.sum(axis=1))
        self.graph_Laplacian = self.degree - self.weight

        # eigenvalues and eigenvectors
        vals, vecs = np.linalg.eig(self.graph_Laplacian)

        # sort these based on the eigenvalues
        vecs = vecs[:,np.argsort(vals)]
        vals = vals[np.argsort(vals)]

        self.model = kmeans.kMeans()
        self.model.train(vecs[:,1:10], labels)


    def __create_knn_weight_matrix(self):
        distances = squareform(pdist(self.data, 'cosine'))  # using cosine similarity for weights
        self.weight = np.exp(-distances / self.sigma)
