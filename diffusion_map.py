from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
import numpy as np

class DiffusionMap():
    def __init__(self, dim=10, sigma=0.5):
        self.dim = dim
        self.sigma = sigma

    def map(self, data):
        self.data = data
        self.get_matrix()
        print("matrix gotten")
        
        # get eigenvalues and vectors of S
        w, V = np.linalg.eigh(self.matrix)

        # get top k eigenvectors based on values
        idx = w.argsort()[::-1]
        V = V[:,idx]

        # PHI = D^-1/2 V
        self.map = np.matmul(self.__degree_matrix_to_the_power_of(-1/2), V)
        self.map = self.map[:, :self.dim]

        return self.map

    def get_matrix(self):
        """ calculates the diffusion matrix """
        # create W
        self.__create_gaussian_weight_matrix()
        
        # M = D^-1 W
        M = np.matmul(self.__degree_matrix_to_the_power_of(-1), self.weight)

        # S = D^1/2 M D^-1/2
        m_d_half = np.matmul(M, self.__degree_matrix_to_the_power_of(-1/2))
        self.matrix = np.matmul(self.__degree_matrix_to_the_power_of(1/2), m_d_half)

    def __degree_matrix_to_the_power_of(self, exponent):
        # get list of degrees
        self.degree = np.sum(self.weight, axis=1)
        return np.diag(self.degree ** exponent)


    def __create_gaussian_weight_matrix(self):
        """
        creates a weight matrix using gaussian kernel
        """
        distances = squareform(pdist(self.data, 'cosine'))  # using cosine similarity for weights
        self.weight = np.exp(-distances / self.sigma)


if __name__ == "__main__":
    flat_embeddings = np.load("data/flattened_embeddings.npy")
    num_cols = int(flat_embeddings.size / 11507)
    sentence_embeddings = flat_embeddings.reshape(11507, int(num_cols))
    mapper = DiffusionMap()
    diff_map = mapper.map(sentence_embeddings)