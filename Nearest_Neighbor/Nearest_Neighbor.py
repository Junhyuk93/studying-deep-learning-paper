import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        """ X is N x D where each row is an sample. Y is l-dimension of size N """
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """ X is N x D where each row is an example we wish predict label for """
        num_test = X.shape[0]

        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis=1)
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]

        return Ypred