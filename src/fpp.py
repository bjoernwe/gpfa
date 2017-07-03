import itertools
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import scipy.spatial.distance

import mdp


class RandomProjection(mdp.Node):

    def __init__(self, output_dim, input_dim=None, dtype=None):
        super(RandomProjection, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        return
    
    def _train(self, x):
        return

    def _stop_training(self):
        D = self.input_dim
        A = np.random.random((D, D))
        A = A + A.T
        _, self.U = scipy.sparse.linalg.eigsh(A, k=self.output_dim)
        return

    def _execute(self, x):
        return x.dot(self.U)


class LPP(mdp.Node):

    def __init__(self, output_dim, k=10, input_dim=None, dtype=None):
        super(LPP, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.k = k
        self.L = None
        self.D = None
        return
    
    def _train(self, x):

        # number of samples
        N, _ = x.shape
        
        # initialize weight matrix W
        W = scipy.sparse.dok_matrix((N, N))
    
        # pairwise distances of data points
        distances = scipy.spatial.distance.pdist(x)
        distances = scipy.spatial.distance.squareform(distances)
        neighbors = [np.argsort(distances[i])[:self.k+1] for i in range(N)]
        
        # neighbor graph
        for s in range(N):
            for t in neighbors[s]:
                W[s,t] = 1
                W[t,s] = 1

        # graph Laplacian
        d = W.sum(axis=1).T
        D = scipy.sparse.dia_matrix((d, 0), shape=(N, N))
        L = D - W

        # projected graph laplacian
        D2 = x.T.dot(D.dot(x))
        L2 = x.T.dot(L.dot(x))

        # add chunk result to global result
        if self.L is None:
            self.L = L2
            self.D = D2
        else:
            self.L += L2
            self.D += D2

        return

    def _stop_training(self):
        self.E, self.U = scipy.linalg.eigh(self.L, b=self.D, eigvals=(0, self.output_dim-1))
    
        # normalize directions
        mask = self.U[0,:] > 0
        self.U = self.U * mask - self.U * ~mask
        return

    def _execute(self, x):
        return x.dot(self.U)
