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



class gPFA(mdp.Node):

    def __init__(self, output_dim, k=5, iterations=1, iteration_dim=None,
                 variance_graph=False, neighborhood_graph=True, 
                 constraint_optimization=True, input_dim=None, dtype=None):
        super(gPFA, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.k = k
        self.iterations = iterations
        self.variance_graph = variance_graph
        self.neighborhood_graph = neighborhood_graph
        self.iteration_dim = iteration_dim
        self.constraint_optimization = constraint_optimization
        self.L = None
        self.D = None
        return
    
    
    def _train(self, x):

        # number of samples
        N, _ = x.shape
        
        # from y we calculate the euclidean distances
        # after the first iteration it contains the projected data
        y = x
        
        # run algorithm several times e
        for l in range(self.iterations):

            # initialize weight matrix W
            W = scipy.sparse.dok_matrix((N, N))
        
            # pairwise distances of data points
            distances = scipy.spatial.distance.pdist(y)
            distances = scipy.spatial.distance.squareform(distances)
            neighbors = [np.argsort(distances[i])[:self.k+1] for i in range(N)]
            
            # future-preserving graph
            if self.variance_graph:
                for t in range(N-1):
                    for (i,j) in itertools.permutations(neighbors[t], 2):
                        if i+1 < N and j+1 < N:
                            W[i+1,j+1] += 1
            else:
                for s in range(N-1):
                    for t in neighbors[s]:
                        if t+1 < N:
                            W[s+1,t+1] = 1
                            W[t+1,s+1] = 1

            # neighborhood graph
            if self.neighborhood_graph:
                if self.variance_graph:
                    for t in range(N):
                        for (i,j) in itertools.permutations(neighbors[t], 2):
                            W[i,j] += 1
                else:
                    for s in range(N):
                        for t in neighbors[s]:
                            W[s,t] = 1
                            W[t,s] = 1
    
            # graph Laplacian
            d = W.sum(axis=1).T
            #d[d==0] = float('inf') 
            D = scipy.sparse.dia_matrix((d, 0), shape=(N, N))
            L = D - W
    
            # projected graph laplacian
            D2 = x.T.dot(D.dot(x))
            L2 = x.T.dot(L.dot(x))

            # (if not the last iteration:) solve and project
            if l < self.iterations-1:
                if self.constraint_optimization:
                    if type(self.iteration_dim) == int:
                        _, U = scipy.linalg.eigh(L2, b=D2, eigvals=(0, self.output_dim-1))
                    else:
                        _, U = scipy.linalg.eigh(L2, b=D2)
                    # normalize eigenvectors 
                    for i in range(U.shape[1]):
                        U[:,i] /= np.linalg.norm(U[:,i])
                else:
                    if type(self.iteration_dim) == int:
                        _, U = scipy.linalg.eigh(L2, eigvals=(0, self.output_dim-1))
                    else:
                        _, U = scipy.linalg.eigh(L2)
                y = x.dot(U)

        # add chunk result to global result
        if self.L is None:
            self.L = L2
            self.D = D2
        else:
            self.L += L2
            self.D += D2

        return


    def _stop_training(self):
        if self.constraint_optimization:
            if self.input_dim == self.output_dim:
                self.E, self.U = scipy.linalg.eigh(self.L, b=self.D)
            else:
                self.E, self.U = scipy.linalg.eigh(self.L, b=self.D, eigvals=(0, self.output_dim-1))
            # normalize eigenvectors 
            for i in range(self.U.shape[1]):
                self.U[:,i] /= np.linalg.norm(self.U[:,i])
        else:
            if self.input_dim == self.output_dim:
                self.E, self.U = scipy.linalg.eigh(self.L)
            else:
                self.E, self.U = scipy.linalg.eigh(self.L, eigvals=(0, self.output_dim-1))
    
        # normalize directions
        mask = self.U[0,:] > 0
        self.U = self.U * mask - self.U * ~mask
        return


    def _execute(self, x):
        return x.dot(self.U)


