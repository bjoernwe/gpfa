import itertools
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import scipy.spatial.distance

import mdp



def calc_predictability_graph_star(data, k):

    # pairwise distances of data points
    N, _ = data.shape
    distances = scipy.spatial.distance.pdist(data)
    distances = scipy.spatial.distance.squareform(distances)
    neighbors = [set(np.argsort(distances[i])[:k+1]).difference([i]) for i in xrange(N)]

    v = 0
    for t in range(N-1):
        x = data[t+1]
        v += np.mean([np.linalg.norm(x - data[s+1])**2 for s in neighbors[t] if s+1 < N])
    v /= N-1
    return v



def calc_predictability_graph_var(data, k):

    if data.ndim == 1:
        data = np.array(data, ndmin=2).T

    # pairwise distances of data points
    N, _ = data.shape
    distances = scipy.spatial.distance.pdist(data)
    distances = scipy.spatial.distance.squareform(distances)
    neighbors = [np.argsort(distances[i])[:k+1] for i in xrange(N)]

    v = 0
    for t in range(N-1):
        v += np.mean([np.linalg.norm(data[i+1] - data[j+1])**2 for i, j in itertools.combinations(neighbors[t], 2) if i+1 < N and j+1 < N])
    v /= N-1
    return .5*v



def calc_predictability_det_var(data, k):

    if data.ndim == 1:
        data = np.array(data, ndmin=2).T

    # pairwise distances of data points
    N, _ = data.shape
    distances = scipy.spatial.distance.pdist(data)
    distances = scipy.spatial.distance.squareform(distances)
    neighbors = [np.argsort(distances[i])[:k+1] for i in xrange(N)]

    v = 0
    for t in range(N-1):
        successors = neighbors[t] + 1
        successors = successors[successors<N]
        suc_dat = data[successors]
        covariance = np.array(np.cov(suc_dat.T), ndmin=2)
        det = np.linalg.det(covariance)
        v += det
    v /= N-1
    return v



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

    def __init__(self, output_dim, k=10, iterations=1, iteration_dim=None,
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

            # pairwise distances of data points
            distances = scipy.spatial.distance.pdist(y)
            distances = scipy.spatial.distance.squareform(distances)
            if isinstance(self.k, int):
                neighbors = [np.argsort(distances[i])[:self.k+1] for i in xrange(N)]
            elif isinstance(self.k, float):
                neighbors = [np.array([j for (j, d) in enumerate(distances[i]) if d <= self.k], dtype=int) for i in xrange(N)]
            else:
                assert False

            # future-preserving graph
            index_list = []
            if self.variance_graph:
                for t in range(N-1):
                    index_list += itertools.permutations(neighbors[t]+1, 2)
            else:
                for s in range(N-1):
                    index_list += [(s+1,t) for t in neighbors[s]+1]
                    index_list += [(t,s+1) for t in neighbors[s]+1]

            # neighborhood graph
            if self.neighborhood_graph:
                if self.variance_graph:
                    for t in range(N):
                        index_list += itertools.permutations(neighbors[t], 2)
                else:
                    for s in range(N):
                        index_list += [(s,t) for t in neighbors[s]]
                        index_list += [(t,s) for t in neighbors[s]]

            # count edges only once
            if not self.variance_graph:
                index_list = list(set(index_list))
                
            # weight matrix from index list
            index_list = np.array(index_list)
            W = scipy.sparse.coo_matrix((np.ones(index_list.shape[0]), (index_list[:,0], index_list[:,1])), shape=(N+1,N+1))
            W = W.tocsr()
            W = W[:N,:N]    # cut the N+1 elements
    
            # graph Laplacian
            d = W.sum(axis=1).T
            #d[d==0] = float('inf') 
            D = scipy.sparse.dia_matrix((d, 0), shape=(N, N))
            L = D - W
            L = L.tocsr()
    
            # projected graph laplacian
            D2 = x.T.dot(D.dot(x))
            L2 = x.T.dot(L.dot(x))

            # (if not the last iteration:) solve and project
            if l < self.iterations-1:
                iteration_dim = self.output_dim if self.iteration_dim is None else min(self.iteration_dim, self.output_dim)
                if self.constraint_optimization:
                    _, U = scipy.linalg.eigh(L2, b=D2, eigvals=(0, iteration_dim-1))
                    # normalize eigenvectors 
                    for i in range(U.shape[1]):
                        U[:,i] /= np.linalg.norm(U[:,i])
                else:
                    _, U = scipy.linalg.eigh(L2, eigvals=(0, iteration_dim-1))
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
            self.E, self.U = scipy.linalg.eigh(self.L, b=self.D, eigvals=(0, self.output_dim-1))
            # normalize eigenvectors 
            for i in range(self.U.shape[1]):
                self.U[:,i] /= np.linalg.norm(self.U[:,i])
        else:
            self.E, self.U = scipy.linalg.eigh(self.L, eigvals=(0, self.output_dim-1))
    
        # normalize directions
        mask = self.U[0,:] > 0
        self.U = self.U * mask - self.U * ~mask
        return


    def _execute(self, x):
        return x.dot(self.U)


