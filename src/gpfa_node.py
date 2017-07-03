import itertools
import numpy as np
import scipy.linalg
import scipy.spatial.distance

import mdp


def calc_predictability_trace_of_avg_cov(x, k, p, ndim=False):
    """
    The main evaluation criterion of GPFA, i.e., equation (2) from the paper.

    :param x: data array
    :param k: number of neighbors for estimate
    :param p: number of past time steps to consider
    :param ndim: n-dimensional evaluation if True
    :return: estimated variance in the next time step
    """

    def _cov(t):
        successors = neighbors[t] + 1
        successors = successors[successors<N]
        suc_dat = x[successors]
        return np.array(np.cov(suc_dat.T), ndmin=2)

    # pairwise distances of data points
    if x.ndim == 1:
        x = np.array(x, ndmin=2).T 
    N, _ = x.shape
    y = concatenate_past(x, p=p)
    tree = scipy.spatial.cKDTree(y)
    neighbors = [tree.query(y[i], k=k+1)[1] for i in xrange(y.shape[0])]
    assert len(neighbors) == N
    
    covariances = map(_cov, range(p-1, N-1))
    covariance = reduce(lambda a,b: a+b, covariances) / (N-p)
    if ndim:
        E, _ = np.linalg.eigh(covariance)
        return E
    result = np.trace(covariance)
    assert np.isfinite(result)
    return result 


def concatenate_past(x, p=1):
    """
    Helper function for time-embedded data
    :param x: data
    :param p: number of time frames embedded into one frame
    :return: embedded data
    """
    if x.ndim == 1:
        x = np.array(x, ndmin=2).T
    N, D = x.shape
    y = np.hstack([x[i:N-p+i+1] for i in range(p)])
    y = np.vstack([np.zeros(D*p) for _ in xrange(p-1)] + [y])
    #assert y.shape == (N-p+1, D*p)
    assert y.shape == (N, D*p)
    return y


class GPFA(mdp.Node):

    def __init__(self, output_dim, k=10, p=1, iterations=10, variance_graph=False, 
                 neighborhood_graph=False, weighted_edges=True, causal_features=True, 
                 generalized_eigen_problem=True, input_dim=None, dtype=None):
        """
        :param output_dim: number of features to extract
        :param k: number of neighbors for estimation
        :param p: number of past time steps to consider
        :param iterations: number of iterations of the GPFA algorithm
        :param variance_graph: GPFA(1) if True, GPFA(2) aka star-shaped graph if False
        :param neighborhood_graph: Additional edges for nearest neighbors like in LPP
        :param weighted_edges: Weight edges as often as they occur (True)
        :param causal_features: Also minimize variance of the past
        :param generalized_eigen_problem: Include the diagonal matrix with node weights into the eigen-problem (see von Luxburg)
        :param input_dim: Will be set automatically
        :param dtype: The default will usually work fine
        """
        super(GPFA, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.k = k
        self.p = p
        self.iterations = iterations
        self.variance_graph = variance_graph
        self.neighborhood_graph = neighborhood_graph
        self.weighted_edges = weighted_edges
        self.causal_features = causal_features
        self.generalized_eigen_problem = generalized_eigen_problem
        self.L = None
        self.D = None
        self.whitening = None
        return
    
    def _train(self, x, is_whitened=False):

        # whiten data
        if not is_whitened:
            self.whitening = mdp.nodes.WhiteningNode(reduce=True)
            self.whitening.train(x)
            x = self.whitening.execute(x)

        # number of samples
        N, dim = x.shape
        p = self.p

        # from y we calculate the euclidean distances
        # after the first iteration it contains the projected data
        y = concatenate_past(x, p=p)
        assert y.shape == (N, p*dim)

        # run algorithm several times e
        for l in range(self.iterations):

            # index lists for neighbors
            tree = scipy.spatial.cKDTree(y)
            neighbors = [tree.query(y[i], k=self.k+1)[1] for i in xrange(y.shape[0])]
            assert len(neighbors) == N

            # future-preserving graph
            index_list = []
            if self.variance_graph:
                for t in range(p-1, N-1):
                    index_list += itertools.permutations(neighbors[t]+1, 2)
            else:
                for s in range(p-1, N-1):
                    index_list += [(s+1,t) for t in neighbors[s]+1]
                    index_list += [(t,s+1) for t in neighbors[s]+1]

            if self.causal_features:
                if self.variance_graph:
                    for t in range(p, N):
                        index_list += itertools.permutations(neighbors[t]-p, 2)
                else:
                    for s in range(p, N):
                        index_list += [(s-p,t) for t in neighbors[s]-p]
                        index_list += [(t,s-p) for t in neighbors[s]-p]

            # neighborhood graph
            if self.neighborhood_graph:
                if self.variance_graph:
                    for t in range(p-1, N):
                        index_list += itertools.permutations(neighbors[t], 2)
                else:
                    for s in range(p-1, N):
                        index_list += [(s,t) for t in neighbors[s]]
                        index_list += [(t,s) for t in neighbors[s]]

            # count edges only once
            if not self.weighted_edges:
                index_list = list(set(index_list))
                
            # weight matrix from index list
            index_list = np.array(index_list)
            index_list = np.delete(index_list, np.where(index_list[:,0] < 0), axis=0)
            index_list = np.delete(index_list, np.where(index_list[:,1] < 0), axis=0)
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
            #W2 = x.T.dot(W.dot(x))

            # (if not the last iteration:) solve and project
            if l < self.iterations-1:
                if self.generalized_eigen_problem:
                    _, U = scipy.linalg.eigh(L2, b=D2, eigvals=(0, self.output_dim-1))
                else:
                    _, U = scipy.linalg.eigh(L2, eigvals=(0, self.output_dim-1))
                # normalize eigenvectors 
                for i in range(U.shape[1]):
                    U[:,i] /= np.linalg.norm(U[:,i])
                y = concatenate_past(x.dot(U), p=self.p)

        # add chunk result to global result
        if self.L is None:
            self.L = L2
            self.D = D2
        else:
            self.L += L2
            self.D += D2

        return

    def _stop_training(self):
        if self.generalized_eigen_problem:
            self.E, self.U = scipy.linalg.eigh(self.L, b=self.D, eigvals=(0, self.output_dim-1))
        else:
            self.E, self.U = scipy.linalg.eigh(self.L, eigvals=(0, self.output_dim-1))
        # normalize eigenvectors 
        for i in range(self.U.shape[1]):
            self.U[:,i] /= np.linalg.norm(self.U[:,i])
    
        # normalize directions
        mask = self.U[0,:] > 0
        self.U = self.U * mask - self.U * ~mask
        return

    def _execute(self, x):
        if self.whitening:
            x = self.whitening.execute(x)
        return x.dot(self.U)
