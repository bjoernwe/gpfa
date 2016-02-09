import itertools
import numpy as np
import scipy.linalg
import scipy.spatial.distance

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel

import mdp



def calc_predictability_trace_of_avg_cov(x, k, p, ndim=False):
    """
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
    return np.trace(covariance)



def concatenate_past(x, p=1):
    if x.ndim == 1:
        x = np.array(x, ndmin=2).T
    N, D = x.shape
    y = np.hstack([x[i:N-p+i+1] for i in range(p)])
    y = np.vstack([np.zeros(D*p) for _ in xrange(p-1)] + [y])
    #assert y.shape == (N-p+1, D*p)
    assert y.shape == (N, D*p)
    return y



class RandomProjection(mdp.Node):

    def __init__(self, output_dim, input_dim=None, dtype=None, seed=None):
        super(RandomProjection, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.rnd = np.random.RandomState(seed=seed)
        return
    
    
    def _train(self, x):
        return


    def _stop_training(self):
        D = self.input_dim
        A = self.rnd.rand(D, D)
        A = A + A.T
        _, self.U = scipy.linalg.eigh(A, eigvals=(0, self.output_dim-1))
        assert np.allclose(self.U.T.dot(self.U), np.eye(self.output_dim, self.output_dim))
        return


    def _execute(self, x):
        return x.dot(self.U)



class LPP(mdp.Node):

    def __init__(self, output_dim, k=10, weighted_edges=False, constraint_optimization=True, input_dim=None, dtype=None):
        super(LPP, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.k = k
        self.weighted_edges = weighted_edges
        self.constraint_optimization = constraint_optimization
        self.L = None
        self.D = None
        return


    def _train(self, x):

        # number of samples
        N, _ = x.shape

        # pairwise distances of data points
        distances = scipy.spatial.distance.pdist(x)
        distances = scipy.spatial.distance.squareform(distances)
        if isinstance(self.k, int):
            neighbors = [np.argsort(distances[i])[:self.k+1] for i in xrange(N)]
        elif isinstance(self.k, float):
            neighbors = [np.array([j for (j, d) in enumerate(distances[i]) if d <= self.k], dtype=int) for i in xrange(N)]
        else:
            assert False

        # neighborhood graph
        index_list = []
        for s in range(N):
            index_list += [(s,t) for t in neighbors[s]]
            index_list += [(t,s) for t in neighbors[s]]

        # count edges only once
        if not self.weighted_edges:
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



class gPFA(mdp.Node):

    def __init__(self, output_dim, k=10, p=1, iterations=10, variance_graph=False, 
                 neighborhood_graph=False, weighted_edges=True, causal_features=True, 
                 input_dim=None, dtype=None):
        super(gPFA, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.k = k
        self.p = p
        self.iterations = iterations
        self.variance_graph = variance_graph
        self.neighborhood_graph = neighborhood_graph
        self.weighted_edges = weighted_edges
        self.causal_features = causal_features
        self.L = None
        self.D = None
        return
    
    
    def _train(self, x):

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
                #iteration_dim = self.output_dim if self.iteration_dim is None else min(self.iteration_dim, self.output_dim)
                _, U = scipy.linalg.eigh(L2, b=D2, eigvals=(0, self.output_dim-1))
                #_, U = scipy.linalg.eigh(W2, b=D2, eigvals=(self.input_dim-self.output_dim-1, self.input_dim-1))
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
        self.E, self.U = scipy.linalg.eigh(self.L, b=self.D, eigvals=(0, self.output_dim-1))
        #self.E, self.U = scipy.linalg.eigh(self.W, b=self.D, eigvals=(self.input_dim-self.output_dim-1, self.input_dim-1))
        # normalize eigenvectors 
        for i in range(self.U.shape[1]):
            self.U[:,i] /= np.linalg.norm(self.U[:,i])
    
        # normalize directions
        mask = self.U[0,:] > 0
        self.U = self.U * mask - self.U * ~mask
        return


    def _execute(self, x):
        return x.dot(self.U)



class gPFAsr(mdp.Node):

    def __init__(self, output_dim, k=10, iterations=10, variance_graph=True, 
                 neighborhood_graph=False, weighted_edges=True, 
                 causal_features=True, input_dim=None, dtype=None):
        super(gPFAsr, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.k = k
        self.iterations = iterations
        self.variance_graph = variance_graph
        self.neighborhood_graph = neighborhood_graph
        self.weighted_edges = weighted_edges
        self.causal_features = causal_features
        #self.W = None
        #self.D = None
        return


    def _train(self, X):

        # number of samples
        N, _ = X.shape

        # from Y we calculate the euclidean distances
        # after the first iteration it contains the projected data
        Y = X

        # run algorithm several times e
        for l in range(self.iterations):

            # index lists for neighbors
            tree = scipy.spatial.cKDTree(Y)
            neighbors = [tree.query(Y[i], k=self.k+1)[1] for i in xrange(N)]

            # future-preserving graph
            index_list = []
            if self.variance_graph:
                for t in range(N-1):
                    index_list += itertools.permutations(neighbors[t]+1, 2)
            else:
                for s in range(N-1):
                    index_list += [(s+1,t) for t in neighbors[s]+1]
                    index_list += [(t,s+1) for t in neighbors[s]+1]

            if self.causal_features:
                if self.variance_graph:
                    for t in range(1, N):
                        index_list += itertools.permutations(neighbors[t]-1, 2)
                else:
                    for s in range(1, N):
                        index_list += [(s-1,t) for t in neighbors[s]-1]
                        index_list += [(t,s-1) for t in neighbors[s]-1]

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
            #D = scipy.sparse.dia_matrix((d, 0), shape=(N, N))
            Dinv = scipy.sparse.dia_matrix((1./d, 0), shape=(N, N))
            V = Dinv.dot(W)
            #L = D - W
            #L = L.tocsr()
    
            #iteration_dim = self.output_dim if self.iteration_dim is None else min(self.iteration_dim, self.output_dim)
            #_, B = scipy.sparse.linalg.eigh(W, b=D, eigvals=(self.input_dim-self.output_dim-1, self.input_dim-1))
            _, B = scipy.sparse.linalg.eigsh(V, k=self.output_dim)
            self.U, _, _, _ = np.linalg.lstsq(X, B)
            assert self.U.shape == (self.input_dim, self.output_dim)
            
            # normalize eigenvectors 
            for i in range(self.U.shape[1]):
                self.U[:,i] /= np.linalg.norm(self.U[:,i])
                    
            # (if not the last iteration:) solve and project
            if l < self.iterations-1:
                Y = X.dot(self.U)

        self.stop_training()
        return


    def _stop_training(self):
        #self.E, self.U = scipy.linalg.eigh(self.W, b=self.D, eigvals=(self.input_dim-self.output_dim-1, self.input_dim-1))
        # normalize eigenvectors 
        #for i in range(self.U.shape[1]):
        #    self.U[:,i] /= np.linalg.norm(self.U[:,i])
    
        # normalize directions
        mask = self.U[0,:] > 0
        self.U = self.U * mask - self.U * ~mask
        return


    def _execute(self, x):
        return x.dot(self.U)



class gPFAkernel(mdp.Node):

    def __init__(self, output_dim, k=10, iterations=10, degree=3, 
                 variance_graph=True, input_dim=None, dtype=None):
        super(gPFAkernel, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.k = k
        self.degree = degree
        self.iterations = iterations
        self.variance_graph = variance_graph
        self.L = None
        self.D = None
        return


    def _remove_duplicates(self, X, tol=1e-6):
        D = pairwise_distances(X)
        idc = np.sum(np.tril(D < tol), axis=1) <= 1
        return X[idc]
    
    
    def _train(self, x):
        
        x = self._remove_duplicates(x, tol=1e-4)
        self.X = x

        # number of samples
        N, _ = x.shape

        # from y we calculate the euclidean distances
        # after the first iteration it contains the projected data
        y = x

        # run algorithm several times e
        for l in range(self.iterations):

            # index lists for neighbors
            tree = scipy.spatial.cKDTree(y)
            neighbors = [tree.query(y[i], k=self.k+1)[1] for i in xrange(N)]

            # future-preserving graph
            index_list = []
            if self.variance_graph:
                for t in range(N-1):
                    index_list += itertools.permutations(neighbors[t]+1, 2)
                for t in range(1, N):
                    index_list += itertools.permutations(neighbors[t]-1, 2)
            else:
                for s in range(N-1):
                    index_list += [(s+1,t) for t in neighbors[s]+1]
                    index_list += [(t,s+1) for t in neighbors[s]+1]
                for s in range(1, N):
                    index_list += [(s-1,t) for t in neighbors[s]-1]
                    index_list += [(t,s-1) for t in neighbors[s]-1]

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
    
            # gram matrix
            #K = polynomial_kernel(y, degree=self.degree)
            K = rbf_kernel(y, gamma=.1)
            self.K = K
            #print N, K.shape, '***', np.linalg.matrix_rank(K)
            D2 = K.T.dot(D.dot(K))
            #L2 = K.T.dot(L.dot(K))
            L2 = K.T.dot(W.dot(K))

            # (if not the last iteration:) solve and project
            if l < self.iterations-1:
                #_, U = scipy.linalg.eigh(L2, b=D2, eigvals=(0, self.output_dim-1))
                _, U = scipy.linalg.eigh(L2, b=D2, eigvals=(N-self.output_dim-1, N-1))
                # normalize eigenvectors 
                for i in range(U.shape[1]):
                    U[:,i] /= np.linalg.norm(U[:,i])
                #print x.T.shape, U.shape
                #y = x.T.dot(U)
                #alphas = polynomial_kernel(y, x, degree=self.degree)
                #print alphas.shape
                y = K.dot(U)

        # add chunk result to global result
        # TODO: no chunks in kernel-version!
        if self.L is None:
            self.L = L2
            self.D = D2
        else:
            self.L += L2
            self.D += D2

        return


    def _stop_training(self):
        #self.E, self.U = scipy.linalg.eigh(self.L, b=self.D, eigvals=(0, self.output_dim-1))
        N, _ = self.X.shape
        self.E, self.U = scipy.linalg.eigh(self.L, b=self.D, eigvals=(N-self.output_dim-1, N-1))
        # normalize eigenvectors 
        for i in range(self.U.shape[1]):
            self.U[:,i] /= np.linalg.norm(self.U[:,i])
    
        # normalize directions
        mask = self.U[0,:] > 0
        self.U = self.U * mask - self.U * ~mask
        return


    def _execute(self, x):
        #y = polynomial_kernel(self.X, x, degree=self.degree)
        y = rbf_kernel(self.X, x, gamma=.1)
        return self.U.T.dot(y).T
