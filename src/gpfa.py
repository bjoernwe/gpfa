import itertools
import numpy as np
import scipy.linalg
import scipy.spatial
import scipy.spatial.distance

import mdp



# def calc_predictability_graph_star(data, k):
#     """
#     Averages the squared distances in a star-like graph.
#     """
# 
#     # pairwise distances of data points
#     N, _ = data.shape
#     distances = scipy.spatial.distance.pdist(data)
#     distances = scipy.spatial.distance.squareform(distances)
#     neighbors = [set(np.argsort(distances[i])[:k+1]).difference([i]) for i in xrange(N)]
# 
#     v = 0
#     for t in range(N-1):
#         x = data[t+1]
#         v += np.mean([np.linalg.norm(x - data[s+1])**2 for s in neighbors[t] if s+1 < N])
#     v /= N-1
#     return v



# def calc_predictability_graph_full(data, k):
#     """
#     Averages the squared distances in a fully connected graph.
#     """
# 
#     if data.ndim == 1:
#         data = np.array(data, ndmin=2).T
# 
#     # pairwise distances of data points
#     N, _ = data.shape
#     distances = scipy.spatial.distance.pdist(data)
#     distances = scipy.spatial.distance.squareform(distances)
#     neighbors = [np.argsort(distances[i])[:k+1] for i in xrange(N)]
# 
#     v = 0
#     for t in range(N-1):
#         v += np.mean([np.linalg.norm(data[i+1] - data[j+1])**2 for i, j in itertools.combinations(neighbors[t], 2) if i+1 < N and j+1 < N])
#     v /= N-1
#     return v



# def calc_predictability_avg_det_of_cov(data, k):
#     """
#     The assumption in the paper that the noise covariance is the same everywhere
#     probably doesn't hold on real-world data sets. Therefore this measure of
#     predictability calculated the determinant of the covariance for each time
#     step and returns the average.  
#     """
#     
#     def _det(t):
#         neighbors = np.array(kdtree.query(data[t], k=k)[1])
#         successors = neighbors + 1
#         successors = successors[successors<N]
#         suc_dat = data[successors]
#         return np.linalg.det(np.array(np.cov(suc_dat.T), ndmin=2))
# 
#     if data.ndim == 1:
#         data = np.array(data, ndmin=2).T
# 
#     # calculate average of determinants
#     N, _ = data.shape
#     kdtree = scipy.spatial.KDTree(data)
#     determinants = map(_det, range(N-1))
#     return np.mean(determinants)



def calc_predictability_det_of_avg_cov(data, k):
    """
    Calculates the predictability as written in the paper. There it is assumed
    that the noise covariance is the same everywhere. Therefore the empirical
    covariances of all time steps are averaged and the determinant calculated in
    the end.
    """
    
    def _cov(t):
        successors = neighbors[t] + 1
        successors = successors[successors<N]
        suc_dat = data[successors]
        return np.array(np.cov(suc_dat.T), ndmin=2)

    if data.ndim == 1:
        data = np.array(data, ndmin=2).T

    # pairwise distances of data points
    N, _ = data.shape
    distances = scipy.spatial.distance.pdist(data)
    distances = scipy.spatial.distance.squareform(distances)
    neighbors = [np.argsort(distances[i])[:k+1] for i in xrange(N)]
    
    covariances = map(_cov, range(N-1))
    covariance = reduce(lambda x,y: x+y, covariances) / (N-1)
    return np.linalg.det(covariance)



def calc_predictability_trace_of_avg_cov(data, k):
    """
    """
    
    def _cov(t):
        successors = neighbors[t] + 1
        successors = successors[successors<N]
        suc_dat = data[successors]
        return np.array(np.cov(suc_dat.T), ndmin=2)

    if data.ndim == 1:
        data = np.array(data, ndmin=2).T

    # pairwise distances of data points
    N, _ = data.shape
    distances = scipy.spatial.distance.pdist(data)
    distances = scipy.spatial.distance.squareform(distances)
    neighbors = [np.argsort(distances[i])[:k+1] for i in xrange(N)]
    
    covariances = map(_cov, range(N-1))
    covariance = reduce(lambda x,y: x+y, covariances) / (N-1)
    return np.trace(covariance)



def calc_predictability_sum_eig(data, k):
    
    def _cov(t):
        successors = neighbors[t] + 1
        successors = successors[successors<N]
        suc_dat = data[successors]
        return np.array(np.cov(suc_dat.T), ndmin=2)
    
    def _eig(C):
        E, _ = np.linalg.eigh(C)
        return np.sum(E)

    if data.ndim == 1:
        data = np.array(data, ndmin=2).T

    # pairwise distances of data points
    N, _ = data.shape
    distances = scipy.spatial.distance.pdist(data)
    distances = scipy.spatial.distance.squareform(distances)
    neighbors = [np.argsort(distances[i])[:k+1] for i in xrange(N)]
    
    covariances = map(_cov, range(N-1))
    eigenvalues = map(_eig, covariances)
    return np.mean(eigenvalues)



def calc_predictability_avg_variance(data, k):
    """
    Calculates the average future variance for each component.
    """
    
    if data.ndim == 1:
        data = np.array(data, ndmin=2).T
        
    dims = data.shape[1]
    result = np.zeros(dims)
        
    for i in range(dims):
        result[i] = calc_predictability_det_of_avg_cov(data[:,i], k=k)
        
    return result



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

    def __init__(self, output_dim, k=10, iterations=1, iteration_dim=None,
                 variance_graph=False, neighborhood_graph=True, weighted_edges=True, 
                 causal_features=False, constraint_optimization=True, input_dim=None, 
                 dtype=None):
        super(gPFA, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.k = k
        self.iterations = iterations
        self.variance_graph = variance_graph
        self.neighborhood_graph = neighborhood_graph
        self.weighted_edges = weighted_edges
        self.iteration_dim = iteration_dim
        self.causal_features = causal_features
        self.constraint_optimization = constraint_optimization
        self.L = None
        self.D = None
        return


    def _train(self, x):

        if self.causal_features:
            x = x[::-1,:]

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



class gPFAkernelized(mdp.Node):

    def __init__(self, output_dim, k=10, iterations=1, iteration_dim=None,
                 variance_graph=False, weighted_edges=True, kernel_poly_degree=2, 
                 constraint_optimization=True, input_dim=None, dtype=None):
        super(gPFAkernelized, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.k = k
        self.iterations = iterations
        self.variance_graph = variance_graph
        self.weighted_edges = weighted_edges
        self.kernel_poly_degree = kernel_poly_degree
        self.iteration_dim = iteration_dim
        self.constraint_optimization = constraint_optimization
        self.L = None
        self.D = None
        self.X = None
        return


    def _train(self, x):

        # remember training data
        assert self.X is None        
        self.X = x

        # kernel matrix
        K = np.dot(x, x.T)**self.kernel_poly_degree
        self.K = K
        
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
            #if isinstance(self.k, int):
            neighbors = [np.argsort(distances[i])[:self.k+1] for i in xrange(N)]
            #elif isinstance(self.k, float):
            #    neighbors = [np.array([j for (j, d) in enumerate(distances[i]) if d <= self.k], dtype=int) for i in xrange(N)]
            #else:
            #    assert False

            # future-preserving graph
            index_list = []
            if self.variance_graph:
                for t in range(N-1):
                    index_list += itertools.permutations(neighbors[t]+1, 2)
            else:
                for s in range(N-1):
                    index_list += [(s+1,t) for t in neighbors[s]+1]
                    index_list += [(t,s+1) for t in neighbors[s]+1]

            # count edges only once
            if not self.weighted_edges:
                index_list = list(set(index_list))
                
            # weight matrix from index list
            index_list = np.array(index_list)
            W = scipy.sparse.coo_matrix((np.ones(index_list.shape[0]), (index_list[:,0], index_list[:,1])), shape=(N+1,N+1))
            #W = W.tocsr()
            W = W.todense()
            W = W[:N,:N]    # cut the N+1 elements
    
            # graph Laplacian
            d = W.sum(axis=1).T
            if d[0,0] == 0:
                W[0,1:] = 1
                W[1:,0] = 1
                d = W.sum(axis=1).T
            print np.sum(d==0)
            D = scipy.sparse.dia_matrix((d, 0), shape=(N, N)).todense()
            L = D - W
    
            #print scipy.linalg.eigh(L)[0]

            # projected graph laplacian
            #D2 = x.T.dot(D.dot(x))
            #L2 = x.T.dot(L.dot(x))
            
            # kernalized eigenvalue problem
            D2 = K.dot(D.dot(K))
            L2 = K.dot(L.dot(K))

            # (if not the last iteration:) solve and project
#             if l < self.iterations-1:
#                 iteration_dim = self.output_dim if self.iteration_dim is None else min(self.iteration_dim, self.output_dim)
#                 if self.constraint_optimization:
#                     _, U = scipy.linalg.eigh(L2, b=D2, eigvals=(0, iteration_dim-1))
#                     # normalize eigenvectors 
#                     for i in range(U.shape[1]):
#                         U[:,i] /= np.linalg.norm(U[:,i])
#                 else:
#                     _, U = scipy.linalg.eigh(L2, eigvals=(0, iteration_dim-1))
#                 y = x.dot(U)

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
            self.E, self.U = scipy.linalg.eigh(self.L, b=self.D, eigvals=(1, self.output_dim))
            # only consider real EVs
            #real_idc = [i for i, e in enumerate(self.E) if e.imag == 0]
            #print len(real_idc)
            #self.E = self.E[real_idc]
            #self.U = self.U[:,real_idc]
            # sort
            #idc = np.argsort(np.abs(self.E))[:self.output_dim]
            #self.E = self.E[idc]
            #self.U = self.U[:,idc]
            # normalize eigenvectors 
            for i in range(self.U.shape[1]):
                self.U[:,i] /= np.linalg.norm(self.U[:,i])
        else:
            self.E, self.U = scipy.linalg.eigh(self.L, eigvals=(0, self.output_dim-1))
    
        # normalize directions
        mask = self.U[0,:] > 0
        self.U = self.U * mask - self.U * ~mask
        
        import matplotlib.pyplot as plt
        plt.plot(self.U[:,1])
        plt.show()
        return


    def _execute(self, x):
        KX = np.dot(self.X, x.T)**self.kernel_poly_degree
        return self.U.T.dot(KX)



