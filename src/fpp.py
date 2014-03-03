import itertools
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import scipy.spatial.distance

import mdp


class FPP(mdp.Node):

    def __init__(self, output_dim, k=10, iterations=1, iteration_dim=None, 
                 preserve_future=True, preserve_past=True, neighbor_graph=False, 
                 input_dim=None, dtype=None):
        super(FPP, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.k = k
        self.iterations = iterations
        self.preserve_future = preserve_future
        self.preserve_past = preserve_past
        self.neighbor_graph = neighbor_graph
        self.iteration_dim = iteration_dim
        if self.iteration_dim is None:
            self.iteration_dim = self.output_dim
        self.L = None
        self.D = None
        return
    
    
#     def _kernel(self, d):
#         print d
#         return np.exp(-.5*(d/self.sigma)**2) / (self.sigma * np.sqrt(2. * np.pi))


#     def _kernel(self, u, v):
#         d = u-v
#         return np.exp(-.5*(d/self.sigma)**2) / (self.sigma * np.sqrt(2. * np.pi))
    
    
    def _train(self, x):

        # number of samples
        N, _ = x.shape
        
        # from y we calculate the euclidean distances
        # after the first iteration it contains the projected data
        y = x
        
        # run algorithm several times 
        for l in range(self.iterations):

            # initialize weight matrix W
            W = scipy.sparse.dok_matrix((N, N))
        
            # pairwise distances of data points
            distances = scipy.spatial.distance.pdist(y)
            distances = scipy.spatial.distance.squareform(distances)
            neighbors = [np.argsort(distances[i])[:self.k+1] for i in range(N)]
    
            # future-preserving graph
            if self.preserve_future:
                for s in range(N-1):
                    for t in neighbors[s]:#[0:self.k+1]:
                        if s != t: # no self-connections
                            if s+1 < N and t+1 < N:
                                W[s+1,t+1] = 1
                                W[t+1,s+1] = 1
    
            # past-preserving graph
            if self.preserve_past:
                for s in range(1, N):
                    for t in neighbors[s]:
                        if s != t: # no self-connections
                            if s-1 > 0 and t-1 >= 0:
                                W[s-1,t-1] = 1
                                W[t-1,s-1] = 1
                            
            # k-nearest-neighbor graph for regularization
            if self.neighbor_graph:
                for i in range(N):
                    for j in neighbors[i]:
                        if i != j:
                            W[i,j] = 1
                            W[j,i] = 1
    
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
                E, U = scipy.sparse.linalg.eigsh(D2-L2, M=D2, k=self.iteration_dim, which='LA')
                E = 1 - E
                #E, U = scipy.linalg.eigh(a=L2, b=D2)
                #(E, U) = (E.real, U.real)
                print min(E), max(E)
                #assert 0 not in E
                #assert float('nan') not in E
                #assert float('nan') not in U 
                #for i in range(len(E)):
                #    U[:,i] = U[:,i] / E[i]**2
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
        self.E, self.U = scipy.sparse.linalg.eigsh(self.D-self.L, M=self.D, k=self.output_dim, which='LA')
        return


    def _execute(self, x):
        return x.dot(self.U)



class gPFA(mdp.Node):

    def __init__(self, output_dim, k=10, iterations=1, iteration_dim=None, 
                 input_dim=None, dtype=None):
        super(gPFA, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.k = k
        self.iterations = iterations
        self.iteration_dim = iteration_dim
        if self.iteration_dim is None:
            self.iteration_dim = self.output_dim
        self.L = None
        self.D = None
        self.C = None
        return

    
    def _train(self, x):
        
        # TODO: whitening

        # number of samples
        N, _ = x.shape
        
        # pairwise distances of data points
        distances = scipy.spatial.distance.pdist(x)
        distances = scipy.spatial.distance.squareform(distances)
        neighbors = [np.array(np.argsort(distances[i])[:self.k+1], dtype=int) for i in range(N-1)]
        
        cov = mdp.utils.CovarianceMatrix()
        for neigh in neighbors:
            neighbor_future = neigh + 1
            neighbor_future = np.setdiff1d(neighbor_future, np.array([N]), assume_unique=True)
            combinations = np.array(list(itertools.combinations(neighbor_future, 2)), dtype=int)
            indices_i = combinations[:,0]
            indices_j = combinations[:,1]
            deltas = x[indices_i] - x[indices_j]
            cov.update(deltas)
            
        C, _, _ = cov.fix()
        if self.C is None:
            self.C = C
        else:
            self.C += C


    def _stop_training(self):
        self.E, self.U = scipy.linalg.eigh(a=self.C, eigvals=(0, self.output_dim-1))
        return


    def _execute(self, x):
        return x.dot(self.U)



class GraphSFA(mdp.Node):

    def __init__(self, output_dim, k=10, normalized_laplacian=True, neighbor_edges=False, input_dim=None, dtype=None):
        super(GraphSFA, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.k = k
        self.normalized_laplacian = normalized_laplacian
        self.neighbor_edges = neighbor_edges
        self.L = None
        self.D = None
        return
    
    
    def _train(self, x):

        # initialize weight matrix W
        N, _ = x.shape
        W = scipy.sparse.dok_matrix((N, N))

        # pairwise distances of data points
        distances = scipy.spatial.distance.pdist(x)
        distances = scipy.spatial.distance.squareform(distances)

        # future-preserving graph
        for s in range(N-1):
            neighbors = np.argsort(distances[s])
            for t in neighbors[0:self.k+1]:
                if t+1 < N:  # no self-connections
                    W[s,t+1] = 1
                    W[t+1,s] = 1

        # k-nearest-neighbor graph for regularization
        if self.neighbor_edges:
            for i in range(N):
                neighbors = np.argsort(distances[i])
                for j in neighbors[0:self.k+1]:
                    if i != j:
                        W[i,j] = 1
                        W[j,i] = 1

        # graph Laplacian
        d = W.sum(axis=1).T
        d[d==0] = float('inf') 
        if self.normalized_laplacian:
            d_inv = 1./d
            D_inv = scipy.sparse.dia_matrix((d_inv, 0), shape=(N, N))
            W = D_inv.dot(W)
            D = scipy.sparse.eye(N, N)
        else:
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
        # calculate the eigen-vectors
        #E, U = scipy.sparse.linalg.eigs(self.L, M=self.D, sigma=0.0, k=self.output_dim, which='LR')
        E, U = scipy.sparse.linalg.eigsh(self.D-self.L, M=self.D, k=self.output_dim, which='LA')
        self.E = E.real
        self.U = U.real
        return


    def _execute(self, x):
        return x.dot(self.U)

