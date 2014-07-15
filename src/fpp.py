import itertools
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import scipy.spatial.distance

from matplotlib import pyplot

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

    def __init__(self, output_dim, k=10, normalized_objective=True, 
                 input_dim=None, dtype=None):
        super(LPP, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.k = k
        self.normalized_objective = normalized_objective
        self.L = None
        self.D = None
        return
    
    
    def _train(self, x):

        # number of samples
        N, _ = x.shape
        
        # from y we calculate the euclidean distances
        # after the first iteration it contains the projected data
        y = x
        
        # initialize weight matrix W
        W = scipy.sparse.dok_matrix((N, N))
    
        # pairwise distances of data points
        distances = scipy.spatial.distance.pdist(y)
        distances = scipy.spatial.distance.squareform(distances)
        neighbors = [np.argsort(distances[i])[:self.k+1] for i in range(N)]
        
        # neighbor graph
        for s in range(N):
            for t in neighbors[s]:#[0:self.k+1]:
                W[s,t] += 1
                W[t,s] += 1

        # graph Laplacian
        d = W.sum(axis=1).T
        #d[d==0] = float('inf') 
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
        if self.normalized_objective:
            self.E, self.U = scipy.sparse.linalg.eigsh(self.L, M=self.D, k=self.output_dim, which='SM')
            for i in range(len(self.E)):
                self.U[:,i] /= np.linalg.norm(self.U[:,i])
        else:
            self.E, self.U = scipy.sparse.linalg.eigsh(self.L, k=self.output_dim, which='SM')
    
        # normalize directions
        mask = self.U[0,:] > 0
        self.U = self.U * mask - self.U * ~mask
        return


    def _execute(self, x):
        return x.dot(self.U)



class FPP(mdp.Node):

    def __init__(self, output_dim, k=10, iterations=1, iteration_dim=None,
                 minimize_variance=False, normalized_objective=True, 
                 input_dim=None, dtype=None):
        super(FPP, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.k = k
        self.iterations = iterations
        self.minimize_variance = minimize_variance
        self.normalized_objective = normalized_objective
        self.iteration_dim = iteration_dim
        #if self.iteration_dim is None:
        #    self.iteration_dim = 'auto'
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
        
        # run algorithm several times e
        for l in range(self.iterations):

            # initialize weight matrix W
            W = scipy.sparse.dok_matrix((N, N))
        
            # pairwise distances of data points
            distances = scipy.spatial.distance.pdist(y)
            distances = scipy.spatial.distance.squareform(distances)
            neighbors = [np.argsort(distances[i])[:self.k+1] for i in range(N)]
            #neighbors = [np.where(distances[i,:] <= 1.5)[0] for i in range(N)]
            #for ne in neighbors:
            #    print len(ne)
            
            # future-preserving graph
            if self.minimize_variance:
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
            if self.minimize_variance:
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
                if self.normalized_objective:
                    if type(self.iteration_dim) == int:
                        E, U = scipy.sparse.linalg.eigsh(L2, M=D2, which='SM', k=self.iteration_dim)
                        for i in range(len(E)):
                            U[:,i] /= np.linalg.norm(U[:,i])
                    else:
                        E, U = scipy.sparse.linalg.eigsh(L2, M=D2, which='SM')
                        for i in range(len(E)):
                            U[:,i] /= np.linalg.norm(U[:,i]) / np.sqrt(E[i])
                else:
                    E, U = scipy.sparse.linalg.eigsh(L2, k=self.iteration_dim, which='SM')
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
        if self.normalized_objective:
            self.E, self.U = scipy.sparse.linalg.eigsh(self.L, M=self.D, k=self.output_dim, which='SM')
            for i in range(len(self.E)):
                self.U[:,i] /= np.linalg.norm(self.U[:,i])
        else:
            self.E, self.U = scipy.sparse.linalg.eigsh(self.L, k=self.output_dim, which='SM')
    
        # normalize directions
        mask = self.U[0,:] > 0
        self.U = self.U * mask - self.U * ~mask
        return


    def _execute(self, x):
        return x.dot(self.U)



class FPPnl(mdp.Node):

    def __init__(self, output_dim, k=10, iterations=1, iteration_dim=None,
                 minimize_variance=False, normalized_objective=True, 
                 input_dim=None, dtype=None):
        super(FPPnl, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.k = k
        self.iterations = iterations
        self.minimize_variance = minimize_variance
        self.normalized_objective = normalized_objective
        self.iteration_dim = iteration_dim
        if self.iteration_dim is None:
            self.iteration_dim = self.output_dim
        self.L = None
        self.D = None
        return
    
    
    def _train(self, x):

        # number of samples
        N, _ = x.shape
        
        # from y we calculate the euclidean distances
        # after the first iteration it contains the projected data
        y = x
        
        # run algorithm several times 
        for l in range(self.iterations):

            # initialize weight matrix W
            #W = scipy.sparse.dok_matrix((N, N))
            W = np.zeros((N, N))
            W -= 0.01
        
            # pairwise distances of data points
            distances = scipy.spatial.distance.pdist(y)
            distances = scipy.spatial.distance.squareform(distances)
            neighbors = [np.argsort(distances[i])[:self.k+1] for i in range(N)]
            #neighbors = [np.where(distances[i,:] <= 1.5)[0] for i in range(N)]
            #for ne in neighbors:
            #    print len(ne)
            
            # future-preserving graph
            if self.minimize_variance:
                for t in range(N-1):
                    for (i,j) in itertools.permutations(neighbors[t], 2):
                        if i+1 < N and j+1 < N:
                            W[i+1,j+1] += 1
            else:
                for s in range(N-1):
                    for t in neighbors[s]:#[0:self.k+1]:
                        if s+1 < N and t+1 < N:
                            W[s+1,t+1] = 1
                            W[t+1,s+1] = 1
                            
            W[0,1] += 1
            W[1,0] += 1
    
            # graph Laplacian
            d = W.sum(axis=1).T
            #d[d==0] = float('inf') 
            D = scipy.sparse.dia_matrix((d, 0), shape=(N, N))
            L = D - W
    
            # projected graph laplacian
            #D2 = x.T.dot(D.dot(x))
            #L2 = x.T.dot(L.dot(x))

            # (if not the last iteration:) solve and project
            if l < self.iterations-1:
                if self.normalized_objective:
                    E, U = scipy.sparse.linalg.eigsh(L, M=D, k=self.iteration_dim, which='SM')
                    for i in range(len(E)):
                        U[:,i] /= np.linalg.norm(U[:,i])
                else:
                    E, U = scipy.sparse.linalg.eigsh(L, k=self.iteration_dim, which='SM')
                #y = x.dot(U)
                y = U

        # add chunk result to global result
        if self.L is None:
            self.L = L
            self.D = D
        else:
            self.L += L
            self.D += D

        return


    def _stop_training(self):
        if self.normalized_objective:
            self.E, self.U = scipy.sparse.linalg.eigsh(self.L, M=self.D, k=self.output_dim+1, which='SM')
            for i in range(len(self.E)):
                self.U[:,i] /= np.linalg.norm(self.U[:,i])
        else:
            self.E, self.U = scipy.sparse.linalg.eigsh(self.L, k=self.output_dim+1, which='SM')
        return


    def _execute(self, x):
        print self.U.shape, x.shape
        return x.dot(self.U)



class gPFA(mdp.Node):

    def __init__(self, output_dim, k=10, iterations=1, iteration_dim=None,
                 minimize_variance=False, normalized_objective=False,
                 input_dim=None, dtype=None):
        super(gPFA, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.k = k
        self.iterations = iterations
        self.iteration_dim = iteration_dim
        self.minimize_variance = minimize_variance
        self.normalized_objective = normalized_objective
        if self.iteration_dim is None:
            self.iteration_dim = self.output_dim
        self.C = None
        return

    
    def _train(self, x):
        
        # TODO: whitening

        # number of samples
        N, _ = x.shape

        # data y is used for calculating the k neighbors
        y = x

        # iterate algorithm            
        for l in range(self.iterations):
                    
            # pairwise distances of data points
            distances = scipy.spatial.distance.pdist(y)
            distances = scipy.spatial.distance.squareform(distances)
            neighbors = [np.array(np.argsort(distances[i])[:self.k+1], dtype=int) for i in range(N-1)]
            cov = mdp.utils.CovarianceMatrix()

            if self.minimize_variance:
                for t, neighborhood in enumerate(neighbors):
                    neighborhood = np.setdiff1d(neighborhood, np.array([N-1]), assume_unique=True)
                    future = neighborhood + 1
                    mu = np.mean(x[future], axis=0)
                    deltas = x[future] - mu#x[t+1]
                    cov.update(deltas)
            else:
                for t, neighborhood in enumerate(neighbors):
                    neighborhood = np.setdiff1d(neighborhood, np.array([N-1]), assume_unique=True)
                    future = neighborhood + 1
                    deltas = x[future] - x[t+1]
                    cov.update(deltas)
    
            C, _, _ = cov.fix(center=False)
    
            # (if not the last iteration:) solve and project
            if l < self.iterations-1:
                #E, U = scipy.linalg.eigh(a=C, eigvals=(0, self.iteration_dim-1))
                E, U = scipy.sparse.linalg.eigsh(C, k=self.iteration_dim, which='SM')
                print min(E), max(E)
                y = x.dot(U)

        if self.C is None:
            self.C = C
        else:
            self.C += C
            

    def _stop_training(self):
        #self.E, self.U = scipy.linalg.eigh(a=self.C, eigvals=(0, self.output_dim-1))
        #self.E, self.U = scipy.linalg.eigh(a=np.eye(self.input_dim)-self.C, eigvals=(self.input_dim-self.output_dim, self.input_dim-1))
        self.E, self.U = scipy.sparse.linalg.eigsh(self.C, k=self.output_dim, which='SM')
        print self.E
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

