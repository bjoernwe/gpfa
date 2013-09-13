import numpy as np
import scipy.sparse.linalg
import scipy.spatial.distance

import mdp


class FuturePreservingMapBase(mdp.Node):

    data = None

    def __init__(self, output_dim, k=10, normalized_laplacian=True, neighbor_edges=True, input_dim=None, dtype=None):
        super(FuturePreservingMapBase, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.k = k
        self.normalized_laplacian = normalized_laplacian
        self.neighbor_edges = neighbor_edges
        
    
    def _train(self, x):
        if self.data is None:
            self.data = x
        else:
            np.vstack((self.data, x))
    
    
    def _stop_training(self):
        """
        Calculates the graph Laplacian.
        """
        
        # initialize weight matrix W
        N, _ = self.data.shape
        W = scipy.sparse.dok_matrix((N, N))
       
        # pairwise distances of data points
        distances = scipy.spatial.distance.pdist(self.data)
        distances = scipy.spatial.distance.squareform(distances)

        # future-preserving graph
        for t in range(N-1):
            neighbors = np.argsort(distances[t+1])
            for v in neighbors[0:self.k+1]:
                u = v-1     # predecessor of neighbor
                if t != u:  # no self-connections
                    W[t,u] = 1
                    W[u,t] = 1
                    
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
        if self.normalized_laplacian:
            d_inv = 1./d
            D_inv = scipy.sparse.dia_matrix((d_inv, 0), shape=(N, N))
            W = D_inv.dot(W)
            D = scipy.sparse.eye(N, N)
        else:
            D = scipy.sparse.dia_matrix((d, 0), shape=(N, N))
        
        # keep matrices
        L = D - W
        self.W = W
        self.D = D
        self.L = L
        return
        


class FuturePreservingMap(FuturePreservingMapBase):
    
    def __init__(self, output_dim, k=10, normalized_laplacian=True, neighbor_edges=True, input_dim=None, dtype=None):
        FuturePreservingMapBase.__init__(self, output_dim=output_dim, k=k, normalized_laplacian=normalized_laplacian, neighbor_edges=neighbor_edges, input_dim=input_dim, dtype=dtype)

        
    def _stop_training(self):
        FuturePreservingMapBase._stop_training(self)
        if self.normalized_laplacian:
            E, U = scipy.sparse.linalg.eigs(self.W, k=self.output_dim, which='LR')
            self.E = E.real
            self.U = U.real
        else:
            self.E, self.U = scipy.sparse.linalg.eigsh(self.L, M=self.D, sigma=0.0, k=self.output_dim, which='LR')
        
        
        
class FuturePreservingMapLinear(FuturePreservingMapBase):
    
    def __init__(self, output_dim, k=10, normalized_laplacian=True, neighbor_edges=True, input_dim=None, dtype=None):
        FuturePreservingMapBase.__init__(self, output_dim=output_dim, k=k, normalized_laplacian=normalized_laplacian, neighbor_edges=neighbor_edges, input_dim=input_dim, dtype=dtype)
    
        
    def _stop_training(self):
        FuturePreservingMapBase._stop_training(self)
        
        # projected graph laplacian
        D2 = self.data.T.dot(self.D.dot(self.data))
        L2 = self.data.T.dot(self.L.dot(self.data))
        #W2 = self.data.T.dot(self.W.dot(self.data))
        
        # calculate the eigen-vectors
        print L2
        self.E, self.U = scipy.sparse.linalg.eigs(L2, M=D2, sigma=0.0, k=self.output_dim, which='LR')
        #self.E, self.U = scipy.sparse.linalg.eigsh(L2, M=D2, k=self.output_dim, which='SM')
        #self.E, self.U = scipy.sparse.linalg.eigsh(W2, k=self.output_dim, which='LM')
        return
    
    
    def _execute(self, x):
        return x.dot(self.U)
    
    