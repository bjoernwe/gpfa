import numpy as np
import matplotlib.pyplot as plt

import mdp


class NonlinearNoiseNode(mdp.Node):
    
    def __init__(self, input_dim=None, dims_modified=1, seed=None, dtype=None):
        super(NonlinearNoiseNode, self).__init__(input_dim=input_dim, dtype=dtype)
        self.dims_modified = dims_modified
        self.weights = None
        self.rnd = np.random.RandomState(seed=seed)
        return
    
    
    def _train(self, X):
        pass
        
        
    def _execute(self, X):
        
        _, D = X.shape
        d1 = self.dims_modified
        d2 = D - d1
        
        if self.weights is None:
            self.weights = [np.random.randn(d2) for _ in range(self.dims_modified)]
        
        Y = X[:,d1:] # noise
        Y2 = Y**2
        
        for i in range(d1):
            X[:,i] += Y2.dot(self.weights[i])
        
        return X
    


if __name__ == '__main__':
    
    N = 1000
    X = np.random.randn(N, 2)
    X[:,0] = np.sin(range(N))
    
    noise_node = NonlinearNoiseNode(dims_modified=1)
    X = noise_node.execute(X)
    
    sfa = mdp.nodes.SFA2Node(output_dim=1)
    sfa.train(X)
    plt.plot(sfa.execute(X))
    #plt.plot(X[:,0])
    plt.show()
    