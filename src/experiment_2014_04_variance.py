import itertools
import numpy as np
from numpy.ma.testutils import assert_almost_equal, almost
    

if __name__ == '__main__':
    
    N = 10
    dim = 2

    data = np.random.random((N, dim))
    
    r1 = [np.linalg.norm(v)**2 for v in (data-np.mean(data, axis=0))]
    r2 = [np.linalg.norm(data[i]-data[j])**2 for i,j in itertools.combinations(range(N), 2)]
    
    print np.sum(r2) / np.sum(r1)
