import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import time
    

class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print 'elapsed time: %f ms' % self.msecs
            
            
            
if __name__ == '__main__':
    
    D = 1000
    N = 10000
    
    # data
    data = np.random.random(size=(N, D))
    A = np.dot(data.T, data)


    print "\nsmallest Eigenvector:\n"

    # SciPy    
    with Timer() as t:
        E, U = scipy.sparse.linalg.eigs(A, k=1, which='SM')
    print "scipy.sparse.linalg.eigs: %fs (%f)" % (t.msecs, E[0].real)

    # SciPy    
    with Timer() as t:
        E, U = scipy.sparse.linalg.eigs(A, k=1, which='SR')
    print "scipy.sparse.linalg.eigs: %fs (%f)" % (t.msecs, E[0].real)


    print "\nsmallest Eigenvector (symmetric):\n"

    # SciPy    
    with Timer() as t:
        E, U = scipy.linalg.eigh(A, eigvals=(0, 0))
    print "scipy.linalg.eigh: %fs (%f)" % (t.msecs, E[0])

    # SciPy    
#     with Timer() as t:
#         E, U = scipy.sparse.linalg.eigsh(A, k=1, which='SM')
#     print "scipy.sparse.linalg.eigsh: %fs (%f)" % (t.msecs, E[0])

    # SciPy    
#     with Timer() as t:
#         E, U = scipy.sparse.linalg.eigsh(A, k=1, which='SA')
#     print "scipy.sparse.linalg.eigsh: %fs (%f)" % (t.msecs, E[0])


    print "\nfull Eigendecomposition (symmetric):\n"
    
    # NumPy symmetric
    with Timer() as t:
        E, U = np.linalg.eigh(A)
    print "np.linalg.eigh: %fs" % (t.msecs)
    
    # SciPy symmetric
    with Timer() as t:
        E, U = scipy.linalg.eigh(A)
    print "scipy.linalg.eigh: %fs" % (t.msecs)

    
    print "\nfull Eigendecomposition:\n"
    
    # NumPy
    with Timer() as t:
        E, U = np.linalg.eig(A)
    idx = np.argsort(E.real)
    print "np.linalg.eig: %fs (%f+%fj)" % (t.msecs, E[idx[0]].real, E[idx[0]].imag)
    
    # SciPy
    with Timer() as t:
        E, U = scipy.linalg.eig(A)
    idx = np.argsort(E.real)
    print "scipy.linalg.eig: %fs (%f+%fj)" % (t.msecs, E[idx[0]].real, E[idx[0]].imag)
    
            