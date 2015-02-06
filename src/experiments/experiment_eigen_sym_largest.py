import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import time
    

if __name__ == '__main__':
    
    N = 1000
    
    # data
    A = np.random.random(size=(N, N))
    X = np.random.randn(3*N, N)
    C = X.T.dot(X)
    
    d = np.log10(np.array(range(N)) + 2)
    D = scipy.sparse.dia_matrix((d, 0), shape=(N, N)).tocsc()
    Dinv = scipy.sparse.dia_matrix((1./d, 0), shape=(N, N)).tocsc()


    print "\nlargest Eigenvector (symmetric):\n"
    
    # SciPy
    start = time.time()
    E, U = scipy.linalg.eigh(C, eigvals=(N-1, N-1))
    end = time.time()
    u0 = U[:,0]
    print "scipy.linalg.eigh: %fs (%f)" % (end - start, E[0])

    # SciPy    
    start = time.time()
    E, U = scipy.sparse.linalg.eigsh(C, k=1, which='LM')
    end = time.time()
    print "scipy.sparse.linalg.eigsh: %fs (%f)" % (end - start, E[0])

    # SciPy    
    start = time.time()
    E, U = scipy.sparse.linalg.eigsh(C, k=1, which='LA')
    end = time.time()
    print "scipy.sparse.linalg.eigsh: %fs (%f)" % (end - start, E[0])


    print "\nlargest general Eigenvector (symmetric):\n"
    
    # SciPy
    start = time.time()
    E, U = scipy.linalg.eigh(C, b=D.toarray(), eigvals=(N-1, N-1))
    end = time.time()
    print "scipy.linalg.eigh: %fs (%f)" % (end - start, E[0])

    # SciPy    
    start = time.time()
    E, U = scipy.sparse.linalg.eigsh(C, M=D, k=1, which='LM')
    end = time.time()
    print "scipy.sparse.linalg.eigsh: %fs (%f)" % (end - start, E[0])

    # SciPy    
    start = time.time()
    E, U = scipy.sparse.linalg.eigsh(C, M=D, k=1, which='LA')
    end = time.time()
    print "scipy.sparse.linalg.eigsh: %fs (%f)" % (end - start, E[0])


    print "\nlargest smart general Eigenvector (symmetric):\n"
    
    # SciPy
    start = time.time()
    E, U = scipy.linalg.eigh(Dinv.dot(C), eigvals=(N-1, N-1))
    end = time.time()
    print "scipy.linalg.eigh: %fs (%f)" % (end - start, E[0])

    # SciPy    
    start = time.time()
    E, U = scipy.sparse.linalg.eigsh(Dinv.dot(C), k=1, which='LM')
    end = time.time()
    print "scipy.sparse.linalg.eigsh: %fs (%f)" % (end - start, E[0])

    # SciPy    
    start = time.time()
    E, U = scipy.sparse.linalg.eigsh(Dinv.dot(C), k=1, which='LA')
    end = time.time()
    print "scipy.sparse.linalg.eigsh: %fs (%f)" % (end - start, E[0])

    # SciPy    
    start = time.time()
    E, U = scipy.sparse.linalg.eigsh(D-C, M=D, k=1, which='LM')
    end = time.time()
    print "scipy.sparse.linalg.eigsh: %fs (%f)" % (end - start, 1-E[0])
