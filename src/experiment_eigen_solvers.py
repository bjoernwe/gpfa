import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import time
    

if __name__ == '__main__':
    
    N = 1000
    
    # data
    A = np.random.random(size=(N, N))
    B = A + A.T


    print "\nsmallest Eigenvector:\n"

    # SciPy    
#     start = time.time()
#     E, U = scipy.sparse.linalg.eigs(A, k=1, which='SM')
#     end = time.time()
#     print "scipy.sparse.linalg.eigs: %fs (%f)" % (end - start, E[0].real)

    # SciPy    
#     start = time.time()
#     E, U = scipy.sparse.linalg.eigs(A, k=1, which='SR')
#     end = time.time()
#     print "scipy.sparse.linalg.eigs: %fs (%f)" % (end - start, E[0].real)


    print "\nlargest Eigenvector:\n"

    # SciPy    
    start = time.time()
    E, U = scipy.sparse.linalg.eigs(A, k=1, which='LM')
    end = time.time()
    print "scipy.sparse.linalg.eigs: %fs (%f)" % (end - start, E[0].real)

    # SciPy    
    start = time.time()
    E, U = scipy.sparse.linalg.eigs(A, k=1, which='LR')
    end = time.time()
    print "scipy.sparse.linalg.eigs: %fs (%f)" % (end - start, E[0].real)


    print "\nsmallest Eigenvector (symmetric):\n"

    # SciPy    
    start = time.time()
    E, U = scipy.linalg.eigh(B, eigvals=(0, 0))
    end = time.time()
    print "scipy.linalg.eigh: %fs (%f)" % (end - start, E[0])

    # SciPy    
#     start = time.time()
#     E, U = scipy.sparse.linalg.eigsh(B, k=1, which='SM')
#     end = time.time()
#     print "scipy.sparse.linalg.eigsh: %fs (%f)" % (end - start, E[0])

    # SciPy    
#     start = time.time()
#     E, U = scipy.sparse.linalg.eigsh(B, k=1, which='SA')
#     end = time.time()
#     print "scipy.sparse.linalg.eigsh: %fs (%f)" % (end - start, E[0])


    print "\nlargest Eigenvector (symmetric):\n"
    
    # SciPy
    start = time.time()
    E, U = scipy.linalg.eigh(B, eigvals=(N-1, N-1))
    end = time.time()
    print "scipy.linalg.eigh: %fs (%f)" % (end - start, E[0])

    # SciPy    
    start = time.time()
    E, U = scipy.sparse.linalg.eigsh(B, k=1, which='LM')
    end = time.time()
    print "scipy.sparse.linalg.eigsh: %fs (%f)" % (end - start, E[0])

    # SciPy    
    start = time.time()
    E, U = scipy.sparse.linalg.eigsh(B, k=1, which='LA')
    end = time.time()
    print "scipy.sparse.linalg.eigsh: %fs (%f)" % (end - start, E[0])


    print "\nfull Eigendecomposition (symmetric):\n"
    
    # NumPy symmetric
    start = time.time()
    E, U = np.linalg.eigh(A)
    end = time.time()
    print "np.linalg.eigh: %fs" % (end - start)
    
    # SciPy symmetric
    start = time.time()
    E, U = scipy.linalg.eigh(B)
    end = time.time()
    print "scipy.linalg.eigh: %fs" % (end - start)

    
    print "\nfull Eigendecomposition:\n"
    
    # NumPy
    start = time.time()
    E, U = np.linalg.eig(A)
    end = time.time()
    print "np.linalg.eig: %fs" % (end - start)
    
    # SciPy
    start = time.time()
    E, U = scipy.linalg.eig(B)
    end = time.time()
    print "scipy.linalg.eig: %fs" % (end - start)
    
            