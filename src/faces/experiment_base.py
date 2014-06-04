import numpy as np
#import pickle
import scipy.spatial.distance

#from matplotlib import pyplot
#from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import mdp
import fpp


def experiment(algorithm, k, iterations, reduce_variance=False, whitening=False, normalize_std=False, additional_noise_dim=0, additional_noise_std=0):

    # parameters
    normalized_objective = True
    additive_noise = 0
    
    # load data file
    faces_raw = np.load('faces.npy')
    faces = np.array(faces_raw, copy=True)
    print faces_raw.shape
    
    # PCA
    if reduce_variance:
        pca = mdp.nodes.PCANode(output_dim=0.99)
        pca.train(faces)
        faces = pca.execute(faces)
        print 'dim after pca:', faces.shape
    N, D = faces.shape
        
    #std = np.std(faces, axis=0)
    #pyplot.hist(std)
        
    # additive noise
    if additive_noise > 0:
        faces += additive_noise * np.random.randn(N, D)
        
    # additional dimensions
    if additional_noise_dim > 0:
        noise = additional_noise_std * np.random.randn(N, additional_noise_dim)
        faces = np.hstack([faces, noise])
    
    # whiten data
    if whitening:
        whitening = mdp.nodes.WhiteningNode(reduce=True)
        whitening.train(faces)
        faces = whitening.execute(faces)
        print 'dim after whitening:', faces.shape

    # normalize component wise
    if normalize_std:
        faces -= np.mean(faces, axis=0)
        faces /= np.std(faces, axis=0)

    # model
    if algorithm == 'random':
        node = fpp.RandomProjection(output_dim=2)
    elif algorithm == 'sfa':
        node = mdp.nodes.SFANode(output_dim=2)
    elif algorithm == 'lpp':
        node = fpp.LPP(output_dim=2,
                   k=k,
                   normalized_objective=normalized_objective)
    elif algorithm == 'fpp':
        node = fpp.FPP(output_dim=2,
                   k=k,
                   iterations=iterations,
                   iteration_dim=4,
                   minimize_variance=False,
                   normalized_objective=normalized_objective)
    else:
        print 'unexpected algorithm', algorithm
        assert False
        

    # training
    node.train(faces)
    node.stop_training()
    result = node.execute(faces)
    
    
    # scale result to variance 1
    cov = np.cov(result.T)
    E, U = np.linalg.eigh(cov)
    W = U.dot(np.diag(1./np.sqrt((np.sum(E)*np.ones(2)))).dot(U.T))
    result = result.dot(W)
    
    #cov = np.cov(result.T)
    #E, U = np.linalg.eigh(cov)
    #assert np.abs(np.sum(E)-1) < 1e-6
    
    return result



def performance_fpp(projected_data, k, baseline_result):
    
    N = projected_data.shape[0]
    
    distances = scipy.spatial.distance.pdist(baseline_result)
    distances = scipy.spatial.distance.squareform(distances)
    neighbors = [np.array(np.argsort(distances[i])[1:k+1], dtype=int) for i in range(N-1)]
    
    performance = 0
    for t, neighborhood in enumerate(neighbors):
        neighborhood = np.setdiff1d(neighborhood, np.array([N-1]), assume_unique=True)
        if len(neighborhood) == 0:
            continue
        assert len(neighborhood) >= 1
        future = neighborhood + 1
        delta_vectors = projected_data[future] - projected_data[t+1]
        deltas = np.sqrt(np.diag(delta_vectors.dot(delta_vectors.T)))
        assert np.all(np.isfinite(delta_vectors))
        assert len(deltas) == len(neighborhood)
        performance += np.mean(deltas)
    performance /= N-1
    return performance



def performance_lpp(projected_data, k, baseline_result):
    
    N = projected_data.shape[0]
    
    distances = scipy.spatial.distance.pdist(baseline_result)
    distances = scipy.spatial.distance.squareform(distances)
    neighbors = [np.array(np.argsort(distances[i])[1:k+1], dtype=int) for i in range(N)]
    
    performance = 0
    for t, neighborhood in enumerate(neighbors):
        assert len(neighborhood) >= 1
        delta_vectors = projected_data[neighborhood] - projected_data[t]
        deltas = np.sqrt(np.diag(delta_vectors.dot(delta_vectors.T)))
        assert np.all(np.isfinite(delta_vectors))
        assert len(deltas) == len(neighborhood)
        performance += np.mean(deltas)
    performance /= N
    return performance

    