import numpy as np
import pickle
import scipy.spatial.distance

from matplotlib import pyplot

import mdp

import fpp


def experiment(algorithm, k, variance_graph, iterations_list, iteration_dim, reduce_variance=False, whitening=False, normalize_std=False, additional_noise_dim=0, additional_noise_std=0, additive_noise=0):

    # parameters
    normalized_objective = True
    
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
                   iterations_list=iterations_list,
                   iteration_dim=iteration_dim,
                   variance_graph=variance_graph,
                   normalized_objective=normalized_objective)
    else:
        print 'unexpected algorithm', algorithm
        assert False
        

    # training
    node.train(faces)
    node.stop_training()
    results = node.execute(faces)
    
    # scale results to sum(E) = 1
    cov = np.cov(results.T)
    E, U = np.linalg.eigh(cov)
    W = U.dot(np.diag(1./np.sqrt((np.sum(E)*np.ones(2)))).dot(U.T))
    results = results.dot(W)
    
    #cov = np.cov(results.T)
    #E, U = np.linalg.eigh(cov)
    #assert np.abs(np.sum(E)-1) < 1e-6
    
    return results


def whiten_data(data):
    whitening = mdp.nodes.WhiteningNode()
    whitening.train(data)
    return whitening.execute(data)


def performance_fpp(projected_data, k, baseline_result):
    
    projected_data = whiten_data(projected_data)
    baseline_result = whiten_data(baseline_result)
    
    N = projected_data.shape[0]
    
    distances = scipy.spatial.distance.pdist(baseline_result)
    distances = scipy.spatial.distance.squareform(distances)
    neighbors = [np.array(np.argsort(distances[i])[1:k+1], dtype=int) for i in range(N-1)]
    
    performance = 0
    number_of_edges = 0
    for t, neighborhood in enumerate(neighbors):
        neighborhood = np.setdiff1d(neighborhood, np.array([N-1]), assume_unique=True)
        if len(neighborhood) == 0:
            continue
        future = neighborhood + 1
        delta_vectors = projected_data[future] - projected_data[t+1]
        squared_distances = np.diag(delta_vectors.dot(delta_vectors.T))
        assert len(neighborhood) >= 1
        assert np.all(np.isfinite(delta_vectors))
        assert len(squared_distances) == len(neighborhood)
        performance += np.sum(squared_distances)
        number_of_edges += len(neighborhood)
    performance = np.sqrt(performance / (number_of_edges - 1))
    return performance



def performance_lpp(projected_data, k, baseline_result):
    
    projected_data = whiten_data(projected_data)
    baseline_result = whiten_data(baseline_result)
    
    N = projected_data.shape[0]
    
    distances = scipy.spatial.distance.pdist(baseline_result)
    distances = scipy.spatial.distance.squareform(distances)
    neighbors = [np.array(np.argsort(distances[i])[1:k+1], dtype=int) for i in range(N)]
    
    performance = 0
    number_of_edges = 0
    for t, neighborhood in enumerate(neighbors):
        neighborhood = np.setdiff1d(neighborhood, np.array([N-1]), assume_unique=True)
        if len(neighborhood) == 0:
            continue
        delta_vectors = projected_data[neighborhood] - projected_data[t]
        squared_distances = np.diag(delta_vectors.dot(delta_vectors.T))
        assert len(neighborhood) >= 1
        assert np.all(np.isfinite(delta_vectors))
        assert len(squared_distances) == len(neighborhood)
        performance += np.sum(squared_distances)
        number_of_edges += len(neighborhood)
    performance = np.sqrt(performance / (number_of_edges - 1))
    return performance

    
if __name__ == '__main__':
    
    k = 5
    variance_graph = False
    do_pca = True
    trials = 2
    noisy_dims = 20
    additive_noise = 0
    iter_dims = [None, 2, 5, 20, 50, 100]
    iterations_list = [1, 2, 3, 4, 5, 7, 10]#, 15, 20]#, 50, 100, 200, 500]

    results = {}
    for d, dim in enumerate(iter_dims):
        results[dim] = {}    
        results[dim]['lpp'] = np.zeros((len(iterations_list),trials))
        results[dim]['fpp'] = np.zeros((len(iterations_list),trials))
        for i, iterations in enumerate(iterations_list):
            print dim, 'x', iterations
            result_without_noise = experiment(algorithm='fpp', k=k, variance_graph=variance_graph, iterations_list=iterations, iteration_dim=dim, reduce_variance=do_pca, additional_noise_dim=0, additional_noise_std=0, additive_noise=0)
            for r in range(trials):
                result_with_noise = experiment(algorithm='fpp', k=k, variance_graph=variance_graph, iterations_list=iterations, iteration_dim=dim, reduce_variance=do_pca, additional_noise_dim=noisy_dims, additional_noise_std=200, additive_noise=additive_noise)
                results[dim]['lpp'][i,r] = performance_lpp(projected_data=result_with_noise, k=k, baseline_result=result_without_noise)
                results[dim]['fpp'][i,r] = performance_fpp(projected_data=result_with_noise, k=k, baseline_result=result_without_noise)

        pyplot.subplot(1,2,1)
        pyplot.errorbar(x=iterations_list, y=np.mean(results[dim]['lpp'], axis=1), yerr=np.std(results[dim]['lpp'], axis=1))
        pyplot.subplot(1,2,2)
        pyplot.errorbar(x=iterations_list, y=np.mean(results[dim]['fpp'], axis=1), yerr=np.std(results[dim]['fpp'], axis=1))
    
    pickle.dump(results, open('results_iterations.pkl', 'wb'))
    
    pyplot.subplot(1, 2, 1)
    pyplot.title('neigborhood')
    pyplot.legend(iter_dims)
    pyplot.subplot(1, 2, 2)
    pyplot.title('future')
    pyplot.legend(iter_dims)
    pyplot.show()
    