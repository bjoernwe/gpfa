import numpy as np
import pickle
import scipy.spatial.distance

#from matplotlib import pyplot
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

#import mdp

from fpp import *


def experiment(algorithm, k, iterations, additional_noise_dim, additional_noise_std):

    # parameters
    reduce_variance = True
    whitening = False
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

    # model
    if algorithm == 'sfa':
        node = mdp.nodes.SFANode(output_dim=2)
    elif algorithm == 'lpp':
        node = LPP(output_dim=2,
                   k=k,
                   normalized_objective=normalized_objective)
    elif algorithm == 'fpp':
        node = FPP(output_dim=2,
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
        deltas = projected_data[future] - projected_data[t+1]
        assert np.all(np.isfinite(deltas))
        performance += np.mean(np.sqrt(np.diag(deltas.dot(deltas.T))))
    performance /= N-1
    return performance



def performance_lpp(projected_data, k, baseline_result):
    
    N = projected_data.shape[0]
    
    distances = scipy.spatial.distance.pdist(baseline_result)
    distances = scipy.spatial.distance.squareform(distances)
    neighbors = [np.array(np.argsort(distances[i])[1:k+1], dtype=int) for i in range(N-1)]
    
    performance = 0
    for t, neighborhood in enumerate(neighbors):
        assert len(neighborhood) >= 1
        deltas = projected_data[neighborhood] - projected_data[t]
        assert np.all(np.isfinite(deltas))
        performance += np.mean(np.sqrt(np.diag(deltas.dot(deltas.T))))
    performance /= N-1
    return performance



if __name__ == '__main__':
    
    k = 3
    iterations = 5
    trials = 3
    dimensions = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]

    baseline_result = {}
    for algorithm in ['sfa', 'lpp', 'fpp']:
    #baseline_result['sfa'] = experiment(algorithm='sfa', k=k, iterations=0, additional_noise_dim=0, additional_noise_std=0)
        baseline_result[algorithm] = experiment(algorithm=algorithm, k=k, iterations=iterations, additional_noise_dim=0, additional_noise_std=0)
    #baseline_result['fpp'] = experiment(algorithm='fpp', k=k, iterations=iterations, additional_noise_dim=0, additional_noise_std=0)

    result = {}
    
    for a, algorithm in enumerate(['sfa', 'lpp', 'fpp']):

        print algorithm
        
        result[algorithm] = {}
        result[algorithm]['sfa'] = np.zeros((len(dimensions),trials))
        result[algorithm]['lpp'] = np.zeros((len(dimensions),trials))
        result[algorithm]['fpp'] = np.zeros((len(dimensions),trials))
        
        for i, dim in enumerate(dimensions):
            print dim
            for r in range(trials):
                tmp_result = experiment(algorithm=algorithm, k=k, iterations=iterations, additional_noise_dim=dim, additional_noise_std=200)
                result[algorithm]['lpp'][i,r] = performance_lpp(projected_data=tmp_result, k=k, baseline_result=baseline_result[algorithm])
                result[algorithm]['fpp'][i,r] = performance_fpp(projected_data=tmp_result, k=k, baseline_result=baseline_result[algorithm])

        pyplot.subplot(1, 3, a+1)
        pyplot.title(algorithm)
        pyplot.errorbar(x=dimensions, y=np.mean(result[algorithm]['lpp'], axis=1), yerr=np.std(result[algorithm]['lpp'], axis=1))
        pyplot.errorbar(x=dimensions, y=np.mean(result[algorithm]['fpp'], axis=1), yerr=np.std(result[algorithm]['fpp'], axis=1))
        pyplot.legend(['neighborhood', 'future'])
        
    pickle.dump(baseline_result, open('baseline_result.pkl', 'wb'))
    pickle.dump(result, open('result.pkl', 'wb'))
    pyplot.show()
    