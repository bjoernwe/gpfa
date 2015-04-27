import itertools
import numpy as np
import scipy.spatial.distance

from matplotlib import pyplot

import mdp

import envs.env_cube
import gpfa


def experiment(data, algorithm, k, variance_graph, iterations, iteration_dim, constraint_optimization, reduce_variance=False, whitening=False, additional_noise_dim=0):

    # PCA
    if reduce_variance:
        pca = mdp.nodes.PCANode(output_dim=0.99)
        pca.train(data)
        data = pca.execute(data)
        print 'dim after pca:', data.shape
    N, _ = data.shape
        
    # additional dimensions
    if additional_noise_dim > 0:
        noise = np.random.rand(N, additional_noise_dim)
        data = np.hstack([data, noise])
    
    # whiten data
    if whitening:
        whitening = mdp.nodes.WhiteningNode(reduce=True)
        whitening.train(data)
        data = whitening.execute(data)

    # model
    if algorithm == 'random':
        node = gpfa.RandomProjection(output_dim=2)
    elif algorithm == 'sfa':
        node = mdp.nodes.SFANode(output_dim=2)
    elif algorithm == 'lpp':
        node = gpfa.LPP(output_dim=2, k=k)
    elif algorithm == 'gpfa':
        node = gpfa.gPFA(output_dim=2,
                    k=k,
                    iterations=iterations,
                    iteration_dim=iteration_dim,
                    variance_graph=variance_graph,
                    constraint_optimization=constraint_optimization)
    else:
        print 'unexpected algorithm', algorithm
        assert False
        
    # training
    node.train(data)
    node.stop_training()
    results = node.execute(data)
    
    # scale result to variance 1
    cov = np.cov(results.T)
    results /= np.sqrt(np.trace(cov))
    assert np.abs(1 - np.sum(np.linalg.eigh(np.cov(results.T))[0])) < 1e-6
    return results


def calc_neighbor_list(data):
    N, _ = data.shape
    distances = scipy.spatial.distance.pdist(data)
    distances = scipy.spatial.distance.squareform(distances)
    return [np.array(np.argsort(distances[i])[:k+1], dtype=int) for i in range(N)]

    
def distances_of_neighbors(projected_data, neighbor_list):
    
    cov_matrix = mdp.utils.CovarianceMatrix(bias=True)
    
    for t, neighborhood in enumerate(neighbor_list):
        if len(neighborhood) == 0:
            continue
        delta_vectors = projected_data[neighborhood] - projected_data[t]
        cov_matrix.update(delta_vectors)

    covariance, _, _ = cov_matrix.fix()
    performance = np.trace(covariance)
    return performance


def variance_of_neighbors(projected_data, neighbor_list):
    
    cov_matrix = mdp.utils.CovarianceMatrix(bias=True)
    
    for neighborhood in neighbor_list:
        if len(neighborhood) == 0:
            continue
        combinations = np.array(list(itertools.combinations(neighborhood, 2)))
        delta_vectors = projected_data[combinations[:,0]] - projected_data[combinations[:,1]]
        cov_matrix.update(delta_vectors)

    covariance, _, _ = cov_matrix.fix()
    performance = np.trace(covariance)
    return performance


def distances_of_future(projected_data, neighbor_list):

    N = projected_data.shape[0]
    cov_matrix = mdp.utils.CovarianceMatrix(bias=True)
    
    for t, neighborhood in enumerate(neighbor_list[:-1]):
        neighborhood = np.setdiff1d(neighborhood, np.array([N-1]), assume_unique=True)
        if len(neighborhood) == 0:
            continue
        future = neighborhood + 1
        delta_vectors = projected_data[future] - projected_data[t+1]
        cov_matrix.update(delta_vectors)
        
    covariance, _, _ = cov_matrix.fix()
    performance = np.trace(covariance)
    return performance


def variance_of_future(projected_data, neighbor_list):

    N = projected_data.shape[0]
    cov_matrix = mdp.utils.CovarianceMatrix(bias=True)
    
    for neighborhood in neighbor_list[:-1]:
        neighborhood = np.setdiff1d(neighborhood, np.array([N-1]), assume_unique=True)
        if len(neighborhood) == 0:
            continue
        future = neighborhood + 1
        combinations = np.array(list(itertools.combinations(future, 2)))
        delta_vectors = projected_data[combinations[:,0]] - projected_data[combinations[:,1]]
        cov_matrix.update(delta_vectors)
        
    covariance, _, _ = cov_matrix.fix()
    performance = np.trace(covariance)
    return performance


if __name__ == '__main__':
    
    k = 5
    steps = 5000
    trials = 20
    whitening = True
    constraint_optimization = True
    iteration_list = [1, 5, 10, 20, 50]
    iteration_color_list = [[i/(1.*len(iteration_list))] * 3 for i in range(len(iteration_list))]
    iteration_marker_list = ['^', 'v', '<', '>', 's']
    noisy_dims_list = [0, 10, 20, 50, 100, 200, 300, 400, 500, 600]
    
    results = {}
    
    for it, iterations in enumerate(iteration_list):

        results[iterations] = {}
        results[iterations]['neighbors'] = np.zeros((len(noisy_dims_list),trials))
        results[iterations]['future'] = np.zeros((len(noisy_dims_list),trials))
        
        for r in range(trials):
                
            env = envs.env_cube.EnvCube()
            data, _, _ = env.do_random_steps(num_steps=steps)
            
            # baseline
            result = experiment(data=data, 
                                algorithm='gpfa', 
                                k=k, 
                                variance_graph=False, 
                                iterations=iterations, 
                                iteration_dim=2, 
                                reduce_variance=False, 
                                constraint_optimization=constraint_optimization, 
                                whitening=whitening,
                                additional_noise_dim=0)
            
            neighbor_list = calc_neighbor_list(result)
            
            for d, noisy_dims in enumerate(noisy_dims_list):
    
                print noisy_dims, 'noisy dimensions, trial', (r+1)
                
                result = experiment(data=data, 
                                    algorithm='gpfa', 
                                    k=k, 
                                    variance_graph=False, 
                                    iterations=iterations, 
                                    iteration_dim=2, 
                                    reduce_variance=False, 
                                    constraint_optimization=constraint_optimization, 
                                    whitening=whitening,
                                    additional_noise_dim=noisy_dims)
                
                results[iterations]['neighbors'][d,r] = distances_of_neighbors(result, neighbor_list)
                results[iterations]['future'][d,r] = distances_of_future(result, neighbor_list)
        
        #facecolor = 'white' if g == 1 else None
        #color = [it/(1.*len(iteration_list)) for _ in range(3)]
        #marker = 'o' if a == 1 else '^'
        
        pyplot.errorbar(x=noisy_dims_list, 
                        y=np.mean(results[iterations]['neighbors'], axis=1), 
                        yerr=np.std(results[iterations]['neighbors'], axis=1), 
                        markersize=10, 
                        marker=iteration_marker_list[it], 
                        markerfacecolor='white', 
                        linestyle='-',
                        color='b')#iteration_color_list[it])
        pyplot.errorbar(x=noisy_dims_list, 
                        y=np.mean(results[iterations]['future'], axis=1), 
                        yerr=np.std(results[iterations]['future'], axis=1), 
                        markersize=10, 
                        marker=iteration_marker_list[it], 
                        markerfacecolor='white', 
                        linestyle='--',
                        color='b')#iteration_color_list[it])

    legend_lines = []
    for i in range(len(iteration_list)):
        legend_lines.append(pyplot.Line2D((0,1),(0,0), color='b', linestyle='', marker=iteration_marker_list[i], markersize=10, markerfacecolor='white'))
    #marker_lpp  = pyplot.Line2D((0,1),(0,0), color='black', linestyle='', marker='^', markersize=8)
    #marker_gpfa_1 = pyplot.Line2D((0,1),(0,0), color='black', linestyle='', marker='o', markersize=8)
    #marker_gpfa_2 = pyplot.Line2D((0,1),(0,0), color='black', linestyle='', marker='o', markersize=8, markerfacecolor='none')
    legend_lines.append(pyplot.Line2D((0,1),(0,0), color='black', linestyle='-'))
    legend_lines.append(pyplot.Line2D((0,1),(0,0), color='black', linestyle='--'))
    pyplot.legend(legend_lines, ['gPFA (1 iteration)'] + ['gPFA (%d iterations)' % it for it in iteration_list[1:]] + ['neighbors', 'successors'], loc=2)
    pyplot.xlabel('number of noisy dimensions')
    pyplot.ylabel('mean squared distances')
    pyplot.show()
