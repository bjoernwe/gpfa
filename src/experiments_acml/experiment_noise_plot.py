import numpy as np
import scipy.spatial.distance

from matplotlib import pyplot

import mdp

import envs.env_cube
import fpp


def experiment(data, algorithm, k, variance_graph, iterations, iteration_dim, reduce_variance=False, whitening=False, additional_noise_dim=0):

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
        node = fpp.RandomProjection(output_dim=2)
    elif algorithm == 'sfa':
        node = mdp.nodes.SFANode(output_dim=2)
    elif algorithm == 'lpp':
        node = fpp.LPP(output_dim=2, k=k)
    elif algorithm == 'gpfa':
        node = fpp.FPP(output_dim=2,
                   k=k,
                   iterations=iterations,
                   iteration_dim=iteration_dim,
                   variance_graph=variance_graph)
    else:
        print 'unexpected algorithm', algorithm
        assert False
        

    # training
    node.train(data)
    node.stop_training()
    results = node.execute(data)
    
    # scale result to variance 1
    cov = np.cov(results.T)
    E, _ = np.linalg.eigh(cov)
    results /= np.sqrt(np.mean(E))
    assert np.abs(1 - np.mean(np.linalg.eigh(np.cov(results.T))[0])) < 1e-6
    
    return results


def calc_neighbor_list(data):
    N, _ = data.shape
    distances = scipy.spatial.distance.pdist(data)
    distances = scipy.spatial.distance.squareform(distances)
    return [np.array(np.argsort(distances[i])[1:k+1], dtype=int) for i in range(N)]

    
def variance_of_future(projected_data, neighbor_list):
    
    N = projected_data.shape[0]
    cov_matrix = mdp.utils.CovarianceMatrix()
    
    for t, neighborhood in enumerate(neighbor_list[:-1]):
        neighborhood = np.setdiff1d(neighborhood, np.array([N-1]), assume_unique=True)
        if len(neighborhood) == 0:
            continue
        future = neighborhood + 1
        delta_vectors = projected_data[future] - projected_data[t+1]
        cov_matrix.update(delta_vectors)
    covariance, _, _ = cov_matrix.fix()
    performance = np.sum(np.linalg.eigh(covariance)[0])
    return performance



def variance_of_neighbors(projected_data, neighbor_list):
    
    cov_matrix = mdp.utils.CovarianceMatrix()
    
    for t, neighborhood in enumerate(neighbor_list):
        if len(neighborhood) == 0:
            continue
        delta_vectors = projected_data[neighborhood] - projected_data[t]
        cov_matrix.update(delta_vectors)
    covariance, _, _ = cov_matrix.fix()
    performance = np.sum(np.linalg.eigh(covariance)[0])
    return performance


if __name__ == '__main__':
    
    k = 5
    steps = 2000
    trials = 2
    noisy_dims_list = [0, 1, 2, 5, 10]#, 20, 50, 100, 200, 300, 400, 500, 600]
    
    results = {}
    
    for a, algorithm in enumerate(['lpp', 'gpfa']):

        results[algorithm] = {}
        
        for g, graph_text in enumerate(['standard_graph', 'variance_graph']):
            
            if algorithm == 'lpp' and graph_text == 'variance_graph':
                continue
                    
            results[algorithm][graph_text] = {}
            results[algorithm][graph_text]['neighbors'] = np.zeros((len(noisy_dims_list),trials))
            results[algorithm][graph_text]['future'] = np.zeros((len(noisy_dims_list),trials))
                
            for r in range(trials):
                    
                env = envs.env_cube.EnvCube()
                data, _, _ = env.do_random_steps(num_steps=steps)
                
                # baseline
                result = experiment(data=data, 
                                    algorithm=algorithm, 
                                    k=k, 
                                    variance_graph=bool(g), 
                                    iterations=5, 
                                    iteration_dim=2, 
                                    reduce_variance=False, 
                                    whitening=True, 
                                    additional_noise_dim=0)
                
                neighbor_list = calc_neighbor_list(result)
                
                for d, noisy_dims in enumerate(noisy_dims_list):
        
                    print noisy_dims, 'noisy dimensions, trial', (r+1)
                    
                    result = experiment(data=data, 
                                        algorithm=algorithm, 
                                        k=k, 
                                        variance_graph=bool(g), 
                                        iterations=5, 
                                        iteration_dim=2, 
                                        reduce_variance=False, 
                                        whitening=True, 
                                        additional_noise_dim=noisy_dims)
                    
                    results[algorithm][graph_text]['neighbors'][d,r] = variance_of_neighbors(result, neighbor_list)
                    results[algorithm][graph_text]['future'][d,r] = variance_of_future(result, neighbor_list)
            
            facecolor = 'white' if g == 1 else None
            color = 'b' if a == 1 else 'r'
            marker = 'o' if a == 1 else '^'
            
            pyplot.errorbar(x=noisy_dims_list, 
                            y=np.mean(results[algorithm][graph_text]['neighbors'], axis=1), 
                            yerr=np.std(results[algorithm][graph_text]['neighbors'], axis=1), 
                            markersize=8, 
                            marker=marker, 
                            markerfacecolor=facecolor, 
                            linestyle='-',
                            color=color)
            pyplot.errorbar(x=noisy_dims_list, 
                            y=np.mean(results[algorithm][graph_text]['future'], axis=1), 
                            yerr=np.std(results[algorithm][graph_text]['future'], axis=1), 
                            markersize=8, marker=marker, 
                            markerfacecolor=facecolor, 
                            linestyle='--',
                            color=color)

    marker_lpp  = pyplot.Line2D((0,1),(0,0), color='black', linestyle='', marker='^', markersize=8)
    marker_gpfa_1 = pyplot.Line2D((0,1),(0,0), color='black', linestyle='', marker='o', markersize=8)
    marker_gpfa_2 = pyplot.Line2D((0,1),(0,0), color='black', linestyle='', marker='o', markersize=8, markerfacecolor='none')
    line_neighbors = pyplot.Line2D((0,1),(0,0), color='black', linestyle='-')
    line_future = pyplot.Line2D((0,1),(0,0), color='black', linestyle='--')
    pyplot.legend([marker_lpp, marker_gpfa_1, marker_gpfa_2, line_neighbors, line_future], ['LPP', 'gPFA (variant 1)', 'gPFA (variant 2)', 'dist. of neighb.', 'dist. of neighb. in next step'], loc=4)
    pyplot.xlabel('number of noisy dimension')
    pyplot.ylabel('distance')
    pyplot.show()
