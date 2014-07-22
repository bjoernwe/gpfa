import numpy as np
import pickle
import scipy.linalg
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
        
    # additive noise
    #if additive_noise > 0:
    #    data += additive_noise * np.random.randn(N, D)
        
    # additional dimensions
    if additional_noise_dim > 0:
        noise = np.random.rand(N, additional_noise_dim)
        data = np.hstack([data, noise])
    
    # whiten data
    if whitening:
        whitening = mdp.nodes.WhiteningNode(reduce=True)
        whitening.train(data)
        data = whitening.execute(data)

    # normalize component wise
    #if normalize_std:
    #    faces -= np.mean(faces, axis=0)
    #    faces /= np.std(faces, axis=0)

    # model
    if algorithm == 'random':
        node = fpp.RandomProjection(output_dim=2)
    elif algorithm == 'sfa':
        node = mdp.nodes.SFANode(output_dim=2)
    elif algorithm == 'lpp':
        node = fpp.LPP(output_dim=2,
                   k=k,
                   normalized_objective=True)
    elif algorithm == 'gpfa':
        node = fpp.FPP(output_dim=2,
                   k=k,
                   iterations=iterations,
                   iteration_dim=iteration_dim,
                   variance_graph=variance_graph,
                   normalized_objective=True)
    else:
        print 'unexpected algorithm', algorithm
        assert False
        

    # training
    node.train(data)
    node.stop_training()
    results = node.execute(data)
    
    # scale results to sum(E) = 1
    cov = np.cov(results.T)
    E, U = np.linalg.eigh(cov)
    W = U.dot(np.diag(1./np.sqrt((np.sum(E)*np.ones(2)))).dot(U.T))
    results = results.dot(W)
    assert np.abs(1 - np.sum(np.linalg.eigh(np.cov(results.T))[0])) < 1e-6
    
    return results


def calc_neighbor_list(data):
    N, _ = data.shape
    distances = scipy.spatial.distance.pdist(data)
    distances = scipy.spatial.distance.squareform(distances)
    return [np.array(np.argsort(distances[i])[1:k+1], dtype=int) for i in range(N)]

    
def variance_of_future(projected_data, neighbor_list):
    
    N = projected_data.shape[0]
    
    cov_matrix = mdp.utils.CovarianceMatrix()
    
    #performance = 0
    #number_of_edges = 0
    for t, neighborhood in enumerate(neighbor_list[:-1]):
        neighborhood = np.setdiff1d(neighborhood, np.array([N-1]), assume_unique=True)
        if len(neighborhood) == 0:
            continue
        future = neighborhood + 1
        delta_vectors = projected_data[future] - projected_data[t+1]
        cov_matrix.update(delta_vectors)
        #squared_distances = np.diag(delta_vectors.dot(delta_vectors.T))
        #assert len(neighborhood) >= 1
        #assert np.all(np.isfinite(delta_vectors))
        #assert len(squared_distances) == len(neighborhood)
        #performance += np.sum(squared_distances)
        #number_of_edges += len(neighborhood)
    #performance = np.sqrt(performance / (number_of_edges - 1))
    covariance, mean, number_of_edges = cov_matrix.fix()
    performance = np.sum(np.linalg.eigh(covariance)[0])
    #print covariance
    #print np.linalg.eigh(covariance)
    return performance



def variance_of_neighbors(projected_data, neighbor_list):
    
    N = projected_data.shape[0]
    
    cov_matrix = mdp.utils.CovarianceMatrix()
    
    #performance = 0
    #number_of_edges = 0
    for t, neighborhood in enumerate(neighbor_list):
        if len(neighborhood) == 0:
            continue
        delta_vectors = projected_data[neighborhood] - projected_data[t]
        cov_matrix.update(delta_vectors)
        #squared_distances = np.diag(delta_vectors.dot(delta_vectors.T))
        #assert len(neighborhood) >= 1
        #assert np.all(np.isfinite(delta_vectors))
        #assert len(squared_distances) == len(neighborhood)
        #performance += np.sum(squared_distances)
        #number_of_edges += len(neighborhood)
    #performance = np.sqrt(performance / (number_of_edges - 1))
    covariance, mean, number_of_edges = cov_matrix.fix()
    performance = np.sum(np.linalg.eigh(covariance)[0])
    #print covariance
    #print np.linalg.eigh(covariance)
    return performance


def get_handle_lists(l):
    """
    Found this here:
    http://nbviewer.ipython.org/gist/leejjoon/5603703
    """
    tree = l._legend_box.get_children()[1]

    for column in tree.get_children():
        for row in column.get_children():
            yield row.get_children()[0].get_children()
    
    
if __name__ == '__main__':
    
    k = 5
    steps = 2000
    variance_graph = False
    do_pca = False
    trials = 1
    noisy_dims = 20
    additive_noise = 0
    iter_dims = [None, 2, 5]#, 20, 50, 100]
    iterations_list = [1, 2, 3, 4, 5]#, 7, 10]#, 15, 20]#, 50, 100, 200, 500]
    noisy_dims_list = [0, 1]#, 2, 5, 10]#, 20, 50, 100, 150]#, 200, 250, 300]
    
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
    l = pyplot.legend([marker_lpp, marker_gpfa_1, marker_gpfa_2, line_neighbors, line_future], ['LPP', 'gPFA (variant 1)', 'gPFA (variant 2)', 'dist. of neighb.', 'dist. of neighb. in next step'])
    handles_list = list(get_handle_lists(l))
    handles = handles_list[0]
    print [method for method in dir(handles[0]) if callable(getattr(handles[0], method))]
    #print handles[0].set_markerfacecolor('b')
    #print handles[1].set_markerfacecolor('white')
    #handles[1].set_markerfacecolor(['none', 'full'])
    pyplot.show()

#     results = {}
#     for d, dim in enumerate(iter_dims):
#         results[dim] = {}    
#         results[dim]['lpp'] = np.zeros((len(iterations_list),trials))
#         results[dim]['fpp'] = np.zeros((len(iterations_list),trials))
#         for i, iterations in enumerate(iterations_list):
#             print dim, 'x', iterations
#             result_without_noise = experiment(steps, algorithm='fpp', k=k, variance_graph=variance_graph, iterations=iterations, iteration_dim=dim, reduce_variance=do_pca, whitening=True, additional_noise_dim=0)
#             for r in range(trials):
#                 result_with_noise = experiment(steps, algorithm='fpp', k=k, variance_graph=variance_graph, iterations=iterations, iteration_dim=dim, reduce_variance=do_pca, whitening=True, additional_noise_dim=noisy_dims)
#                 results[dim]['lpp'][i,r] = variance_of_neighbors(projected_data=result_with_noise, k=k, baseline_result=result_without_noise)
#                 results[dim]['fpp'][i,r] = variance_of_future(projected_data=result_with_noise, k=k, baseline_result=result_without_noise)
# 
#         pyplot.subplot(1,2,1)
#         pyplot.errorbar(x=iterations_list, y=np.mean(results[dim]['lpp'], axis=1), yerr=np.std(results[dim]['lpp'], axis=1))
#         pyplot.subplot(1,2,2)
#         pyplot.errorbar(x=iterations_list, y=np.mean(results[dim]['fpp'], axis=1), yerr=np.std(results[dim]['fpp'], axis=1))
#     
#     pickle.dump(results, open('results_iterations.pkl', 'wb'))
#     
#     pyplot.subplot(1, 2, 1)
#     pyplot.title('neigborhood')
#     pyplot.legend(iter_dims)
#     pyplot.subplot(1, 2, 2)
#     pyplot.title('future')
#     pyplot.legend(iter_dims)
#     pyplot.show()
    