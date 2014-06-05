import numpy as np
import pickle

from matplotlib import pyplot

import experiment_base as eb


if __name__ == '__main__':
    
    k = 5
    minimize_variance = False
    do_pca = True
    trials = 2
    noisy_dims = 20
    additive_noise = 0
    iter_dims = [None, 2, 5, 20, 50, 100]
    iterations = [1, 2, 3, 4, 5, 7, 10]#, 15, 20]#, 50, 100, 200, 500]

    results = {}
    for d, dim in enumerate(iter_dims):
        results[dim] = {}    
        results[dim]['lpp'] = np.zeros((len(iterations),trials))
        results[dim]['fpp'] = np.zeros((len(iterations),trials))
        for i, iter in enumerate(iterations):
            print dim, 'x', iter
            result_without_noise = eb.experiment(algorithm='fpp', k=k, minimize_variance=minimize_variance, iterations=iter, iteration_dim=dim, reduce_variance=do_pca, additional_noise_dim=0, additional_noise_std=0, additive_noise=0)
            for r in range(trials):
                result_with_noise    = eb.experiment(algorithm='fpp', k=k, minimize_variance=minimize_variance, iterations=iter, iteration_dim=dim, reduce_variance=do_pca, additional_noise_dim=noisy_dims, additional_noise_std=200, additive_noise=additive_noise)
                results[dim]['lpp'][i,r] = eb.performance_lpp(projected_data=result_with_noise, k=k, baseline_result=result_without_noise)
                results[dim]['fpp'][i,r] = eb.performance_fpp(projected_data=result_with_noise, k=k, baseline_result=result_without_noise)

        pyplot.subplot(1,2,1)
        pyplot.errorbar(x=iterations, y=np.mean(results[dim]['lpp'], axis=1), yerr=np.std(results[dim]['lpp'], axis=1))
        pyplot.subplot(1,2,2)
        pyplot.errorbar(x=iterations, y=np.mean(results[dim]['fpp'], axis=1), yerr=np.std(results[dim]['fpp'], axis=1))
    
    pickle.dump(results, open('results_iterations.pkl', 'wb'))
    
    pyplot.subplot(1, 2, 1)
    pyplot.title('neigborhood')
    pyplot.legend(iter_dims)
    pyplot.subplot(1, 2, 2)
    pyplot.title('future')
    pyplot.legend(iter_dims)
    pyplot.show()
    