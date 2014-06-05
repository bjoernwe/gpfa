import numpy as np
import pickle

from matplotlib import pyplot

import experiment_base as eb


if __name__ == '__main__':
    
    k = 5
    iterations = 5
    iteration_dim = 50
    trials = 2
    dimensions = [0, 1, 2, 5, 10, 20]#, 50, 100, 200, 500]

    baseline_result = {}
    for algorithm in ['random', 'lpp', 'fpp']:
        baseline_result[algorithm] = {}
        for do_pca in [False, True]:
            baseline_result[algorithm][do_pca] = eb.experiment(algorithm=algorithm, k=k, iterations=iterations, iteration_dim=iteration_dim, reduce_variance=do_pca, additional_noise_dim=0, additional_noise_std=0, additive_noise=0)

    results = {}

    for p, do_pca in enumerate([False, True]):
    
        for a, algorithm in enumerate(['lpp', 'fpp']):
    
            print algorithm
            
            results[algorithm] = {}
            results[algorithm]['lpp'] = np.zeros((len(dimensions),trials))
            results[algorithm]['fpp'] = np.zeros((len(dimensions),trials))
            
            for i, dim in enumerate(dimensions):
                print dim
                for r in range(trials):
                    tmp_result = eb.experiment(algorithm=algorithm, k=k, iterations=iterations, iteration_dim=iteration_dim, reduce_variance=do_pca, additional_noise_dim=dim, additional_noise_std=200, additive_noise=0)
                    results[algorithm]['lpp'][i,r] = eb.performance_lpp(projected_data=tmp_result, k=k, baseline_result=baseline_result[algorithm][do_pca])
                    results[algorithm]['fpp'][i,r] = eb.performance_fpp(projected_data=tmp_result, k=k, baseline_result=baseline_result[algorithm][do_pca])
    
            pyplot.subplot(2, 2, 2*p+a+1)
            pyplot.title(np.array(['', 'pca + '])[do_pca] + algorithm)
            pyplot.errorbar(x=dimensions, y=np.mean(results[algorithm]['lpp'], axis=1), yerr=np.std(results[algorithm]['lpp'], axis=1))
            pyplot.errorbar(x=dimensions, y=np.mean(results[algorithm]['fpp'], axis=1), yerr=np.std(results[algorithm]['fpp'], axis=1))
            pyplot.legend(['neighborhood', 'future'])
        
    pickle.dump(baseline_result, open('baseline_result.pkl', 'wb'))
    pickle.dump(results, open('results.pkl', 'wb'))
    pyplot.show()
    