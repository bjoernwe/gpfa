import numpy as np

from matplotlib import pyplot
#from matplotlib.offsetbox import OffsetImage, AnnotationBbox

#import mdp

import experiment_base as eb


if __name__ == '__main__':
    
    k = 5
    iterations = 5

    baseline_result = {}
    for algorithm in ['lpp', 'fpp']:
        baseline_result[algorithm] = eb.experiment(algorithm=algorithm, k=k, iterations=iterations, reduce_variance=False, whitening=False, normalize_std=False, additional_noise_dim=0, additional_noise_std=0)

    results = {}
    for a, algorithm in enumerate(['lpp', 'fpp']):
        results[algorithm] = {}
        for p, params in enumerate([(False, False, False), (False, False, True), (True, False, False), (False, True, False)]):
            results[algorithm][p] = {}
            tmp_result = eb.experiment(algorithm=algorithm, k=k, iterations=iterations, reduce_variance=params[0], whitening=params[1], normalize_std=params[2], additional_noise_dim=0, additional_noise_std=0)
            results[algorithm][p]['neighborhood'] = eb.performance_lpp(projected_data=tmp_result, k=k, baseline_result=baseline_result[algorithm])
            results[algorithm][p]['future']       = eb.performance_fpp(projected_data=tmp_result, k=k, baseline_result=baseline_result[algorithm])

    for m, measure in enumerate(['neighborhood', 'future']):
        pyplot.subplot(1, 2, m+1)
        pyplot.title(measure)
        pyplot.bar(np.arange(4), [results['lpp'][i][measure] for i in range(4)], width=.35, color='b')
        pyplot.bar(np.arange(4)+.35, [results['fpp'][i][measure] for i in range(4)], width=.35, color='r')

    pyplot.show()
    