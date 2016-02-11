import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

import experiment_base as eb



def main():
    
    datasets = [(eb.Datasets.EEG, 2000, 1., 1.),
                (eb.Datasets.Face, 1965/2, 1., .99),
                (eb.Datasets.Mario_window, 2000, 1., .99),
                (eb.Datasets.MEG, 375/2, 1., .99),
                (eb.Datasets.RatLab, 2000, .5, .96),
                (eb.Datasets.Tumor, 500/2, .25, .99),
                ]
    
    for algorithm, kwargs in [(eb.Algorithms.SFA, {}), 
                              (eb.Algorithms.GPFA2, {'p': 1, 'k': 5}),
                              (eb.Algorithms.GPFA2, {'p': 2, 'k': 5}),
                              (eb.Algorithms.GPFA2, {'p': 3, 'k': 5})]:
        
        plt.figure()
        plt.suptitle('%s %s' % (algorithm, kwargs))
        
        for i, (dataset, N, scaling, keep_variance) in enumerate(datasets):
            plt.subplot(3, 2, i+1)
            plt.title(dataset)
            chunks = []
            for use_test_set in [False, True]: 
                chunks.append(eb.calc_projected_data(algorithm=algorithm, 
                                                     # data 
                                                     data=dataset, 
                                                     output_dim=1,
                                                     N=N,
                                                     scaling=scaling,
                                                     keep_variance=keep_variance, 
                                                     use_test_set=use_test_set,
                                                     # algorithm
                                                     p=kwargs.get('p', 1),
                                                     k=kwargs.get('k', 5),
                                                     iterations=50,
                                                     # misc 
                                                     seed=0))
            for i, color in enumerate(['b', 'r']):
                plt.plot(range(i*N, (i+1)*N), chunks[i], color=color)
        
    plt.show()



if __name__ == '__main__':
    main()
        