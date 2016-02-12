import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

import experiment_base as eb



def main():
    
                # dataset                   N       scale   pca
    datasets = [(eb.Datasets.EEG,           2000,   1.,     1.),
                (eb.Datasets.Face,          1965/2, 1.,     .99),
                (eb.Datasets.Mario_window,  2000,   1.,     .99),
                (eb.Datasets.MEG,           375/2,  1.,     .99),
                (eb.Datasets.RatLab,        2000,   .5,     .96),
                (eb.Datasets.Tumor,         500/2,  .25,    .99),
                ]
    
    k = 5
    K = 1
    iterations = 50
    for a, (algorithm, kwargs) in enumerate([(eb.Algorithms.Random, {}),
                                             (eb.Algorithms.SFA, {}), 
                                             (eb.Algorithms.PFA, {'p': 1, 'K': K}),
                                             (eb.Algorithms.PFA, {'p': 2, 'K': K}),
                                             (eb.Algorithms.GPFA2, {'p': 1, 'k': k, 'iterations': iterations}),
                                             (eb.Algorithms.GPFA2, {'p': 2, 'k': k, 'iterations': iterations}),
                                             ]):
        
        plt.figure(figsize=(22., 12.))
        plt.suptitle('%s %s' % (algorithm, kwargs))
        
        for d, (dataset, N, scaling, keep_variance) in enumerate(datasets):
            plt.subplot(3, 2, d+1)
            plt.title(dataset)
            chunks = []
            for use_test_set in [False, True]: 
                chunks.append(eb.calc_projected_data(algorithm=algorithm, 
                                                     # data 
                                                     data=dataset, 
                                                     output_dim=2,
                                                     N=N,
                                                     scaling=scaling,
                                                     keep_variance=keep_variance, 
                                                     use_test_set=use_test_set,
                                                     # misc 
                                                     seed=0,
                                                     **kwargs)[0])
            for i, color in enumerate(['b', 'r']):
                for j, linestyle in enumerate(['-', ':']):
                    plt.plot(range(i*N, (i+1)*N), chunks[i][:,j], color=color, linestyle=linestyle)
                    
        plt.savefig('experiment_2016_02_plot_signals_%d.pdf' % a)
        
    plt.show()



if __name__ == '__main__':
    main()
        