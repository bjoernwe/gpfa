import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

import experiment_base as eb



def main():
    
                # dataset                   N       scale   layers
    datasets = [#(eb.Datasets.Crowd1,        3067/2,   1.,     None),
                (eb.Datasets.Crowd2,        2350/2,   1.,     None),
                #(eb.Datasets.Face,          1965/2,  1.,    None),
                #(eb.Datasets.Mario,          2000,   1.,    None),
                #(eb.Datasets.Mario_window,  2000,   1.,     None),
                #(eb.Datasets.Mouth,         2000,   1.,     None),
                #(eb.Datasets.RatLab,        2000,   1.,     None),
                (eb.Datasets.Traffic,       2000,   1.,     None),
                ]
    
    k = 5
    K = 1
    iterations = 50
    for d, (dataset, N, scaling, layers) in enumerate(datasets):
            
        plt.figure(figsize=(22., 12.))
        plt.suptitle('%s' % (dataset))
        
        for a, (algorithm, kwargs) in enumerate([(eb.Algorithms.HiSFA, {'n_layers': layers}), 
                                                 ]):
            
            plt.subplot(2, 1, a+1)
            plt.title(algorithm)
            chunks = []
            for use_test_set in [False, True]: 
                chunks.append(eb.calc_projected_data(algorithm=algorithm, 
                                                     # data 
                                                     dataset=dataset, 
                                                     output_dim=2,
                                                     N=N,
                                                     scaling=scaling,
                                                     keep_variance=1., 
                                                     use_test_set=use_test_set,
                                                     # misc 
                                                     seed=0,
                                                     **kwargs)[0])
            for i, color in enumerate(['b', 'r']):
                for j, linestyle in enumerate(['-', ':']):
                    plt.plot(range(i*N, (i+1)*N), chunks[i][:,j], color=color, linestyle=linestyle)
                    
        #plt.savefig('experiment_2016_02_plot_signals_%d.pdf' % a)
        
    plt.show()



if __name__ == '__main__':
    main()
        