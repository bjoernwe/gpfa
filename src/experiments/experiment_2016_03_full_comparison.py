import collections
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

import experiment_base as eb



def main():

    output_dim = 10
    expansion = True
    
                # dataset                   N       scale   hi-net
    datasets = [#(eb.Datasets.Crowd1,        [2000,1067],   .25,  {'expansion': expansion, 'channels_xy_1': (5,3), 'spacing_xy_1': (5,3), 'channels_xy_n': (2,1), 'spacing_xy_n': (2,1), 'node_output_dim': 10}),
                #(eb.Datasets.Crowd2,        [2000,350],   .25,  {'expansion': expansion, 'channels_xy_1': (5,3), 'spacing_xy_1': (5,3), 'channels_xy_n': (2,1), 'spacing_xy_n': (2,1), 'node_output_dim': 10}),
                #(eb.Datasets.Mario_window,  2000,   1.,     None),
                #(eb.Datasets.RatLab,        2000,   1.,     {}),
                #(eb.Datasets.Traffic,       2000,   .25,     {}),#{'channels_xy_1': (8,8), 'spacing_xy_1': (4,4), 'channels_xy_n': (2,2), 'spacing_xy_n': (2,2)}),
                
                (eb.Datasets.Face,          [1500,465],   1.,    {'expansion': expansion, 'channels_xy_1': (3,5), 'spacing_xy_1': (3,5), 'channels_xy_n': (2,1), 'spacing_xy_n': (2,1), 'node_output_dim': 10}),
                #(eb.Datasets.Mario,          2000,   1.,    {'expansion': expansion, 'channels_xy_1': (5,4), 'spacing_xy_1': (5,4), 'channels_xy_n': (2,1), 'spacing_xy_n': (2,1), 'node_output_dim': 10}),
                #(eb.Datasets.Mouth,         2000,   1.,     {'expansion': expansion, 'channels_xy_1': (5,3), 'spacing_xy_1': (5,3), 'channels_xy_n': (2,1), 'spacing_xy_n': (2,1), 'node_output_dim': 10}),
                ]
    
    k = 2
    K = 1
    p = 1
    iterations = 50

    algorithms = [(eb.Algorithms.HiSFA, eb.Measures.delta_ndim, {}), 
                  (eb.Algorithms.HiPFA, None, {'expansion': expansion, 'p': p, 'K': K}),
                  (eb.Algorithms.HiGPFA2, eb.Measures.gpfa_ndim, {'expansion': expansion, 'p': p, 'k': k, 'iterations': iterations}),
                 ]
    
    measures = [(eb.Measures.delta_ndim, {}), 
                (eb.Measures.gpfa_ndim, {'k': k, 'p': p})]
    
    for _, (dataset, N, scaling, kwargs_dat) in enumerate(datasets):
            
        plt.figure(figsize=(22., 11.))
        plt.suptitle('%s : %s' % (dataset, kwargs_dat))
        
        chunks = {}
        for a, (algorithm, _, kwargs_alg) in enumerate(algorithms):
            
            plt.subplot(4, 1, a+1)
            plt.title('%s : %s' % (algorithm, kwargs_alg))
            
            kwargs = dict(kwargs_dat)
            kwargs.update(kwargs_alg)
            
            chunks[algorithm] = []
            for use_test_set in [False, True]: 
                chunks[algorithm].append(eb.calc_projected_data(algorithm=algorithm, 
                                                     # data 
                                                     dataset=dataset, 
                                                     output_dim=output_dim,
                                                     N=N,
                                                     scaling=scaling,
                                                     keep_variance=1., 
                                                     use_test_set=use_test_set,
                                                     # misc 
                                                     seed=0,
                                                     **kwargs)[0])
            
            colors = ['b', 'r']
            for d in range(output_dim-1, -1, -1):
                if isinstance(N, collections.Iterable):
                    plt.plot(range(N[0]), chunks[algorithm][0][:,d], color=_color(c=colors[0], d=d))
                    plt.plot(range(N[0], N[0]+N[1]), chunks[algorithm][1][:,d], color=_color(c=colors[1], d=d))
                else:
                    for i, color in enumerate(colors):
                        plt.plot(range(i*N, (i+1)*N), chunks[algorithm][i][:,d], color=_color(c=color, d=d))
                        
    
        for m, (measure, kwargs_mea) in enumerate(measures):
            
            plt.subplot(4,2,7+m)
            plt.ylabel(measure)
                
            for a, (algorithm, _, kwargs_alg) in enumerate(algorithms):
                
                kwargs = dict(kwargs_dat)
                kwargs.update(kwargs_alg)
                kwargs.update(kwargs_mea)
                errors = eb.prediction_error(measure=measure, 
                                             dataset=dataset, 
                                             algorithm=algorithm, 
                                             output_dim=output_dim, 
                                             N=N, 
                                             use_test_set=False,
                                             **kwargs)
                colors = ['b', 'r', 'g']
                w = .8 / len(algorithms)
                plt.bar(np.arange(output_dim)+1+a*w, height=errors, width=w, color=colors[a])
    
    
                    
    plt.show()



def _color(c, d):
    tb = np.array([0,0,1], dtype=np.float)
    tr = np.array([1,0,0], dtype=np.float)
    tw = np.array([1,1,1], dtype=np.float)
    f = 1. / (d+1)**1.5
    if c == 'r':
        c = tr
    elif c == 'b':
        c = tb
    else:
        assert False
    return f*c + (1-f)*tw



if __name__ == '__main__':
    main()
        