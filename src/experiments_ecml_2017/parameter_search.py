import matplotlib.pyplot as plt
import numpy as np
import mkl
import requests
import scipy.stats

from collections import OrderedDict

import experiments.experiment_base as eb

import plot



def main():
    
    mkl.set_num_threads(1)

    default_args = {'N2': 5000, 
                    'seed': 0, 
                    #'keep_variance': 1.,
                    'noisy_dims': 0, 
                    'repetitions': 50, 
                    'include_random': True, 
                    'include_sfa':    True, 
                    'include_sffa':   False, 
                    'include_foreca': False, # 
                    'include_gfa1':   True, 
                    'include_gfa2':   True,
                    'use_test_set':   True,  #
                    'causal_features': True,
                    'cachedir': '/scratch/weghebvc',
                    'manage_seed': 'external',
                    'processes': None}

    datasets = [#({'dataset': eb.Datasets.Kai,           'N': 2500, 'k': 40, 'p': 1, 'K': 0, 'noisy_dims':  10, 'output_dim': 2, 'iterations': 50, 'k_eval': 40}, {}, (0, 4),   (0,6)),
                #({'dataset': eb.Datasets.EEG,           'N': 10000, 'k': 2, 'p': 2, 'K': 0, 'keep_variance':  1., 'output_dim': 2, 'iterations': 50, 'k_eval': 10}, {}, None, None),
                #({'dataset': eb.Datasets.EEG2,          'N': 10000, 'k': 2, 'p': 2, 'K': 0, 'keep_variance':  1., 'output_dim': 2, 'iterations': 50, 'k_eval': 10}, {}, None, None),
                #({'dataset': eb.Datasets.EEG2_stft_128, 'N': 10000, 'k': 2, 'p': 2, 'K': 0, 'keep_variance': .98, 'output_dim': 2, 'iterations': 50, 'k_eval': 10}, {}, None, None),
                #({'dataset': eb.Datasets.WAV_11k,       'N':  6000, 'k': 2, 'p': 1, 'K': 0, 'keep_variance': .98, 'output_dim': 5, 'iterations': 50}, {'N': [2000, 3000, 4000, 5000, 6000]}),
                #({'dataset': eb.Datasets.WAV2_22k,      'N':  7000, 'k': 2, 'p': 6, 'K': 1, 'keep_variance': .96, 'output_dim': 5, 'iterations': 50}, {'N': [2000, 3000, 4000, 5000, 6000, 7000]}),
                ({'dataset': eb.Datasets.WAV_22k,       'N': 10000, 'k': 10, 'p': 5, 'K': 10, 'keep_variance': .99, 'output_dim': 5, 'iterations': 50, 'k_eval': 10}, {}, (0,4),   (0,6)),
                ({'dataset': eb.Datasets.WAV3_22k,      'N': 10000, 'k':  2, 'p': 7, 'K':  4, 'keep_variance': .99, 'output_dim': 5, 'iterations': 50, 'k_eval': 10}, {}, (0,5),   (0,9)),
                ({'dataset': eb.Datasets.WAV4_22k,      'N': 10000, 'k': 20, 'p': 6, 'K':  0, 'keep_variance': .99, 'output_dim': 5, 'iterations': 50, 'k_eval': 10}, {}, (-1, 15), None),
                ({'dataset': eb.Datasets.Mario_window,  'N': 10000, 'k':  1, 'p': 1, 'K':  1, 'keep_variance': .99, 'output_dim': 5, 'iterations': 50, 'k_eval': 10}, {}, (0,4),   (0,3)),
                ({'dataset': eb.Datasets.Traffic_window, 'repetitions': 5,'N': 10000, 'k': 2,  'p': 1, 'K':  1, 'keep_variance': .99, 'output_dim': 5, 'iterations': 50, 'k_eval': 10}, {}, (0,4),   (0,3)),
                ]
    
    experiments = OrderedDict([('p', (range(1,9), .07, (0, 9))), 
                               ('k', ([1, 2, 5, 10, 15, 20, 30, 50], .3, (-1, 55))),
                               ('k_eval', ([1, 2, 5, 10, 15, 20, 30, 50], .3, (-1, 55))),
                               ('N', ([2000, 4000, 6000, 8000, 10000], 50, (1000, 11000))),
                               ('iterations', ([1, 10, 30, 50, 100], .07, (-10, 110))),
                               ('output_dim', (range(2,13), .07, (0, 14))),
                               #('K', (range(11), 0, (0, 11))),
                               ])
    
    wilcoxon_results = 'Wilcoxon with %d repetitions' % default_args['repetitions']            
    
    for d, (dataset_args, overide_args, ylim, ylim2) in enumerate(datasets):
        
        #plt.figure(figsize=(12., 10.))
        #plt.suptitle('%s\n%s' % (dataset_args['dataset'], default_args))        
        
        for i, (experiment_arg, (experiment_values, x_offset, xlim)) in enumerate(experiments.items()):
        
            kwargs = dict(default_args)
            kwargs.update(dataset_args)
            kwargs[experiment_arg] = experiment_values
            
            if experiment_arg in overide_args:
                kwargs[experiment_arg] = overide_args[experiment_arg]

            #plt.subplot(2, 3, i+1)
            plt.figure(figsize=(5*1.4, 3*1.4))
            legend = True if i==5 else False
            results = plot.plot_experiment(legend=legend, x_offset=x_offset, y_label=True, **kwargs)
            
            plt.xlabel(plt.gca().xaxis.label.get_text() + ' (default: %s)' % dataset_args.get(experiment_arg, None))
            plt.title('abcdef'[i])
            plt.xlim(xlim)
            if i<5:
                plt.ylim(ylim)
            if i==5 and ylim2 is not None:
                plt.ylim(ylim2)
            
            plt.savefig('results_%d%s.eps' % ((d+1), 'abcdef'[i]))

            # table of Wilcoxon comparisons
            if experiment_arg == 'output_dim':
                wilcoxon_results += str(dataset_args['dataset']) + '\n'
                for alg_ref in [eb.Algorithms.GPFA1, eb.Algorithms.GPFA2]:
                    algs_comp = [eb.Algorithms.SFA, eb.Algorithms.PFA]
                    if dataset_args['dataset'] == eb.Datasets.Mario_window:
                        index = results[alg_ref].iter_args[experiment_arg].index(12)
                    else:
                        index = results[alg_ref].iter_args[experiment_arg].index(dataset_args[experiment_arg])
                    p_values = map(lambda x: scipy.stats.wilcoxon(results[alg_ref].values[index], results[x].values[index])[1], algs_comp)
                    wilcoxon_results += '%s vs SFA / PFA : %0.4f & %0.4f\n' % (alg_ref, p_values[0], p_values[1])
        with open("results_wilcoxon.txt", "w") as text_file:
            text_file.write(wilcoxon_results)
    print wilcoxon_results

    plt.show()



def notify(msg):
    requests.post('http://api.pushover.net/1/messages.json', 
                  data=dict(token='aAKkJ12jjmZgbqjXj4hxKGrYJC9jh3',
                            user='uoKXUWzBShY3k4sfkRowhGCi1gv8w5',
                            message=msg))



if __name__ == '__main__':
    try:
        main()
    finally:
        #notify('finished')
        pass
    plt.show()
    
