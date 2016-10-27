import matplotlib.pyplot as plt
import numpy as np
import mkl
import requests
#import sys

from collections import OrderedDict

import experiments_proxy.experiment_base as eb

import plot

#sys.path.append('/home/weghebvc/workspace/git/environments_new/src/')
from envs import env_data
from envs import env_data2d
from envs.env_data import EnvData
from envs.env_data2d import EnvData2D
from envs.env_kai import EnvKai



def main():
    
    mkl.set_num_threads(1)

    default_args = {'n_train': 1000,
                    'n_test': 200, 
                    'seed': 0, 
                    'noisy_dims': 0, 
                    'pca': 1.,
                    'repetitions': 20, 
                    'include_random': False, 
                    'include_sfa':    True, 
                    'include_sffa':   False,
                    'include_pfa':    True, 
                    'include_foreca': False, 
                    'include_gfa1':   False, 
                    'include_gfa2':   False,
                    'measure':        eb.Measures.pfa,
                    'use_test_set':   True,  #
                    'causal_features': True,
                    'cachedir': None,#'/scratch/weghebvc',
                    'manage_seed': 'external',
                    'processes': None}

    datasets = [#({'env': EnvKai, 'dataset': None,  'n_train': 2500, 'k': 40, 'p': 1, 'K': 0, 'noisy_dims':  10, 'output_dim': 2, 'iterations': 50, 'k_eval': 40}, {}, (0, 4),   (0,6)),
                {'env': EnvData, 'dataset': env_data.Datasets.EEG,  'k': 2, 'p': 10, 'K': 0, 'pca':  1., 'output_dim': 5, 'iterations': 50, 'k_eval': 10},
                {'env': EnvData, 'dataset': env_data.Datasets.EEG2, 'k': 2, 'p': 10, 'K': 0, 'pca':  1., 'output_dim': 5, 'iterations': 50, 'k_eval': 10},
                {'env': EnvData, 'dataset': env_data.Datasets.FIN_EQU_FUNDS, 'k': 2, 'p': 10, 'K': 0, 'pca':  1., 'output_dim': 5, 'iterations': 50, 'k_eval': 10},
                #({'dataset': eb.Datasets.EEG2_stft_128, 'N': 10000, 'k': 2, 'p': 2, 'K': 0, 'keep_variance': .98, 'output_dim': 2, 'iterations': 50, 'k_eval': 10}, {}, None, None),
                #({'dataset': eb.Datasets.WAV_11k,       'N':  6000, 'k': 2, 'p': 1, 'K': 0, 'keep_variance': .98, 'output_dim': 5, 'iterations': 50}, {'N': [2000, 3000, 4000, 5000, 6000]}),
                #({'dataset': eb.Datasets.WAV2_22k,      'N':  7000, 'k': 2, 'p': 6, 'K': 1, 'keep_variance': .96, 'output_dim': 5, 'iterations': 50}, {'N': [2000, 3000, 4000, 5000, 6000, 7000]}),
                #({'env': EnvData, 'dataset': env_data.Datasets.STFT1, 'n_train': 10000, 'k': 10, 'p': 5, 'K': 10, 'pca': .99, 'output_dim': 5, 'iterations': 50, 'k_eval': 10}, {}, (0,4),   (0,6)),
                #({'dataset': eb.Datasets.WAV3_22k,      'N': 10000, 'k':  2, 'p': 7, 'K':  4, 'keep_variance': .99, 'output_dim': 5, 'iterations': 50, 'k_eval': 10}, {}, (0,5),   (0,9)),
                #({'dataset': eb.Datasets.WAV4_22k,      'N': 10000, 'k': 20, 'p': 6, 'K':  0, 'keep_variance': .99, 'output_dim': 5, 'iterations': 50, 'k_eval': 10}, {}, (-1, 15), None),
                #({'env': EnvData2D, 'dataset': env_data2d.Datasets.Mario,   'n_train': 10000, 'k':  1, 'p': 1, 'K':  1, 'pca': .99, 'output_dim': 5, 'iterations': 50, 'k_eval': 10, 'window': ((70,70),(90,90))}, {}, (0,4), (0,3)),
                #({'env': EnvData2D, 'dataset': env_data2d.Datasets.Traffic, 'n_train': 10000, 'k':  2, 'p': 1, 'K':  1, 'pca': .99, 'output_dim': 5, 'iterations': 50, 'k_eval': 10, 'window': ((35,65),(55,85))}, {}, (0,4), (0,3)),
                #({'dataset': eb.Datasets.Traffic_window, 'repetitions': 5,'N': 10000, 'k': 2,  'p': 1, 'K':  1, 'keep_variance': .99, 'output_dim': 5, 'iterations': 50, 'k_eval': 10}, {}, (0,4),   (0,3)),
                #{'env': EnvData, 'dataset': env_data.Datasets.FIN_EQU_FUNDS, 'n_train': 800, 'n_test': 200, 'k': 2, 'p': 8, 'K':  1, 'pca': 1., 'output_dim': 3, 'iterations': 50, 'k_eval': 5},
                ]
    
    experiments = OrderedDict([('p', range(1,11)), 
                               #('k', [1, 2, 5, 10, 15, 20, 30, 50]),
                               #('k_eval', ([1, 2, 5, 10, 15, 20, 30, 50], .3, (-1, 55))),
                               ('n_train', range(200, 1000, 100)),
                               #('iterations', ([1, 10, 30, 50, 100], .07, (-10, 110))),
                               #('output_dim', range(2,6)),
                               #('K', range(11)),
                               ])
    
    for d, dataset_args in enumerate(datasets):
        
        for i, (experiment_arg, experiment_values) in enumerate(experiments.items()):
        
            kwargs = dict(default_args)
            kwargs.update(dataset_args)
            kwargs[experiment_arg] = experiment_values
            
            #plt.subplot(2, 3, i+1)
            plt.figure(figsize=(5*1.4, 3*1.4))
            legend = True if i==5 else False
            plot.plot_experiment(legend=legend, y_label=True, **kwargs)
            
            plt.xlabel(plt.gca().xaxis.label.get_text() + ' (default: %s)' % dataset_args.get(experiment_arg, None))
            plt.title('abcdef'[i])
            
            #plt.savefig('results_%d%s.eps' % ((d+1), 'abcdef'[i]))

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
    
