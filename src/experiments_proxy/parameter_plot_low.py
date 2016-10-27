import matplotlib.pyplot as plt
import numpy as np
import mkl
#import sys

from collections import OrderedDict

import experiments_proxy.experiment_base as eb

import plot

#sys.path.append('/home/weghebvc/workspace/git/environments_new/src/')
from envs import env_data
from envs.env_data import EnvData



def main():
    
    mkl.set_num_threads(1)

    default_args = {'n_train': 1000,
                    'n_test': 200, 
                    'seed': 0, 
                    'noisy_dims': 0, 
                    'pca': 1.,
                    'output_dim': 5,
                    'iterations': 50,
                    'k_eval': 10,
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

    datasets = [{'env': EnvData, 'dataset': env_data.Datasets.EEG,  'k': 1, 'p': 1, 'K': 0},
                {'env': EnvData, 'dataset': env_data.Datasets.EEG2, 'k': 1, 'p': 1, 'K': 0},
                {'env': EnvData, 'dataset': env_data.Datasets.FIN_EQU_FUNDS, 'k': 1, 'p': 10, 'K': 0},
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_EHG, 'k': 1, 'p': 10, 'K': 0},
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MGH, 'k': 1, 'p': 2, 'K': 0},
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MMG, 'k': 2, 'p': 2, 'K': 0},
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_UCD, 'k': 2, 'p': 1, 'K': 0},
                ]
    
    experiments = OrderedDict([('p', range(1,11)), 
                               #('k', [1, 2, 5, 10, 15, 20, 30, 50]),
                               #('k_eval', ([1, 2, 5, 10, 15, 20, 30, 50], .3, (-1, 55))),
                               ('n_train', range(200, 1100, 100)),
                               #('iterations', ([1, 10, 30, 50, 100], .07, (-10, 110))),
                               #('output_dim', range(2,6)),
                               #('K', range(11)),
                               ])
    
    for _, dataset_args in enumerate(datasets):
        
        for _, (experiment_arg, experiment_values) in enumerate(experiments.items()):
        
            kwargs = dict(default_args)
            kwargs.update(dataset_args)
            kwargs[experiment_arg] = experiment_values
            
            plt.figure()
            plot.plot_experiment(**kwargs)
            
            plt.xlabel(plt.gca().xaxis.label.get_text() + ' (default: %s)' % dataset_args.get(experiment_arg, None))

    plt.show()



if __name__ == '__main__':
    main()
    
