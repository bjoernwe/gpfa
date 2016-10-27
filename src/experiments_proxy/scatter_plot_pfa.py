import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import mkl
import requests
#import sys

import explot as ep

import experiments_proxy.experiment_base as eb

#sys.path.append('/home/weghebvc/workspace/git/environments_new/src/')
from envs import env_data
from envs import env_data2d
from envs.env_data import EnvData
from envs.env_data2d import EnvData2D
from envs.env_kai import EnvKai
from envs.env_random import EnvRandom



def main():
    
    #mkl.set_num_threads(1)

    default_args = {'seed':         0,
                    'n_train':      10000, 
                    'n_test':       4000, 
                    'pca':          .99,
                    'noisy_dims':   0,
                    'output_dim':   range(1,11), 
                    'algorithm':    eb.Algorithms.PFA, 
                    'measure':      eb.Measures.pfa,
                    'use_test_set': True,
                    'repetitions':  5,
                    'cachedir':     '/scratch/weghebvc',
                    'manage_seed':  'external',
                    'processes':    None}

    datasets = [{'env': EnvRandom, 'dataset': None, 'ndim': 20, 'pca': 1., 'K': 0, 'p': 1},
                #{'env': EnvKai, 'dataset': None, 'noisy_dims': 10, 'pca': 1., 'output_dim': [1,2]},
                {'env': EnvData, 'dataset': env_data.Datasets.EEG,  'pca': 1., 'K': 0, 'p': 10}, # K=0,  p=10
                {'env': EnvData, 'dataset': env_data.Datasets.EEG2, 'pca': 1., 'K': 0, 'p': 10}, # K=0,  p=10
                #{'env': EnvData, 'dataset': env_data.Datasets.HAPT, 'n_train': 5000, 'n_test': 2500},
                #{'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_EHG, 'pca': 1.},
                #{'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MGH, 'pca': 1.},
                #{'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MMG, 'pca': 1.},
                #{'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_UCD, 'pca': 1.},
                {'env': EnvData, 'dataset': env_data.Datasets.STFT1, 'K': 10, 'p': 8},           # K=10, p=8
                {'env': EnvData, 'dataset': env_data.Datasets.STFT2, 'K':  0, 'p': 1},           # K=0,  p=1
                {'env': EnvData, 'dataset': env_data.Datasets.STFT3, 'K':  0, 'p': 1},           # K=0,  p=1
                {'env': EnvData2D, 'dataset': env_data2d.Datasets.Mario,         'K': 0, 'p': 3, 'window': ((70,70),(90,90))},  # K=0, p=3
                {'env': EnvData2D, 'dataset': env_data2d.Datasets.Traffic,       'K': 1, 'p': 3, 'window': ((35,65),(55,85))},  # K=1, p=3
                {'env': EnvData2D, 'dataset': env_data2d.Datasets.SpaceInvaders, 'K': 1, 'p': 8, 'window': ((16,30),(36,50))},  # K=1, p=8
                ]
    
    colors = iter(matplotlib.cm.rainbow(np.linspace(0, 1, len(datasets))))
    markers = iter(['*', 'o', '^', 'v', '<', '>', 'd', 's', '*'])

    for _, dataset_args in enumerate(datasets):

        # PFA/GPFA signals
        kwargs = dict(default_args)
        kwargs.update(dataset_args)
        result = ep.evaluate(eb.prediction_error, argument_order=['output_dim'], ignore_arguments=['window'], **kwargs)
        X = np.mean(result.values, axis=-1) # 1st axis = output_dim, last axis = repetitions
        
        # SFA signals for comparison
        kwargs.update({'algorithm': eb.Algorithms.SFA})
        result_sfa = ep.evaluate(eb.prediction_error, argument_order=['output_dim'], ignore_arguments=['window'], **kwargs)
        Y = np.mean(result_sfa.values, axis=-1) # 1st axis = output_dim, last axis = repetitions

        # plot
        color = next(colors)
        marker = next(markers)
        label = '%s<%s>' % (dataset_args['env'], dataset_args['dataset'])
        plt.scatter(X, Y, c=color, marker=marker, label=label, s=70)

    # 
    plt.plot([1e-4, 8], [1e-4,8], '-')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best', prop={'size': 8})
    plt.show()



if __name__ == '__main__':
    main()
    
