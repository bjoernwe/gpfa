import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#import mkl
#import sys

import explot as ep

import experiments_proxy.experiment_base as eb

#sys.path.append('/home/weghebvc/workspace/git/environments_new/src/')
from envs import env_data
from envs import env_data2d
from envs.env_data import EnvData
from envs.env_data2d import EnvData2D
from envs.env_random import EnvRandom



def main():
    
    #mkl.set_num_threads(1)

    default_args_global = {'seed':         0,
                           'noisy_dims':   0,
                           'limit_data':   20000,
                           'algorithm':    eb.Algorithms.PFA, 
                           'measure':      eb.Measures.pfa,
                           'use_test_set': True,
                           'repetitions':  50,
                           'cachedir':     '/scratch/weghebvc',
                           'manage_seed':  'external',
                           'processes':    None}

    default_args_low = {'seed':         0,
                        'n_train':      1000, 
                        'n_test':       200, 
                        'pca':          1.,
                        'output_dim':   range(1,6)}

    default_args_high = {'seed':         0,
                         'n_train':      10000, 
                         'n_test':       2000, 
                         'pca':          .99,
                         'output_dim':   range(1,11)}

    datasets_low = [{'env': EnvData, 'dataset': env_data.Datasets.EEG},
                    {'env': EnvData, 'dataset': env_data.Datasets.EEG2},
                    {'env': EnvData, 'dataset': env_data.Datasets.EIGHT_EMOTION},
                    {'env': EnvData, 'dataset': env_data.Datasets.FIN_EQU_FUNDS},
                    {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_EHG},
                    {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MGH},
                    {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MMG},
                    {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_UCD}
                    ]
    
    parameters_low =  {env_data.Datasets.PHYSIO_MGH: {'p': 6, 'K': 1},
                       env_data.Datasets.PHYSIO_EHG: {'p': 10, 'K': 2},
                       env_data.Datasets.PHYSIO_UCD: {'p': 10, 'K': 1},
                       env_data.Datasets.PHYSIO_MMG: {'p': 6, 'K': 1},
                       env_data.Datasets.EIGHT_EMOTION: {'p': 10, 'K': 6},
                       env_data.Datasets.FIN_EQU_FUNDS: {'p': 10, 'K': 0},
                       env_data.Datasets.EEG: {'p': 10, 'K': 0},
                       env_data.Datasets.EEG2: {'p': 10, 'K': 0}}
                
    datasets_high = [{'env': EnvRandom, 'dataset': None, 'ndim': 200, 'K': 0, 'p': 1},
                     {'env': EnvData, 'dataset': env_data.Datasets.HAPT, 'n_train': 5000},
                     {'env': EnvData, 'dataset': env_data.Datasets.STFT1},
                     {'env': EnvData, 'dataset': env_data.Datasets.STFT2},
                     {'env': EnvData, 'dataset': env_data.Datasets.STFT3},
                     {'env': EnvData2D, 'dataset': env_data2d.Datasets.Mario,         'window': ((70,70),(90,90))},
                     {'env': EnvData2D, 'dataset': env_data2d.Datasets.Traffic,       'window': ((35,65),(55,85))},
                     {'env': EnvData2D, 'dataset': env_data2d.Datasets.SpaceInvaders, 'window': ((16,30),(36,50))}
                     ]
    
    parameters_high = {env_data.Datasets.HAPT: {'p': 10, 'K': 1},
                       env_data.Datasets.STFT1: {'p': 10, 'K': 12},
                       env_data.Datasets.STFT2: {'p': 10, 'K': 12},
                       env_data.Datasets.STFT3: {'p': 8, 'K': 0},
                       env_data2d.Datasets.Mario: {'p': 4, 'K': 0},
                       env_data2d.Datasets.Traffic: {'p': 10, 'K': 0},
                       env_data2d.Datasets.SpaceInvaders: {'p': 10, 'K': 1}}
    
    colors = iter(matplotlib.cm.get_cmap('Set1')(np.linspace(0, 1, len(datasets_low) + len(datasets_high))))
    markers = iter(['*', 'o', '^', 'v', '<', '>', 'd', 's'] * 2)
    plt.plot([1e-7, 1e2], [1e-7, 1e2], '-', zorder=3)

    for default_args, datasets, parameters in zip([default_args_low, default_args_high], 
                                                  [datasets_low, datasets_high], 
                                                  [parameters_low, parameters_high]):
        
        for _, dataset_args in enumerate(datasets):
            
            print dataset_args['dataset']
            
            # PFA/GPFA signals
            kwargs = dict(default_args_global)
            kwargs.update(default_args)
            kwargs.update(dataset_args)
            dataset = dataset_args['dataset']
            if dataset in parameters:
                kwargs.update(parameters[dataset])
            result = ep.evaluate(eb.prediction_error, argument_order=['output_dim'], ignore_arguments=['window'], **kwargs)
            
            # SFA signals for comparison
            kwargs.update({'algorithm': eb.Algorithms.SFA})
            result_sfa = ep.evaluate(eb.prediction_error, argument_order=['output_dim'], ignore_arguments=['window'], **kwargs)
    
            # point cloud
            color = next(colors)
            marker = next(markers)
            for i in range(result.values.shape[0]):
                plt.scatter(result.values[i], result_sfa.values[i], c=color, marker=marker, label=None, s=80, alpha=.3, linewidths=0, zorder=1)

            # plot
            X = np.mean(result.values, axis=-1) # 1st axis = output_dim, last axis = repetitions
            Y = np.mean(result_sfa.values, axis=-1) # 1st axis = output_dim, last axis = repetitions
            label = '%s<%s>' % (dataset_args['env'], dataset_args['dataset'])
            plt.scatter(X, Y, c=color, marker=marker, label=label, s=80, zorder=2)

    # 
    plt.xlabel('error of PFA')
    plt.ylabel('error of SFA')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best', prop={'size': 8})
    plt.show()



if __name__ == '__main__':
    main()
    
