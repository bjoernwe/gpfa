import matplotlib.pyplot as plt
import mkl
#import sys

import experiments_proxy.experiment_base as eb

#sys.path.append('/home/weghebvc/workspace/git/environments_new/src/')
from envs import env_data
from envs import env_data2d
from envs.env_data import EnvData
from envs.env_data2d import EnvData2D
from envs.env_random import EnvRandom

from scatter_plot import scatter_plot



def main():
    
    mkl.set_num_threads(1)

    default_args_global = {'algorithm':    eb.Algorithms.GPFA2, 
                           'measure':      eb.Measures.gpfa,
                           'iterations':   30,
                           'k_eval':       10,
                           'n_train':      10000, 
                           'n_test':       2000, 
                           'seed':         0,
                           'noisy_dims':   0,
                           'limit_data':   25000,
                           'use_test_set': True,
                           'repetitions':  50,
                           'cachedir':     '/scratch/weghebvc',
                           'manage_seed':  'external',
                           'processes':    None}

    default_args_low = {'pca':          1.,
                        'output_dim':   range(1,6)}

    default_args_high = {'pca':          .99,
                         'output_dim':   range(1,11)}
    
    datasets_low = [{'env': EnvData, 'dataset': env_data.Datasets.EEG},
                    {'env': EnvData, 'dataset': env_data.Datasets.EEG2},
                    {'env': EnvData, 'dataset': env_data.Datasets.EIGHT_EMOTION, 'n_train': 1000, 'n_test': 200},
                    {'env': EnvData, 'dataset': env_data.Datasets.FIN_EQU_FUNDS, 'n_train': 1000, 'n_test': 200},
                    {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_EHG},
                    {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MGH},
                    {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_UCD}
                    ]
    
    datasets_high = [{'env': EnvRandom, 'dataset': None, 'ndim': 200, 'k': 2, 'p': 1},
                     {'env': EnvData, 'dataset': env_data.Datasets.HAPT, 'n_train': 5000},
                     {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MMG, 'pca': 1.},
                     {'env': EnvData, 'dataset': env_data.Datasets.STFT1},
                     {'env': EnvData, 'dataset': env_data.Datasets.STFT2},
                     {'env': EnvData, 'dataset': env_data.Datasets.STFT3},
                     {'env': EnvData2D, 'dataset': env_data2d.Datasets.Mario,         'window': ((70,70),(90,90))},
                     {'env': EnvData2D, 'dataset': env_data2d.Datasets.Traffic,       'window': ((35,65),(55,85))},
                     {'env': EnvData2D, 'dataset': env_data2d.Datasets.SpaceInvaders, 'window': ((16,30),(36,50))}
                     ]
    
    parameters_low =  {env_data.Datasets.PHYSIO_MGH: {'p': 2, 'k': 2},
                       env_data.Datasets.PHYSIO_EHG: {'p': 6, 'k': 1},
                       env_data.Datasets.PHYSIO_UCD: {'p': 1, 'k': 5},
                       env_data.Datasets.EIGHT_EMOTION: {'p': 10, 'k': 1},
                       env_data.Datasets.FIN_EQU_FUNDS: {'p': 2, 'k': 1},
                       env_data.Datasets.EEG: {'p': 1, 'k': 2},
                       env_data.Datasets.EEG2: {'p': 1, 'k': 2}}
                
    parameters_high = {env_data.Datasets.HAPT: {'p': 1, 'k': 10},
                       env_data.Datasets.PHYSIO_MMG: {'p': 2, 'k': 2},
                       env_data.Datasets.STFT1: {'p': 4, 'k': 10},
                       env_data.Datasets.STFT2: {'p': 4, 'k': 2},
                       env_data.Datasets.STFT3: {'p': 4, 'k': 1},
                       env_data2d.Datasets.Mario: {'p': 1, 'k': 1},
                       env_data2d.Datasets.Traffic: {'p': 1, 'k': 5},
                       env_data2d.Datasets.SpaceInvaders: {'p': 2, 'k': 5}}
    
    plt.plot([1e-4, 1e3], [1e-4, 1e3], '-')

    scatter_plot(default_args_global=default_args_global,
                 default_args_low=default_args_low,
                 default_args_high=default_args_high,
                 datasets_low=datasets_low, 
                 datasets_high=datasets_high, 
                 parameters_low=parameters_low, 
                 parameters_high=parameters_high)

    # 
    plt.xlabel('error of GPFA')
    plt.ylabel('error of SFA')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best', prop={'size': 8})
    plt.show()



if __name__ == '__main__':
    main()
    
