import matplotlib.pyplot as plt
#import mkl
#import sys

import experiments_proxy.experiment_base as eb

#sys.path.append('/home/weghebvc/workspace/git/environments_new/src/')
from envs import env_data
from envs.env_data import EnvData
from envs.env_random import EnvRandom

from scatter_plot import scatter_plot



def main():
    
    #mkl.set_num_threads(1)

    default_args_global = {'seed':         0,
                           'noisy_dims':   0,
                           'limit_data':   20000,
                           'algorithm':    eb.Algorithms.ForeCA, 
                           'measure':      eb.Measures.omega,
                           'use_test_set': True,
                           'repetitions':  50,
                           'cachedir':     '/scratch/weghebvc',
                           'manage_seed':  'external',
                           'processes':    None}

    default_args_low = {'seed':         0,
                        'n_train':      1000, 
                        'n_test':       200, 
                        'pca':          1.,
                        'output_dim':   5,
                        'omega_dim':    range(5)}

    datasets_low = [{'env': EnvRandom, 'dataset': None, 'ndim': 20},
                    {'env': EnvData, 'dataset': env_data.Datasets.EEG},
                    {'env': EnvData, 'dataset': env_data.Datasets.EEG2},
                    {'env': EnvData, 'dataset': env_data.Datasets.EIGHT_EMOTION},
                    {'env': EnvData, 'dataset': env_data.Datasets.FIN_EQU_FUNDS},
                    {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_EHG},
                    {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MGH},
                    {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MMG},
                    {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_UCD}
                    ]
                
    plt.plot([1e-1, 1e2], [1e-1, 1e2], '-')

    scatter_plot(default_args_global=default_args_global,
                 default_args_low=default_args_low,
                 default_args_high={},
                 datasets_low=datasets_low, 
                 datasets_high=[], 
                 parameters_low={}, 
                 parameters_high={})

    #
    plt.xlabel('predictability of ForeCA')
    plt.ylabel('predictability of SFA')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best', prop={'size': 8})
    plt.show()



if __name__ == '__main__':
    main()
    
