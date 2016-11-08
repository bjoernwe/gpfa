import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import mkl
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

    default_args_global = {'seed':         0,
                           'noisy_dims':   0,
                           'limit_data':   20000,
                           'algorithm':    eb.Algorithms.ForeCA, 
                           'measure':      eb.Measures.omega,
                           'use_test_set': True,
                           'repetitions':  20,
                           'cachedir':     '/scratch/weghebvc',
                           'manage_seed':  'external',
                           'processes':    None}

    default_args_low = {'seed':         0,
                        'n_train':      1000, 
                        'n_test':       200, 
                        'pca':          1.,
                        'output_dim':   5,
                        'omega_dim':    range(5)}

    datasets_low = [{'env': EnvData, 'dataset': env_data.Datasets.EEG,  'k': 2, 'p': 1},
                    {'env': EnvData, 'dataset': env_data.Datasets.EEG2, 'k': 1, 'p': 1},
                    {'env': EnvData, 'dataset': env_data.Datasets.EIGHT_EMOTION, 'k': 1, 'p': 10},
                    {'env': EnvData, 'dataset': env_data.Datasets.FIN_EQU_FUNDS, 'k': 1, 'p': 10},
                    {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_EHG, 'k': 1, 'p': 2},
                    {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MGH, 'k': 2, 'p': 2},
                    {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MMG, 'k': 2, 'p': 1},
                    {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_UCD, 'k': 2, 'p': 1}
                    ]
                
    colors = iter(matplotlib.cm.rainbow(np.linspace(0, 1, len(datasets_low))))
    markers = iter(['*', 'o', '^', 'v', '<', '>', 'd', 's'] * 2)
    plt.plot([1e0, 1e2], [1e0, 1e2], '-')

    for default_args, datasets in zip([default_args_low], [datasets_low]):
        
        for _, dataset_args in enumerate(datasets):
            
            # PFA/GPFA signals
            kwargs = dict(default_args_global)
            kwargs.update(default_args)
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
            plt.scatter(X, Y, c=color, marker=marker, label=label, s=80)

    # 
    plt.xlabel('predictability of ForeCA')
    plt.ylabel('predictability of SFA')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best', prop={'size': 8})
    plt.show()



if __name__ == '__main__':
    main()
    
