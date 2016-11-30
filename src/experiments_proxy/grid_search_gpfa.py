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



def main():
    
    #mkl.set_num_threads(1)

    default_args_global = {'p':            [1,2,4,6,8],
                           'k':            [1,2,5,10],
                           'algorithm':    eb.Algorithms.GPFA2, 
                           'measure':      eb.Measures.gpfa,
                           'iterations':   30,
                           'k_eval':       10,
                           'n_train':      10000, 
                           'n_test':       2000, 
                           'seed':         0,
                           'noisy_dims':   0,
                           'limit_data':   25000,
                           'use_test_set': True,
                           'cachedir':     '/scratch/weghebvc',
                           'manage_seed':  'external',
                           'processes':    None}

    default_args_low = {'pca':          1.,
                        'output_dim':   range(1,6),
                        'repetitions':  5}

    default_args_high = {'pca':          .99,
                         'output_dim':   range(1,11),
                         'repetitions':  10}

    datasets_low = [{'env': EnvData, 'dataset': env_data.Datasets.EEG},
                    {'env': EnvData, 'dataset': env_data.Datasets.EEG2},
                    {'env': EnvData, 'dataset': env_data.Datasets.EIGHT_EMOTION, 'n_train': 1000, 'n_test': 200},
                    {'env': EnvData, 'dataset': env_data.Datasets.FIN_EQU_FUNDS, 'n_train': 1000, 'n_test': 200},
                    {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_EHG},
                    {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MGH},
                    {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_UCD}
                    ]

    datasets_high = [{'env': EnvData, 'dataset': env_data.Datasets.HAPT, 'n_train': 5000},
                     {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MMG, 'pca': 1.},
                     {'env': EnvData, 'dataset': env_data.Datasets.STFT1},
                     {'env': EnvData, 'dataset': env_data.Datasets.STFT2},
                     {'env': EnvData, 'dataset': env_data.Datasets.STFT3},
                     {'env': EnvData2D, 'dataset': env_data2d.Datasets.Mario,         'window': ((70,70),(90,90))},
                     {'env': EnvData2D, 'dataset': env_data2d.Datasets.Traffic,       'window': ((35,65),(55,85))},
                     {'env': EnvData2D, 'dataset': env_data2d.Datasets.SpaceInvaders, 'window': ((16,30),(36,50))}
                     ]
    
    # dict to store results in
    for default_args, datasets in zip([default_args_low, default_args_high], 
                                      [datasets_low, datasets_high]):
    
        result_dict = {}

        for dataset_args in datasets:
    
            # run cross-validation        
            kwargs = dict(default_args_global)
            kwargs.update(default_args)
            kwargs.update(dataset_args)
            result = ep.evaluate(eb.prediction_error, argument_order=['output_dim'], ignore_arguments=['window'], **kwargs)
            result_averaged = np.mean(result.values, axis=(0, -1)) # average out 1st axis = output_dim and last axis = repetitions
    
            if len(result.iter_args) == 3:
                # grid search
                parameters = result.iter_args.items()[1:] # iter_args w/o output_dim
                idc_min = np.unravel_index(np.argmin(result_averaged), result_averaged.shape) # convert to 2D index
                # assemble result dict
                result_dict[dataset_args['dataset']] = {}
                for i, idx in enumerate(idc_min):
                    result_dict[dataset_args['dataset']][parameters[i][0]] = parameters[i][1][idx]
                print dataset_args['env'], dataset_args['dataset']
            elif len(result.iter_args) == 2:
                # plot results for the one changing parameter
                iter_arg_values = result.iter_args.values()[1]
                plt.figure()
                plt.plot(iter_arg_values, result_averaged)
                plt.xlabel(result.iter_args.keys()[1])
                plt.title(dataset_args['dataset'])
            else:
                assert False
        
        # print results as code ready to copy & paste
        print '{'
        for dataset, values in result_dict.items():
            print '                       env_data.%s: %s,' % (dataset.__str__(), values)
        print '}'
        
    plt.show()



if __name__ == '__main__':
    main()
    
