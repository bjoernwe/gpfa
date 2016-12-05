import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import mkl
#import scipy.stats
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

    default_args_global = {'algorithm':    eb.Algorithms.PFA, 
                           'measure':      eb.Measures.pfa,
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
    
    datasets_high = [{'env': EnvRandom, 'dataset': None, 'ndim': 200, 'K': 0, 'p': 1},
                     {'env': EnvData, 'dataset': env_data.Datasets.HAPT, 'n_train': 5000},
                     {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MMG, 'pca': 1.},
                     {'env': EnvData, 'dataset': env_data.Datasets.STFT1},
                     {'env': EnvData, 'dataset': env_data.Datasets.STFT2},
                     {'env': EnvData, 'dataset': env_data.Datasets.STFT3},
                     {'env': EnvData2D, 'dataset': env_data2d.Datasets.Mario,         'window': ((70,70),(90,90))},
                     {'env': EnvData2D, 'dataset': env_data2d.Datasets.Traffic,       'window': ((35,65),(55,85))},
                     {'env': EnvData2D, 'dataset': env_data2d.Datasets.SpaceInvaders, 'window': ((16,30),(36,50))}
                     ]
    
    parameters_low =  {env_data.Datasets.PHYSIO_MGH: {'p': 10, 'K': 0},
                       env_data.Datasets.PHYSIO_EHG: {'p': 10, 'K': 2},
                       env_data.Datasets.PHYSIO_UCD: {'p': 10, 'K': 0},
                       env_data.Datasets.EIGHT_EMOTION: {'p': 8, 'K': 0},
                       env_data.Datasets.FIN_EQU_FUNDS: {'p': 10, 'K': 10},
                       env_data.Datasets.EEG: {'p': 10, 'K': 0},
                       env_data.Datasets.EEG2: {'p': 6, 'K': 1}}

    parameters_high = {env_data.Datasets.HAPT: {'p': 10, 'K': 0},
                       env_data.Datasets.PHYSIO_MMG: {'p': 2, 'K': 0},
                       env_data.Datasets.STFT1: {'p': 10, 'K': 10},
                       env_data.Datasets.STFT2: {'p': 10, 'K': 10},
                       env_data.Datasets.STFT3: {'p': 10, 'K': 1},
                       env_data2d.Datasets.Mario: {'p': 2, 'K': 0},
                       env_data2d.Datasets.Traffic: {'p': 1, 'K': 0},
                       env_data2d.Datasets.SpaceInvaders: {'p': 2, 'K': 0}}
    
    colors = iter(matplotlib.cm.get_cmap('Set1')(np.linspace(0, 1, len(datasets_low) + len(datasets_high))))
    markers = iter(['*', 'o', '^', 'v', '<', '>', 'd', 's'] * 2)
    
    for default_args, datasets, parameters in zip([default_args_low, default_args_high], 
                                                  [datasets_low, datasets_high], 
                                                  [parameters_low, parameters_high]):
        
        for dataset_args in datasets:
            
            print dataset_args['dataset']
            
            # PFA/GPFA signals
            kwargs = dict(default_args_global)
            kwargs.update(default_args)
            kwargs.update(dataset_args)
            dataset = dataset_args['dataset']
            if dataset in parameters:
                kwargs.update(parameters[dataset])
            result = ep.evaluate(eb.prediction_error, argument_order=['output_dim'], ignore_arguments=['window'], **kwargs)
            kwargs['measure'] = eb.Measures.ndims
            result_dims = ep.evaluate(eb.prediction_error, argument_order=['output_dim'], ignore_arguments=['window'], **kwargs)
            
            # point cloud
            color = next(colors)
            marker = next(markers)
            for i in range(result.values.shape[0]):
                plt.scatter(result_dims.values[i], result.values[i], c=color, marker=marker, label=None, s=80, alpha=.3, linewidths=0, zorder=1)

            # plot
            mu = np.mean(result.values, axis=-1) # last axis = repetitions
            values0 = (result.values.T - mu).T
            values0_dummy = np.array(values0, copy=True)
            values0_dummy[values0 < 0] = np.NaN
            errors_pos = np.sqrt(np.nanmean(values0_dummy**2, axis=-1))
            values0_dummy = np.array(values0, copy=True)
            values0_dummy[values0 > 0] = np.NaN
            errors_neg = np.sqrt(np.nanmean(values0_dummy**2, axis=-1))
 
            mu_dims = np.mean(result_dims.values, axis=-1) # 1st axis = output_dim, last axis = repetitions
            errors_dims = np.std(result_dims.values, axis=-1)
 
            label = '%s<%s>' % (dataset_args['env'], dataset_args['dataset'])
            xerr = errors_dims
            yerr = np.vstack([errors_neg, errors_pos])
            plt.errorbar(mu_dims, mu, xerr=xerr, yerr=yerr, c=color, marker=marker, markersize=7, label=label, zorder=2)
            
    # 
    plt.title(default_args_global['algorithm'])
    plt.xlabel('number of dimensions')
    plt.ylabel('error on test')
    #plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best', prop={'size': 8})
    plt.show()



if __name__ == '__main__':
    main()
    
