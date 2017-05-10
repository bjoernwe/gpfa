import mdp
import numpy as np

import explot as ep

import experiments_proxy.experiment_base as eb

from envs import env_data
from envs import env_data2d
from envs.env_data import EnvData
from envs.env_data2d import EnvData2D
from envs.env_random import EnvRandom



default_args_global = {'n_train':      10000, 
                       'n_test':       2000,
                       'seed':         1,
                       'noisy_dims':   0,
                       'limit_data':   100000,
                       'use_test_set': True}

default_args_explot = {'repetitions':  20,
                       'cachedir':     '/scratch/weghebvc',
                       'manage_seed':  'repetition_index',
                       'verbose':      True,
                       'processes':    None}

default_args_low  = {#'pca':         1.,
                     'output_dim':  range(1,6),
                     #'output_dim_max': 5,
                     }

algorithm_measures = {eb.Algorithms.HiSFA:  eb.Measures.delta,
                      eb.Algorithms.HiSFFA: eb.Measures.delta,
                      eb.Algorithms.HiPFA:  eb.Measures.pfa,
                      eb.Algorithms.HiGPFA: eb.Measures.gpfa,
                      }

algorithm_args = {eb.Algorithms.HiSFA:  {},
                  eb.Algorithms.HiSFFA: {},
                  eb.Algorithms.HiPFA:  {},
                  eb.Algorithms.HiGPFA: {'iterations': 30,
                                         'k_eval': 10}
                  }

dataset_args_hi = [{'env': EnvData2D, 'dataset': env_data2d.Datasets.SpaceInvaders, 'scaling': (50,50), 'window': ((0,14),( 52, 66)), 'pca': 1., 'whitening': False},
                   {'env': EnvData2D, 'dataset': env_data2d.Datasets.Mario,         'scaling': (50,50), 'window': ((0,20),(120,140)), 'pca': 1., 'whitening': False},
                   {'env': EnvData2D, 'dataset': env_data2d.Datasets.Traffic,       'scaling': (50,50), 'window': ((0,30),( 90,120)), 'pca': 1., 'whitening': False},
                   ]

# extracting 10 dimensions when dim >= 20, extracting 5 otherwise
dataset_default_args = {env_data2d.Datasets.Mario: default_args_low,
                        env_data2d.Datasets.Traffic: default_args_low,
                        env_data2d.Datasets.SpaceInvaders: default_args_low}

# results from grid-search
algorithm_parameters = {eb.Algorithms.HiPFA: {env_data2d.Datasets.SpaceInvaders: {'p': 2, 'K': 0},
                                              env_data2d.Datasets.Mario: {'p': 2, 'K': 0},
                                              env_data2d.Datasets.Traffic: {'p': 1, 'K': 0}},
                        eb.Algorithms.HiGPFA:{env_data2d.Datasets.SpaceInvaders: {'p': 1, 'k': 5},
                                              env_data2d.Datasets.Mario: {'p': 1, 'k': 2},
                                              env_data2d.Datasets.Traffic: {'p': 1, 'k': 10}}}



def get_results(alg, overide_args={}, include_random=True):

    results = {}
    
    for args in dataset_args_hi:
        env = args['env']
        dataset = args['dataset']
        print dataset
        if not include_random and env is EnvRandom:
            continue
        kwargs = dict(default_args_global)
        kwargs.update(default_args_explot)
        kwargs['algorithm'] = alg
        kwargs['measure'] = algorithm_measures[alg]
        kwargs.update(args)
        kwargs.update(dataset_default_args.get(dataset, {}))
        kwargs.update(algorithm_parameters.get(alg, {}).get(dataset, {}))
        kwargs.update(algorithm_args.get(alg, {}))
        kwargs.update(overide_args)
    
        #print 'results: %s' % kwargs
        results[dataset] = ep.evaluate(eb.prediction_error, argument_order=['output_dim'], ignore_arguments=['window', 'scaling'], **kwargs)
        
    return results



def get_signals(alg, overide_args={}, include_random=True, repetition_index=0):

    results = {}

    for args in dataset_args_hi:
        env = args['env']
        dataset = args['dataset']
        print dataset
        if not include_random and env is EnvRandom:
            continue
        kwargs = dict(default_args_global)
        kwargs['algorithm'] = alg
        kwargs['measure'] = algorithm_measures[alg]
        kwargs.update(args)
        kwargs.update(dataset_default_args.get(dataset, {}))
        kwargs.update(algorithm_parameters.get(alg, {}).get(dataset, {}))
        kwargs.update(algorithm_args.get(alg, {}))
        kwargs.update({'output_dim': 5, 'output_dim_max': 5})
        kwargs.update(overide_args)
    
        try:
            # list of repetition indices?
            projected_data_list = []
            for i in repetition_index:
                projected_data, _, [_, _] = eb.calc_projected_data(repetition_index=i, **kwargs)
                projected_data_list.append(projected_data)
            projected_data = np.stack(projected_data_list, axis=2)
            data_train     = None
            data_test      = None
        except TypeError:
            projected_data, _, [data_train, data_test] = eb.calc_projected_data(repetition_index=repetition_index, **kwargs)
        result = {'projected_data': projected_data, 'data_train': data_train, 'data_test': data_test}
        results[dataset] = result

    return results



if __name__ == '__main__':
    print get_results(eb.Algorithms.SFA)

