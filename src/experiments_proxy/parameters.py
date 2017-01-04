import explot as ep

import experiments_proxy.experiment_base as eb

from envs import env_data
from envs import env_data2d
from envs.env_data import EnvData
from envs.env_data2d import EnvData2D
from envs.env_random import EnvRandom

default_args_global = {'n_train':      10000, 
                       'n_test':       2000,
                       'seed':         0,
                       'noisy_dims':   0,
                       'limit_data':   25000,
                       'use_test_set': True,
                       'repetitions':  50,
                       'cachedir':     '/scratch/weghebvc',
                       'manage_seed':  'external',
                       'processes':    None}

default_args_low  = {'pca':         1.,
                     'output_dim':  range(1,6),
                     #'repetitions': 10
                     }

default_args_high = {'pca':         .99,
                     'output_dim':  range(1,11),
                     #'repetitions': 5
                     }

algorithm_measures = {eb.Algorithms.SFA:    eb.Measures.delta,
                      eb.Algorithms.SFFA:   eb.Measures.delta,
                      eb.Algorithms.ForeCA: eb.Measures.omega,
                      eb.Algorithms.PFA:    eb.Measures.pfa,
                      eb.Algorithms.GPFA2:  eb.Measures.gpfa}

algorithm_args = {eb.Algorithms.ForeCA: {'n_train':      1000, 
                                         'n_test':       200,
                                         #'pca':          1.,
                                         'output_dim':   5,
                                         'omega_dim':    range(5)},
                  eb.Algorithms.GPFA2:  {'iterations':   30,
                                         'k_eval':       10,}}

dataset_args = [{'env': EnvData, 'dataset': env_data.Datasets.EEG},
                {'env': EnvData, 'dataset': env_data.Datasets.EEG2},
                {'env': EnvData, 'dataset': env_data.Datasets.EIGHT_EMOTION, 'n_train': 1000, 'n_test': 200},
                {'env': EnvData, 'dataset': env_data.Datasets.FIN_EQU_FUNDS, 'n_train': 1000, 'n_test': 200},
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_EHG},
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MGH},
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_UCD},
                {'env': EnvRandom, 'dataset': None, 'ndim': 200, 'K': 0, 'p': 1, 'k': 1, 'output_dim': range(1,11)},
                {'env': EnvData, 'dataset': env_data.Datasets.HAPT, 'n_train': 5000},
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MMG, 'pca': 1.},
                {'env': EnvData, 'dataset': env_data.Datasets.STFT1},
                {'env': EnvData, 'dataset': env_data.Datasets.STFT2},
                {'env': EnvData, 'dataset': env_data.Datasets.STFT3},
                {'env': EnvData2D, 'dataset': env_data2d.Datasets.Mario,         'window': ((70,70),(90,90))},
                {'env': EnvData2D, 'dataset': env_data2d.Datasets.SpaceInvaders, 'window': ((16,30),(36,50))},
                {'env': EnvData2D, 'dataset': env_data2d.Datasets.Traffic,       'window': ((35,65),(55,85))}
                ]

datasets_low_dimensional = set([env_data.Datasets.EEG,
                                env_data.Datasets.EEG2,
                                env_data.Datasets.EIGHT_EMOTION,
                                env_data.Datasets.FIN_EQU_FUNDS,
                                env_data.Datasets.PHYSIO_EHG,
                                env_data.Datasets.PHYSIO_MGH,
                                env_data.Datasets.PHYSIO_UCD])

dataset_default_args = {env_data.Datasets.PHYSIO_MGH: default_args_low,
                        env_data.Datasets.PHYSIO_EHG: default_args_low,
                        env_data.Datasets.PHYSIO_UCD: default_args_low,
                        env_data.Datasets.EIGHT_EMOTION: default_args_low,
                        env_data.Datasets.FIN_EQU_FUNDS: default_args_low,
                        env_data.Datasets.EEG: default_args_low,
                        env_data.Datasets.EEG2: default_args_low,
                        #
                        env_data.Datasets.HAPT: default_args_high,
                        env_data.Datasets.PHYSIO_MMG: default_args_high,
                        env_data.Datasets.STFT1: default_args_high,
                        env_data.Datasets.STFT2: default_args_high,
                        env_data.Datasets.STFT3: default_args_high,
                        env_data2d.Datasets.Mario: default_args_high,
                        env_data2d.Datasets.Traffic: default_args_high,
                        env_data2d.Datasets.SpaceInvaders: default_args_high}

# results from grid-search
algorithm_parameters = {eb.Algorithms.PFA: {env_data.Datasets.EEG: {'p': 10, 'K': 0},
                                            env_data.Datasets.EEG2: {'p': 8, 'K': 0},
                                            env_data.Datasets.EIGHT_EMOTION: {'p': 8, 'K': 0},
                                            env_data.Datasets.FIN_EQU_FUNDS: {'p': 10, 'K': 8},
                                            env_data.Datasets.PHYSIO_EHG: {'p': 10, 'K': 2},
                                            env_data.Datasets.PHYSIO_MGH: {'p': 10, 'K': 0},
                                            env_data.Datasets.PHYSIO_UCD: {'p': 10, 'K': 0},
                                            #
                                            env_data.Datasets.HAPT: {'p': 10, 'K': 0},
                                            env_data.Datasets.PHYSIO_MMG: {'p': 2, 'K': 10},
                                            env_data.Datasets.STFT1: {'p': 10, 'K': 10},
                                            env_data.Datasets.STFT2: {'p': 10, 'K': 10},
                                            env_data.Datasets.STFT3: {'p': 10, 'K': 1},
                                            env_data2d.Datasets.Mario: {'p': 2, 'K': 0},
                                            env_data2d.Datasets.Traffic: {'p': 1, 'K': 0},
                                            env_data2d.Datasets.SpaceInvaders: {'p': 10, 'K': 0}},
                        eb.Algorithms.GPFA2: {env_data.Datasets.EEG: {'p': 1, 'k': 1},
                                              env_data.Datasets.EEG2: {'p': 1, 'k': 1},
                                              env_data.Datasets.EIGHT_EMOTION: {'p': 2, 'k': 10},
                                              env_data.Datasets.FIN_EQU_FUNDS: {'p': 6, 'k': 1},
                                              env_data.Datasets.PHYSIO_EHG: {'p': 4, 'k': 10},
                                              env_data.Datasets.PHYSIO_MGH: {'p': 1, 'k': 10},
                                              env_data.Datasets.PHYSIO_UCD: {'p': 2, 'k': 10},
                                              #
                                              env_data.Datasets.HAPT: {'p': 1, 'k': 10},
                                              env_data.Datasets.PHYSIO_MMG: {'p': 1, 'k': 5},
                                              env_data.Datasets.STFT1: {'p': 2, 'k': 1},
                                              env_data.Datasets.STFT2: {'p': 6, 'k': 1},
                                              env_data.Datasets.STFT3: {'p': 6, 'k': 1},
                                              env_data2d.Datasets.Mario: {'p': 1, 'k': 2},
                                              env_data2d.Datasets.SpaceInvaders: {'p': 1, 'k': 10},
                                              env_data2d.Datasets.Traffic: {'p': 2, 'k': 2}}
}



def get_results(alg, overide_args={}, include_random=True, only_low_dimensional=False):

    results = {}
    
    for args in dataset_args:
        env = args['env']
        dataset = args['dataset']
        print dataset
        if only_low_dimensional and dataset not in datasets_low_dimensional:
            continue
        if not include_random and env is EnvRandom:
            continue
        kwargs = dict(default_args_global)
        kwargs['algorithm'] = alg
        kwargs['measure'] = algorithm_measures[alg]
        kwargs.update(args)
        kwargs.update(dataset_default_args.get(dataset, {}))
        kwargs.update(algorithm_parameters.get(alg, {}).get(dataset, {}))
        kwargs.update(algorithm_args.get(alg, {}))
        kwargs.update(overide_args)
    
        results[dataset] = ep.evaluate(eb.prediction_error, argument_order=['output_dim'], ignore_arguments=['window'], **kwargs)
        
    return results



def get_signals(alg, overide_args={}, include_random=True, only_low_dimensional=False):

    results = {}
    
    for args in dataset_args:
        env = args['env']
        dataset = args['dataset']
        print dataset
        if only_low_dimensional and dataset not in datasets_low_dimensional:
            continue
        if not include_random and env is EnvRandom:
            continue
        kwargs = dict(default_args_global)
        kwargs['algorithm'] = alg
        kwargs['measure'] = algorithm_measures[alg]
        kwargs.update(args)
        kwargs.update(dataset_default_args.get(dataset, {}))
        kwargs.update(algorithm_parameters.get(alg, {}).get(dataset, {}))
        kwargs.update(algorithm_args.get(alg, {}))
        kwargs.update(overide_args)
    
        projected_data, _, [data_train, data_test] = ep.evaluate(eb.calc_projected_data, argument_order=['output_dim'], ignore_arguments=['window'], **kwargs)
        result = {'projected_data': projected_data, 'data_train': data_train, 'data_test': data_test}
        results[dataset] = result
        
    return results



if __name__ == '__main__':
    print get_results(eb.Algorithms.SFA)

