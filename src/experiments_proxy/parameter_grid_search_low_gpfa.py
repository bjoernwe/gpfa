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
    
    mkl.set_num_threads(1)

    default_args = {'p':            [1,2,4,6,8,10,15,20,30],
                    'k':            [1,2,5,10],
                    'iterations':   30,
                    'k_eval':       10,
                    'seed':         0,
                    'n_train':      1000, 
                    'n_test':       200,
                    'limit_data':   20000, 
                    'pca':          1.,
                    'noisy_dims':   0,
                    'output_dim':   range(1,6), 
                    'algorithm':    eb.Algorithms.GPFA2, 
                    'measure':      eb.Measures.gpfa,
                    'use_test_set': True,
                    'repetitions':  10,
                    'cachedir':     '/scratch/weghebvc',
                    'manage_seed':  'external',
                    'processes':    None}

    datasets = [{'env': EnvData, 'dataset': env_data.Datasets.EEG},  # p=1, k=2
                {'env': EnvData, 'dataset': env_data.Datasets.EEG2}, # p=1, k=1
                {'env': EnvData, 'dataset': env_data.Datasets.EIGHT_EMOTION},   # p=10, k=1 
                {'env': EnvData, 'dataset': env_data.Datasets.FIN_EQU_FUNDS},   # p=10, k=1
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_EHG},  # p=2, k=1
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MGH},  # p=2, k=2
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MMG},  # p=1, k=2
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_UCD},  # p=1, k=2
                ]
    
    for _, dataset_args in enumerate(datasets):

        # run cross-validation        
        kwargs = dict(default_args)
        kwargs.update(dataset_args)
        result = ep.evaluate(eb.prediction_error, argument_order=['output_dim'], ignore_arguments=['window'], **kwargs)
        result_averaged = np.mean(result.values, axis=(0, -1)) # 1st axis = output_dim, last axis = repetitions

        if len(result.iter_args) == 3:  # grid search
            # print best parameters
            parameters = result.iter_args.items()[1:] # iter_args w/o output_dim
            idc_min = np.unravel_index(np.argmin(result_averaged), result_averaged.shape) # convert to 2D index
            print dataset_args['env'], dataset_args['dataset']
            print '  ', ', '.join(['%s = %d' % (parameters[i][0], parameters[i][1][idx]) for i, idx in enumerate(idc_min)])
            print ''
        elif len(result.iter_args) == 2:
            iter_arg_values = result.iter_args.values()[1]
            plt.figure()
            plt.plot(iter_arg_values, result_averaged)
            plt.xlabel(result.iter_args.keys()[1])
        else:
            assert False
        
    plt.show()



def notify(msg):
    requests.post('http://api.pushover.net/1/messages.json', 
                  data=dict(token='aAKkJ12jjmZgbqjXj4hxKGrYJC9jh3',
                            user='uoKXUWzBShY3k4sfkRowhGCi1gv8w5',
                            message=msg))



if __name__ == '__main__':
    try:
        main()
    finally:
        #notify('finished')
        pass
    plt.show()
    
