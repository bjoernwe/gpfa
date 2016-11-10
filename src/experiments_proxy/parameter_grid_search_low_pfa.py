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



def main():
    
    mkl.set_num_threads(1)

    default_args = {'p':            [1,2,4,6,8,10],
                    'K':            [0,1] + range(2, 11, 2),
                    'seed':         0,
                    'n_train':      1000, 
                    'n_test':       200, 
                    'limit_data':   20000,
                    'pca':          1.,
                    'noisy_dims':   0,
                    'output_dim':   range(1,6), 
                    'algorithm':    eb.Algorithms.PFA, 
                    'measure':      eb.Measures.pfa,
                    'use_test_set': True,
                    'repetitions':  10,
                    'cachedir':     '/scratch/weghebvc',
                    'manage_seed':  'external',
                    'processes':    None}

    datasets = [{'env': EnvData, 'dataset': env_data.Datasets.EEG},
                {'env': EnvData, 'dataset': env_data.Datasets.EEG2},
                {'env': EnvData, 'dataset': env_data.Datasets.EIGHT_EMOTION},
                {'env': EnvData, 'dataset': env_data.Datasets.FIN_EQU_FUNDS},
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_EHG},
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MGH},
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MMG},
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_UCD},
                ]
    
    # dict to store results in
    result_dict = {}
    
    for _, dataset_args in enumerate(datasets):

        # run cross-validation        
        kwargs = dict(default_args)
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
        else:
            assert False
        
    # print results as code ready to copy & paste
    print '{'
    for dataset, values in result_dict.items():
        print '                       env_data.%s: %s,' % (dataset.__str__(), values)
    print '}'
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
    
