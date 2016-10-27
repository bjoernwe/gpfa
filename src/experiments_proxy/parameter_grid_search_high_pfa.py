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
    
    #mkl.set_num_threads(1)

    default_args = {'p':            range(1,11),
                    'K':            [0,1] + range(2, 13, 2),
                    'seed':         0,
                    'n_train':      1000, 
                    'n_test':       200, 
                    'limit_data':   20000,
                    'pca':          .99,
                    'noisy_dims':   0,
                    'output_dim':   range(1,6), 
                    'algorithm':    eb.Algorithms.PFA, 
                    'measure':      eb.Measures.pfa,
                    'use_test_set': True,
                    'repetitions':  5,
                    'cachedir':     '/scratch/weghebvc',
                    'manage_seed':  'external',
                    'processes':    1}

    datasets = [#{'env': EnvKai, 'dataset': None, 'noisy_dims': 10, 'pca': 1., 'output_dim': [1,2]},
                {'env': EnvData, 'dataset': env_data.Datasets.EEG,  'pca': 1.}, # K=0,  p=10
                {'env': EnvData, 'dataset': env_data.Datasets.EEG2, 'pca': 1.}, # K=0,  p=10
                {'env': EnvData, 'dataset': env_data.Datasets.FIN_EQU_FUNDS, 'pca': 1.},
                #{'env': EnvData, 'dataset': env_data.Datasets.HAPT, 'n_train': 5000, 'n_test': 2500},
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_EHG, 'pca': 1.},
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MGH, 'pca': 1.},
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_MMG, 'pca': 1.},
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_UCD, 'pca': 1.},
                #{'env': EnvData, 'dataset': env_data.Datasets.STFT1},           # K=10, p=8
                #{'env': EnvData, 'dataset': env_data.Datasets.STFT2},           # K=0,  p=1
                #{'env': EnvData, 'dataset': env_data.Datasets.STFT3},           # K=0,  p=1
                #{'env': EnvData2D, 'dataset': env_data2d.Datasets.Mario,   'window': ((70,70),(90,90))},        # K=0, p=3
                #{'env': EnvData2D, 'dataset': env_data2d.Datasets.Traffic, 'window': ((35,65),(55,85))},        # K=1, p=3
                #{'env': EnvData2D, 'dataset': env_data2d.Datasets.SpaceInvaders, 'window': ((16,30),(36,50))},  # K=1, p=8
                ]
    
    for _, dataset_args in enumerate(datasets):

        # run cross-validation        
        kwargs = dict(default_args)
        kwargs.update(dataset_args)
        result = ep.evaluate(eb.prediction_error, argument_order=['output_dim'], ignore_arguments=['window'], **kwargs)

        # print best parameters
        parameters = result.iter_args.items()[1:] # iter_args w/o output_dim
        result_averaged = np.mean(result.values, axis=(0, -1)) # 1st axis = output_dim, last axis = repetitions 
        idc_min = np.unravel_index(np.argmin(result_averaged), result_averaged.shape) # convert to 2D index
        print dataset_args['env'], dataset_args['dataset']
        print '  ', ', '.join(['%s = %d' % (parameters[i][0], parameters[i][1][idx]) for i, idx in enumerate(idc_min)])
        
        #plt.plot(result_averaged.T)
        #plt.show()

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
    
