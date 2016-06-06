import matplotlib.pyplot as plt
import numpy as np
import requests

from collections import OrderedDict

import experiments.experiment_base as eb

import plot



def main():

    default_args = {'N2': 5000, 
                    'seed': 1, 
                    'noisy_dims': 0, 
                    'repetitions': 25, 
                    'include_random': True, 
                    'include_sfa':    True, 
                    'include_sffa':   True, 
                    'include_foreca': False, # 
                    'include_gfa1':   True, 
                    'include_gfa2':   True,
                    'use_test_set':   True,  #
                    'cachedir': '/scratch/weghebvc',
                    'manage_seed': 'auto',
                    'processes': None}

    datasets = [#({'dataset': eb.Datasets.EEG,           'N': 10000, 'k': 2, 'p': 2, 'K': 0, 'keep_variance':  1., 'output_dim': 2, 'iterations': 50, 'k_eval': 10}, {}, None, None),
                #({'dataset': eb.Datasets.EEG2,          'N': 10000, 'k': 2, 'p': 2, 'K': 0, 'keep_variance':  1., 'output_dim': 2, 'iterations': 50, 'k_eval': 10}, {}, None, None),
                #({'dataset': eb.Datasets.EEG2_stft_128, 'N': 10000, 'k': 2, 'p': 2, 'K': 0, 'keep_variance': .98, 'output_dim': 2, 'iterations': 50, 'k_eval': 10}, {}, None, None),
                #({'dataset': eb.Datasets.WAV_11k,       'N':  6000, 'k': 2, 'p': 1, 'K': 0, 'keep_variance': .98, 'output_dim': 5, 'iterations': 50}, {'N': [2000, 3000, 4000, 5000, 6000]}),
                #({'dataset': eb.Datasets.WAV2_22k,      'N':  7000, 'k': 2, 'p': 6, 'K': 1, 'keep_variance': .96, 'output_dim': 5, 'iterations': 50}, {'N': [2000, 3000, 4000, 5000, 6000, 7000]}),
                ({'dataset': eb.Datasets.WAV_22k,       'N': 10000, 'k': 10, 'p': 5, 'K': 10, 'keep_variance': .99, 'output_dim': 5, 'iterations': 50, 'k_eval': 10}, {}, (0, 4),   (0,6)),
                ({'dataset': eb.Datasets.WAV3_22k,      'N': 10000, 'k':  2, 'p': 5, 'K':  4, 'keep_variance': .99, 'output_dim': 5, 'iterations': 50, 'k_eval': 10}, {}, (0, 5),   (0,7)),
                ({'dataset': eb.Datasets.WAV4_22k,      'N': 10000, 'k': 15, 'p': 5, 'K':  0, 'keep_variance': .99, 'output_dim': 5, 'iterations': 50, 'k_eval': 10}, {}, (-1, 15), None),
                ]
    
    experiments = OrderedDict([('p', (range(1,9), .1, (0, 9))), 
                               ('k', ([2, 5, 10, 15, 20, 30, 50], .5, (-1, 55))),
                               ('k_eval', ([2, 5, 10, 15, 20, 30, 50], .5, (-1, 55))),
                               ('N', ([2000, 4000, 6000, 8000, 10000, 12000], 50, (1000, 13000))),
                               ('iterations', ([1, 10, 30, 50, 100], 1, (-10, 110))),
                               ('output_dim', (range(1,11), .1, (0, 11))),
                               #('K': range(11))
                               #('keep_variance': list(np.arange(.94, .999, .02))), 
                               ])
    
    for d, (dataset_args, overide_args, ylim, ylim2) in enumerate(datasets):
        
        #plt.figure(figsize=(12., 10.))
        #plt.suptitle('%s\n%s' % (dataset_args['dataset'], default_args))        
        
        for i, (experiment_arg, (experiment_values, x_offset, xlim)) in enumerate(experiments.items()):
        
            kwargs = dict(default_args)
            kwargs.update(dataset_args)
            kwargs[experiment_arg] = experiment_values
            
            if experiment_arg in overide_args:
                kwargs[experiment_arg] = overide_args[experiment_arg]

            #plt.subplot(2, 3, i+1)
            plt.figure(figsize=(5*1.4, 3*1.4))
            legend = True if i==5 else False
            y_label = i%3 == 0
            plot.plot_experiment(legend=legend, x_offset=x_offset, y_label=y_label, **kwargs)
            
            plt.xlabel(plt.gca().xaxis.label.get_text() + ' (default: %s)' % dataset_args.get(experiment_arg, None))
            plt.title('abcdef'[i])
            plt.xlim(xlim)
            plt.ylim(ylim)
            if i==5 and ylim2 is not None:
                plt.ylim(ylim2)
            
            plt.savefig('results_%d%s.eps' % ((d+1), 'abcdef'[i]))

    #plt.show()



def notify(msg):
    requests.post('http://api.pushover.net/1/messages.json', 
                  data=dict(token='aAKkJ12jjmZgbqjXj4hxKGrYJC9jh3',
                            user='uoKXUWzBShY3k4sfkRowhGCi1gv8w5',
                            message=msg))



if __name__ == '__main__':
    try:
        main()
    finally:
        notify('finished')
    plt.show()
    
