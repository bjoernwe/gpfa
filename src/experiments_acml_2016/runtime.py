import matplotlib.pyplot as plt
import numpy as np
import requests

from collections import OrderedDict

import experiments.experiment_base as eb

import plot


def main():
    
    eb.set_cachedir(cachedir=None)

    default_args = {'N2': 100, 
                    'seed': 0, 
                    'keep_variance': 1., 
                    'repetitions': 5, 
                    'include_random': False, #
                    'include_sfa':    True, 
                    'include_foreca': True, 
                    'include_gfa1':   True, 
                    'include_gfa2':   True, 
                    'use_test_set':   True,
                    'plot_time':      True,
                    'cachedir': '/scratch/weghebvc/timing',
                    'processes': 1}

    datasets = [({'dataset': eb.Datasets.Random, 'N': 300, 'k': 10, 'p': 2, 'K': 0, 'noisy_dims': 10, 'output_dim': 1, 'iterations': 50, 'k_eval': 10}, {}),
                ]
    
    experiments = OrderedDict([#('p', range(1,11)), 
                               #('k_eval', [2, 5, 10, 15, 20, 30, 50]),
                               ('N', (range(100, 501, 50), (90, 510))),# + [1500, 2000]),
                               #('iterations', [1, 10, 20, 50, 100]),#, 150],
                               #('output_dim', range(1,11)),
                               #('K', range(11)),
                               ('noisy_dims', ([2, 10, 20, 50, 100, 150, 200], (-8, 210))),
                               ('k', ([2, 5, 10, 15, 20, 25, 30], (0, 32))),
                               ])
    
    for d, (dataset_args, overide_args) in enumerate(datasets):
        
        #plt.figure(figsize=(7.7, 4.5))
        #plt.suptitle('%s\n%s' % (dataset_args['dataset'], default_args))        
        
        for i, (experiment_arg, (experiment_values, xlim)) in enumerate(experiments.items()):
        
            kwargs = dict(default_args)
            kwargs.update(dataset_args)
            kwargs[experiment_arg] = experiment_values
            
            if experiment_arg in overide_args:
                kwargs[experiment_arg] = overide_args[experiment_arg]

            #plt.subplot(2, 2, i+1)
            plt.figure(figsize=(5, 4.5))
            legend = True if i==2 else False
            y_label = i%2 == 0
            plot.plot_experiment(legend=legend, y_label=y_label, **kwargs)
            plt.xlabel(plt.gca().xaxis.label.get_text() + ' (default: %s)' % dataset_args.get(experiment_arg, None))
            plt.title('abcdef'[i])
            plt.xlim(xlim)
            plt.savefig('runtime_%d%s.eps' % ((d+1), 'abcdef'[i]))
            
        #plt.savefig('results_%d.eps' % (d+1))

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
    
