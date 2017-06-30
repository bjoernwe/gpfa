import matplotlib
import matplotlib.pyplot as plt
import mkl
import numpy as np

import experiments_proxy.experiment_base as eb
import parameters



def main():

    mkl.set_num_threads(1)

    plot_alg_names = {eb.Algorithms.None:   'input data',
                      eb.Algorithms.Random: 'Random',
                      eb.Algorithms.SFA:    'SFA',
                      eb.Algorithms.SFFA:   "SFA'",
                      eb.Algorithms.ForeCA: 'ForeCA',
                      eb.Algorithms.PFA:    'PFA',
                      eb.Algorithms.GPFA2:  'GPFA',
                      }
    
    #algs = [eb.Algorithms.SFA,
    #        ]
    #repetitions = 3 #parameters.default_args_global['repetitions']

    results = {}
    #for alg in algs:
    results = parameters.get_signals(eb.Algorithms.SFA, overide_args={'use_test_set': False}, repetition_index=0)
        
    figsize = (20,11)
    plt.figure(figsize=figsize)
            
    for idat, dataset_args in enumerate(parameters.dataset_args):

        env = dataset_args['env']
        dataset = dataset_args['dataset']

        plt.subplot(4, 4, idat+1)
        plt.title(dataset)
        
        if not dataset in results:
            continue
        
        #plt.subplot2grid(shape=(4,4), loc=(idat,ialg))
        model = results[dataset]['model']
        deltas = model.d
        plt.plot(deltas)
        plt.ylim([0,2])

#             plt.plot(power_spectrum, c='b')
#             plt.xticks([])
#             plt.yticks([])
#             margin = signal_length // 60
#             plt.xlim([-margin, signal_length//2 + margin])
#             if idat == 0:
#                 plt.title(plot_alg_names[alg], fontsize=12)
#             elif idat == 15:
#                 plt.xlabel('frequencies')
#             if ialg == 0:
#                 plt.ylabel(eb.get_dataset_name(env=env, ds=dataset), rotation=0, horizontalalignment='right', verticalalignment='top')
        
#     plt.subplot2grid(shape=(16,5), loc=(0,2))
#     plt.title('ForeCA', fontsize=12)
#     plt.gca().axis('off')

    plt.subplots_adjust(left=0.1, right=.99, bottom=0.04, top=.96)
    plt.savefig('fig_deltas.eps')
    plt.show()



if __name__ == '__main__':
    main()
    
