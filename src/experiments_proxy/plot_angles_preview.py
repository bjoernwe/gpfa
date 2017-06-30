import matplotlib.pyplot as plt
import mkl
import numpy as np

import experiments_proxy.experiment_base as eb
import parameters

from envs import env_data
from envs import env_data2d
from envs.env_data import EnvData
from envs.env_data2d import EnvData2D
from envs.env_random import EnvRandom


def main():

    mkl.set_num_threads(1)

    plot_alg_names = {eb.Algorithms.Random: 'Random',
                      eb.Algorithms.SFA:    'SFA',
                      eb.Algorithms.SFFA:   "SFA'",
                      eb.Algorithms.ForeCA: 'ForeCA',
                      eb.Algorithms.PFA:    'PFA',
                      eb.Algorithms.GPFA2:  'GPFA',
                      }

    algs = [#eb.Algorithms.Random,
            eb.Algorithms.ForeCA,
            #eb.Algorithms.SFA,
            #eb.Algorithms.SFFA,
            eb.Algorithms.PFA,
            eb.Algorithms.GPFA2
            ]
    
    datasets = [{'env': EnvData, 'dataset': env_data.Datasets.EEG},
                {'env': EnvData, 'dataset': env_data.Datasets.EIGHT_EMOTION},
                {'env': EnvData, 'dataset': env_data.Datasets.PHYSIO_UCD}
                ]


    results_angle = {}
    
    for alg in algs:
        results_angle[alg] = {}
        for min_principal_angle in [False, True]:
            results_angle[alg][min_principal_angle] = parameters.get_results(alg, overide_args={'measure': eb.Measures.angle_to_sfa_signals, 'min_principal_angle': min_principal_angle, 'use_test_set': False})
        
    for _, dataset_dict in enumerate(datasets):
        
        env = dataset_dict['env']
        dataset = dataset_dict['dataset']
        #if not dataset in results_angle[alg][False]:
        #    continue
        
        figsize = (10,2.5)# if alg is eb.Algorithms.ForeCA else (10,6)
        plt.figure(figsize=figsize)
        #plt.suptitle(plot_alg_names[alg])
        plt.suptitle(eb.get_dataset_name(env=env, ds=dataset), fontsize=12)
            
        idx = 0
        for _, alg in enumerate(algs):
        
            for min_principal_angle in [False, True]:

                # angles
                n_rows = 1#3 if alg is eb.Algorithms.ForeCA else 4
                plt.subplot(n_rows, 3, idx+1)
                #plt.subplot2grid(shape=(n_algs,4), loc=(a,3))
                values = results_angle[alg][min_principal_angle][dataset].values
                d, _ = values.shape
                plt.errorbar(x=range(1,d+1), y=np.mean(values, axis=1), yerr=np.std(values, axis=1))
                xlim_max = 5.5 #if alg is eb.Algorithms.ForeCA else 10.5 
                plt.xlim(.5, xlim_max)
                plt.ylim(-.2, np.pi/2+.2)
                if idx % 4 == 0:
                    plt.ylabel('angle')
                    #plt.gcf().canvas.draw()
                    #labels = [item.get_text() for item in plt.gca().get_xticklabels()]
                    #labels[1] = "$0$"
                    #labels[4] = "$\pi/2$"
                    #plt.gca().set_yticklabels(labels)
                else:
                    plt.gca().set_yticklabels([])
                plt.xlabel('M')
                    
            # title
            plt.title(plot_alg_names[alg], fontsize=12)
                    
            idx += 1
    
        plt.subplots_adjust(hspace=.4, wspace=.15, left=0.07, right=.96, bottom=.18, top=.82)            
        plt.savefig('fig_angles_prev_%s.eps' % eb.get_dataset_name(env=env, ds=dataset).lower())
    plt.show()



if __name__ == '__main__':
    main()
    
