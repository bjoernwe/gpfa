import matplotlib.pyplot as plt
import mkl
import numpy as np

import experiments_proxy.experiment_base as eb
import parameters



def main():

    mkl.set_num_threads(1)

    plot_alg_names = {eb.Algorithms.Random: 'Random',
                      eb.Algorithms.SFA:    'SFA',
                      eb.Algorithms.SFFA:   "SFA'",
                      eb.Algorithms.ForeCA: 'ForeCA',
                      eb.Algorithms.PFA:    'PFA',
                      eb.Algorithms.GPFA2:  'GPFA',
                      }

    algs = [eb.Algorithms.PFA,
            eb.Algorithms.GPFA2
            ]

    search_range_p = {eb.Algorithms.PFA:   [1,2,4,6,8,10],
                      eb.Algorithms.GPFA2: [1,2,4,6]}

    results_angle = {}
    
    for alg in algs:
        results_angle[alg] = {}
        #for min_principal_angle in [False, True]:
        override_args = {'measure': eb.Measures.angle_to_p1, 
                         #'min_principal_angle': min_principal_angle,
                         'principal_angle_idx' : range(5), 
                         'use_test_set': False,
                         'output_dim': 5,
                         'output_dim_max': 5,
                         'p': search_range_p[alg]}
        results_angle[alg] = parameters.get_results(alg, overide_args=override_args)
        
    for _, alg in enumerate(algs):
        
        figsize = (10,4.5) if alg is eb.Algorithms.ForeCA else (10,6)
        plt.figure(figsize=figsize)
        plt.suptitle(plot_alg_names[alg])
            
        idx = 0
        for _, dataset_args in enumerate(parameters.dataset_args):

            env = dataset_args['env']
            dataset = dataset_args['dataset']
            if not dataset in results_angle[alg]:
                continue
            
            for i in range(5):

                # angles
                n_rows = 3 if alg is eb.Algorithms.ForeCA else 4
                plt.subplot(n_rows, 4, idx+1)
                #plt.subplot2grid(shape=(n_algs,4), loc=(a,3))
                values = results_angle[alg][dataset].values[i]
                d, _ = values.shape
                plt.errorbar(x=range(1,d+1), y=np.mean(values, axis=1), yerr=np.std(values, axis=1))
                plt.xlim(.5, 6.5)
                plt.ylim(-.2, np.pi/2+.2)
                if idx % 4 == 0:
                    plt.ylabel('angle')
                else:
                    plt.gca().set_yticklabels([])
                if (alg is eb.Algorithms.ForeCA and idx in [5,6,7,8]) or idx >= 12:
                    plt.xlabel('p')
                else:
                    plt.gca().set_xticklabels([])
                    
                # title
                plt.title(eb.get_dataset_name(env=env, ds=dataset, latex=False), fontsize=12)
                
            idx += 1

        if alg is eb.Algorithms.ForeCA:
            plt.subplots_adjust(hspace=.4, wspace=.15, left=0.07, right=.96, bottom=.1, top=.88)
        else:
            plt.subplots_adjust(hspace=.4, wspace=.15, left=0.07, right=.96, bottom=.08, top=.92)
            
        plt.savefig('fig_angles_to_p1_%s.eps' % plot_alg_names[alg].lower())
    plt.show()



if __name__ == '__main__':
    main()
    
