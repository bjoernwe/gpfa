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

    algs = [#eb.Algorithms.Random,
            #eb.Algorithms.ForeCA,
            #eb.Algorithms.SFA,
            #eb.Algorithms.SFFA,
            #eb.Algorithms.PFA,
            eb.Algorithms.GPFA2
            ]

    results_angle = {}
    
    for alg in algs:
        results_angle[alg] = {}
        for min_principal_angle in [False, True]:
            results_angle[alg][min_principal_angle] = parameters.get_results(alg, overide_args={'measure': eb.Measures.angle_to_sfa_signals, 'min_principal_angle': min_principal_angle,  'use_test_set': False})
        
    for _, alg in enumerate(algs):
            
        plt.figure(figsize=(10,6))
        plt.suptitle(plot_alg_names[alg])
            
        for ids, dataset_args in enumerate(parameters.dataset_args):

            for min_principal_angle in [False, True]:

                env = dataset_args['env']
                dataset = dataset_args['dataset']
                if not dataset in results_angle[alg][min_principal_angle]:
                    continue
                
                # angles
                plt.subplot(4, 4, ids+1)
                #plt.subplot2grid(shape=(n_algs,4), loc=(a,3))
                values = results_angle[alg][min_principal_angle][dataset].values
                d, _ = values.shape
                plt.errorbar(x=range(1,d+1), y=np.mean(values, axis=1), yerr=np.std(values, axis=1))
                plt.xlim(.5, 10+.5)
                plt.ylim(-.2, np.pi/2+.2)
                if ids % 4 == 0:
                    plt.ylabel('angle')
                else:
                    plt.gca().set_yticklabels([])
                if ids >= 12:
                    plt.xlabel('# dimensions')
                else:
                    plt.gca().set_xticklabels([])
                    
                # title
                plt.title(eb.get_dataset_name(env=env, ds=dataset, latex=False), fontsize=12)

        plt.subplots_adjust(hspace=.4, wspace=.15, left=0.07, right=.96, bottom=.08, top=.92)
            
    plt.show()



if __name__ == '__main__':
    main()
    
