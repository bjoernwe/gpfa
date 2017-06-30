import matplotlib.pyplot as plt
import mkl
import numpy as np

import experiments_proxy.experiment_base as eb
import parameters_hi



def main():

    mkl.set_num_threads(1)

    plot_alg_names = {eb.Algorithms.HiSFA:  'hSFA',
                      eb.Algorithms.HiPFA:  'hPFA',
                      eb.Algorithms.HiGPFA: 'hGPFA',
                      }

    algs = [eb.Algorithms.HiPFA,
            eb.Algorithms.HiGPFA
            ]

    results_angle = {}
    
    for alg in algs:
        results_angle[alg] = {}
        for min_principal_angle in [False, True]:
            results_angle[alg][min_principal_angle] = parameters_hi.get_results(alg, overide_args={'measure': eb.Measures.angle_to_sfa_signals, 'min_principal_angle': min_principal_angle, 'use_test_set': False, 'angle_to_hisfa': True})
        
    for _, alg in enumerate(algs):
        
        plt.figure(figsize=(10,2.1))
        plt.suptitle(plot_alg_names[alg])
            
        idx = 0
        for _, dataset_args in enumerate(parameters_hi.dataset_args_hi):

            env = dataset_args['env']
            dataset = dataset_args['dataset']
            if not dataset in results_angle[alg][False]:
                continue
            
            for min_principal_angle in [False, True]:

                # angles
                plt.subplot(1, 3, idx+1)
                #plt.subplot2grid(shape=(n_algs,4), loc=(a,3))
                values = results_angle[alg][min_principal_angle][dataset].values
                print values.shape
                d, _ = values.shape
                plt.errorbar(x=range(1,d+1), y=np.mean(values, axis=1), yerr=np.std(values, axis=1))
                plt.xlim(.5, 5.5)
                plt.ylim(-.2, np.pi/2+.2)
                if idx % 4 == 0:
                    plt.ylabel('angle')
                else:
                    plt.gca().set_yticklabels([])
                if idx >= 0:
                    plt.xlabel('# dimensions')
                else:
                    plt.gca().set_xticklabels([])
                    
                # title
                plt.title(eb.get_dataset_name(env=env, ds=dataset, latex=False), fontsize=12)
                
            idx += 1

        plt.subplots_adjust(hspace=.4, wspace=.15, left=0.16, right=.84, bottom=.25, top=.79)
        plt.savefig('fig_angles_%s.eps' % plot_alg_names[alg].lower())
        
    plt.show()



if __name__ == '__main__':
    main()
    
