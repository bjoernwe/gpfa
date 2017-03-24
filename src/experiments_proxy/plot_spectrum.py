import matplotlib
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
    
    algs = [eb.Algorithms.None,
            #eb.Algorithms.Random,
            eb.Algorithms.SFA,
            eb.Algorithms.ForeCA,
            #eb.Algorithms.SFFA,
            eb.Algorithms.PFA,
            eb.Algorithms.GPFA2
            ]
    
    repetitions = parameters.default_args_global['repetitions']

    results_train = {}
    for alg in algs:
        results_train[alg] = parameters.get_signals(alg, overide_args={'use_test_set': False}, repetition_index=range(repetitions))
        
    for ia, alg in enumerate(algs):
            
        #figsize = (10,4.5) if alg is eb.Algorithms.ForeCA else (10,6)
        plt.figure()#figsize=figsize)
        plt.suptitle(plot_alg_names[alg])
            
        idx = 0
        for _, dataset_args in enumerate(parameters.dataset_args):

            env = dataset_args['env']
            dataset = dataset_args['dataset']
            if not dataset in results_train[alg]:
                continue

            signals_train = np.mean(results_train[alg][dataset]['projected_data'], axis=-1)
            N_train, D_train = signals_train.shape
            print (N_train, D_train)

            # plot signals
            n_rows = 3 if alg is eb.Algorithms.ForeCA else 4
            plt.subplot(n_rows, 4, idx+1)
            plt.xlim(-20, N_train//2+20)
            #plt.yscale('log')
            
            # FFT
            if alg is not eb.Algorithms.None:
                spectrum_train = np.abs(np.fft.fft(signals_train[:,0]))[:N_train//2]
            else:
                spectrum_train = np.abs(np.fft.fft(np.mean(signals_train, axis=1)))[:N_train//2]
            plt.plot(spectrum_train, c='b')
                    
            # title
            plt.title(eb.get_dataset_name(env=env, ds=dataset, latex=False), fontsize=12)
            
            idx += 1
            

    if alg is eb.Algorithms.ForeCA:
        plt.subplots_adjust(hspace=.4, wspace=.15, left=0.07, right=.96, bottom=.1, top=.88)
    else:
        plt.subplots_adjust(hspace=.4, wspace=.15, left=0.07, right=.96, bottom=.08, top=.92)
        
    plt.show()



if __name__ == '__main__':
    main()
    