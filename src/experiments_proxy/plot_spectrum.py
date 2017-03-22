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
    
    algs = [eb.Algorithms.Random,
            eb.Algorithms.SFA,
            eb.Algorithms.ForeCA,
            #eb.Algorithms.SFFA,
            eb.Algorithms.PFA,
            eb.Algorithms.GPFA2
            ]

    results_test  = {}
    results_train = {}
    
    for alg in algs:
        results_test[alg]  = parameters.get_signals(alg)#, repetition_index=range(3))
        results_train[alg] = parameters.get_signals(alg, overide_args={'use_test_set': False})#, repetition_index=range(3))
        
    alphas = np.linspace(0, 1, 6)[::-1]

    for ia, alg in enumerate(algs):
            
        figsize = (10,4.5) if alg is eb.Algorithms.ForeCA else (10,6)
        plt.figure(figsize=figsize)
        plt.suptitle(plot_alg_names[alg])
            
        idx = 0
        for _, dataset_args in enumerate(parameters.dataset_args):

            env = dataset_args['env']
            dataset = dataset_args['dataset']
            if not dataset in results_test[alg]:
                continue

            signals_train = np.mean(results_train[alg][dataset]['projected_data'], axis=2)
            signals_test  = np.mean(results_test[alg][dataset]['projected_data'], axis=2)
            N_train = signals_train.shape[0]
            N_test  = signals_test.shape[0]
            print N_train, N_test

            # plot signals
            n_rows = 3 if alg is eb.Algorithms.ForeCA else 4
            plt.subplot(n_rows, 4, idx+1)
            plt.xlim(-20, N_train//2+20)
            #plt.yscale('log')
            
            # FFT
            for i in range(1)[::-1]:
                spectrum_train = np.abs(np.fft.fft(signals_train[:,i]))[:N_train//2]
                plt.plot(spectrum_train, c='b', alpha=alphas[i])
            for i in range(1)[::-1]:
                spectrum_test  = np.abs(np.fft.fft(signals_test[:,i]))[:N_test//2]
                xscale = N_train * np.arange(N_test//2, dtype=np.float) / N_test
                plt.plot(xscale, -spectrum_test, c='r', alpha=alphas[i])
                
                    
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
    