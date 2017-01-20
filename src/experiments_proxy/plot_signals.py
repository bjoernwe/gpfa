import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import experiments_proxy.experiment_base as eb
import parameters



def main():

    algs = [eb.Algorithms.ForeCA,
            eb.Algorithms.SFA,
            #eb.Algorithms.SFFA,
            eb.Algorithms.PFA,
            eb.Algorithms.GPFA2
            ]
    n_algs = len(algs)

    repetition_index = 0

    results_test  = {}
    results_train = {}
    
    for alg in algs:
        only_low_dimensional = alg is eb.Algorithms.ForeCA
        results_test[alg]  = parameters.get_signals(alg, only_low_dimensional=only_low_dimensional, repetition_index=repetition_index)
        results_train[alg] = parameters.get_signals(alg, only_low_dimensional=only_low_dimensional, overide_args={'use_test_set': False}, repetition_index=repetition_index)
        
    
    alphas = np.linspace(0, 1, 6)[::-1]

        #only_low_dimensional = alg is eb.Algorithms.ForeCA
        #results_test  = [parameters.get_signals(alg, only_low_dimensional=only_low_dimensional, repetition_index=i) for i in range(3)]
        #results_train = [parameters.get_signals(alg, only_low_dimensional=only_low_dimensional, overide_args={'use_test_set': False}, repetition_index=i) for i in range(3)]
        
    for dataset_args in parameters.dataset_args:

        plt.figure()
            
        for a, alg in enumerate(algs):
            
            dataset = dataset_args['dataset']
            if not dataset in results_test[alg]:
                continue
            
            signals_train = results_train[alg][dataset]['projected_data']
            signals_test  = results_test[alg][dataset]['projected_data']
            N_train = signals_train.shape[0]
            N_test  = signals_test.shape[0]
            print N_train, N_test

            # plot signals
            plt.subplot2grid(shape=(n_algs,4), loc=(a,0), colspan=2)
            for i in range(5)[::-1]:
                signal_train = signals_train[:,i]
                signal_test  = signals_test[:,i]
                signal_train = signal_train[-1000:]
                signal_test  = signal_test[:1000]
                n_train = signal_train.shape[0]
                n_test  = signal_test.shape[0]
                plt.plot(range(n_train), signal_train, c='b', alpha=alphas[i])
                plt.plot(range(n_train, n_train+n_test), signal_test, c='r', alpha=alphas[i])
                plt.ylabel(alg)
            # FFT
            plt.subplot2grid(shape=(n_algs,4), loc=(a,2))
            for i in range(5)[::-1]:
                spectrum_train = np.abs(np.fft.fft(signals_train[:,i]))[:N_train//2]
                plt.plot(spectrum_train, c='b', alpha=alphas[i])
            plt.subplot2grid(shape=(n_algs,4), loc=(a,3))
            for i in range(5)[::-1]:
                spectrum_test  = np.abs(np.fft.fft(signals_test[:,i]))[:N_test//2]
                plt.plot(spectrum_test, c='r', alpha=alphas[i])
                
            # title
            dataset_str = '%s<%s>' % (dataset_args['env'], dataset_args['dataset'])
            plt.suptitle(dataset_str)
            
        # 
        #plt.plot([1e-6, 1e2], [1e-6, 1e2], '-', zorder=3)
        #plt.xlabel('dimensions of input data')
        #plt.xscale('log')
        #plt.yscale('log')
        #plt.legend(loc='best', prop={'size': 8})
        
    plt.show()



if __name__ == '__main__':
    main()
    