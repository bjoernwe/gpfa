import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import experiments_proxy.experiment_base as eb
import parameters



def main():

    algs = [eb.Algorithms.SFA,
            eb.Algorithms.ForeCA,
            #eb.Algorithms.SFFA,
            eb.Algorithms.PFA,
            eb.Algorithms.GPFA2
            ]
    n_algs = len(algs)

    results_test  = {}
    results_train = {}
    
    for alg in algs:
        only_low_dimensional = alg is eb.Algorithms.ForeCA
        results_test[alg]  = parameters.get_signals(alg, only_low_dimensional=only_low_dimensional, repetition_index=range(3))
        results_train[alg] = parameters.get_signals(alg, only_low_dimensional=only_low_dimensional, overide_args={'use_test_set': False}, repetition_index=range(3))
        
    alphas = np.linspace(0, 1, 6)[::-1]

    for ia, alg in enumerate(algs):
            
        plt.figure()
        plt.suptitle(alg)
            
        for ids, dataset_args in enumerate(parameters.dataset_args):
            
            dataset = dataset_args['dataset']
            if not dataset in results_test[alg]:
                continue

            signals_train = np.mean(results_train[alg][dataset]['projected_data'], axis=2)
            signals_test  = np.mean(results_test[alg][dataset]['projected_data'], axis=2)
            N_train = signals_train.shape[0]
            N_test  = signals_test.shape[0]
            print N_train, N_test

            # plot signals
            plt.subplot(4, 4, ids+1)
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
            #dataset_str = '%s<%s>' % (dataset_args['env'], dataset_args['dataset'])
            dataset_str = '%s' % (dataset_args['dataset'])
            plt.title(dataset_str)
            
        plt.subplots_adjust(hspace=.4, wspace=.3, left=0.02, right=.98, bottom=.02, top=.95)
        
    plt.show()



if __name__ == '__main__':
    main()
    