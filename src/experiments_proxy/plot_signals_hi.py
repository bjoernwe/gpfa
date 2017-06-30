import matplotlib.pyplot as plt
import mkl
import numpy as np

import experiments_proxy.experiment_base as eb
import parameters_hi



def main():

    mkl.set_num_threads(1)

    algs = [eb.Algorithms.HiSFA,
            #eb.Algorithms.ForeCA,
            #eb.Algorithms.SFFA,
            #eb.Algorithms.HiPFA,
            #eb.Algorithms.HiGPFA,
            ]
    repetition_index = 0

    results_test  = {}
    results_train = {}
    
    for alg in algs:
        results_test[alg]  = parameters_hi.get_signals(alg, overide_args={}, repetition_index=repetition_index)
        results_train[alg] = parameters_hi.get_signals(alg, overide_args={'use_test_set': False}, repetition_index=repetition_index)
        
    alphas = np.linspace(0, 1, 11)[::-1]

    for ia, alg in enumerate(algs):
            
        plt.figure()
        plt.suptitle(alg)
            
        for ids, dataset_args in enumerate(parameters_hi.dataset_args_hi):
            
            dataset = dataset_args['dataset']
            if not dataset in results_test[alg]:
                continue

            signals_train = results_train[alg][dataset]['projected_data']
            signals_test  = results_test[alg][dataset]['projected_data']
            N_train = signals_train.shape[0]
            N_test  = signals_test.shape[0]
            print N_train, N_test

            plt.subplot(4, 4, ids+1)
            
            for i in range(3)[::-1]:
                signal_train = signals_train[:,i]
                signal_test  = signals_test[:,i]
                signal_train = signal_train[-10000:]
                signal_test  = signal_test[:10000]
                n_train = signal_train.shape[0]
                n_test  = signal_test.shape[0]
                sign = np.sign(np.correlate(signal_train, results_train[eb.Algorithms.HiSFA][dataset]['projected_data'][:,i])[0])
                plt.plot(range(n_train), sign*signal_train, c='b', alpha=alphas[i])
                plt.plot(range(n_train, n_train+n_test), sign*signal_test, c='r', alpha=alphas[i])
                plt.ylabel(alg)

            # title
            #dataset_str = '%s<%s>' % (dataset_args['env'], dataset_args['dataset'])
            dataset_str = '%s' % (dataset_args['dataset'])
            plt.title(dataset_str)
            
        plt.subplots_adjust(hspace=.4, wspace=.3, left=0.02, right=.98, bottom=.02, top=.95)
        
    plt.show()



if __name__ == '__main__':
    main()
    