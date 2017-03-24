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
    
    algs = [#eb.Algorithms.Random,
            eb.Algorithms.SFA,
            eb.Algorithms.ForeCA,
            #eb.Algorithms.SFFA,
            eb.Algorithms.PFA,
            eb.Algorithms.GPFA2
            ]

    results_train = {}
    for alg in algs:
        results_train[alg] = parameters.get_signals(alg, overide_args={'use_test_set': False}, repetition_index=range(3))
        
    #figsize = (10,4.5) if alg is eb.Algorithms.ForeCA else (10,6)
    plt.figure()#figsize=figsize)
            
    for id, dataset_args in enumerate(parameters.dataset_args):
        
        idx = -1

        for ia, alg in enumerate(algs):
            
            idx += 1
            
            env = dataset_args['env']
            dataset = dataset_args['dataset']
            if not dataset in results_train[alg]:
                continue

            signals_train = np.mean(results_train[alg][dataset]['projected_data'], axis=-1)
            N_train = signals_train.shape[0]

            plt.subplot2grid(shape=(16,4), loc=(id, ia))
            #plt.xlim(-20, N_train//2+20)
            plt.xscale('log')
            
            # FFT
            spectrum_train = np.abs(np.fft.fft(signals_train[:,0]))[:N_train//2]
            plt.plot(spectrum_train, c='b')
                    
            # title
            #plt.title(eb.get_dataset_name(env=env, ds=dataset, latex=False), fontsize=12)
            
    plt.show()



if __name__ == '__main__':
    main()
    