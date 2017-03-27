import matplotlib
import matplotlib.pyplot as plt
import mkl
import numpy as np

import experiments_proxy.experiment_base as eb
import parameters



def main():

    mkl.set_num_threads(1)

    plot_alg_names = {eb.Algorithms.None:   'input data',
                      eb.Algorithms.Random: 'Random',
                      eb.Algorithms.SFA:    'SFA',
                      eb.Algorithms.SFFA:   "SFA'",
                      eb.Algorithms.ForeCA: 'ForeCA',
                      eb.Algorithms.PFA:    'PFA',
                      eb.Algorithms.GPFA2:  'GPFA',
                      }
    
    algs = [eb.Algorithms.None,
            eb.Algorithms.Random,
            eb.Algorithms.SFA,
            eb.Algorithms.ForeCA,
            #eb.Algorithms.SFFA,
            eb.Algorithms.PFA,
            eb.Algorithms.GPFA2
            ]
    
    repetitions = 10 #parameters.default_args_global['repetitions']

    results_train = {}
    for alg in algs:
        stack_result = not alg is eb.Algorithms.None
        results_train[alg] = parameters.get_signals(alg, overide_args={'use_test_set': False}, repetition_index=range(repetitions), stack_result=stack_result)
        
    #figsize = (10,4.5) if alg is eb.Algorithms.ForeCA else (10,6)
    plt.figure()#figsize=figsize)
            
    for idat, dataset_args in enumerate(parameters.dataset_args):

        for ialg, alg in enumerate(algs):
            
            dataset = dataset_args['dataset']
            if not dataset in results_train[alg]:
                continue
            
            if alg is eb.Algorithms.None:
                spectra_list = []
                for input_data in results_train[alg][dataset]['projected_data']:
                    input_avg = np.mean(input_data, axis=-1)    # average over dimensions
                    spectrum = np.fft.fft(input_avg)
                    spectra_list.append(spectrum)
                spectra = np.vstack(spectra_list).T
                spectrum = np.mean(spectra, axis=-1)    # average over experiments (repetitions)
            else:
                avg_signals = np.mean(results_train[alg][dataset]['projected_data'], axis=-1) # average over repetitions
                avg_first_signal = avg_signals[:,0]
                spectrum = np.fft.fft(avg_first_signal)

            N_train = spectrum.shape[0]
            power_spectrum = np.abs(spectrum)[:N_train//2]

            plt.subplot2grid(shape=(16,6), loc=(idat,ialg))
            plt.plot(power_spectrum, c='b')
            if idat == 0:
                plt.title(alg)
        
    plt.show()



if __name__ == '__main__':
    main()
    
