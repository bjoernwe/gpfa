import matplotlib
import matplotlib.pyplot as plt
import mkl
import numpy as np

import experiments_proxy.experiment_base as eb
import parameters
import parameters_hi


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
            #eb.Algorithms.Random,
            eb.Algorithms.SFA,
            eb.Algorithms.ForeCA,
            #eb.Algorithms.SFFA,
            eb.Algorithms.PFA,
            eb.Algorithms.GPFA2
            ]
    
    algs_hi = [eb.Algorithms.HiSFA,
               eb.Algorithms.HiPFA,
               eb.Algorithms.HiGPFA,
               ]
    
    repetitions = parameters.default_args_explot['repetitions']
    repetitions_hi = parameters_hi.default_args_explot['repetitions']

    results_train = {}
    for alg in algs:
        stack_result = not alg is eb.Algorithms.None
        r = parameters.algorithm_args[alg].get('repetitions', repetitions)
        results_train[alg] = parameters.get_signals(alg, overide_args={'use_test_set': False, 'output_dim': 1, 'seed': range(r)}, stack_result=stack_result)
    for alg in algs_hi:
        r = parameters_hi.algorithm_args[alg].get('repetitions', repetitions_hi)
        results_train[alg] = parameters_hi.get_signals(alg, overide_args={'use_test_set': False, 'output_dim': 1, 'seed': range(r)})
        
    figsize = (20,11)
    plt.figure(figsize=figsize)
            
    for idat, dataset_args in enumerate(parameters.dataset_args):

        for ialg, alg in enumerate(algs + algs_hi):
            
            env = dataset_args['env']
            dataset = dataset_args['dataset']
            if not dataset in results_train[alg]:
                continue
            
            spectra_list = []
            if alg is eb.Algorithms.None:
                for input_data in results_train[alg][dataset]['projected_data']:
                    for signal in input_data.T:
                        spectrum = np.fft.fft(signal)
                        spectra_list.append(spectrum)
            else:
                for signal in results_train[alg][dataset]['projected_data'][:,0,:].T:
                    spectrum = np.fft.fft(signal)
                    spectra_list.append(spectrum)
            spectra = np.vstack(spectra_list).T
            spectrum = np.mean(spectra, axis=-1)    # average over repetitions and dimensions

            signal_length = spectrum.shape[0]
            power_spectrum = np.abs(spectrum)[:signal_length//2]

            plt.subplot2grid(shape=(16,8), loc=(idat,ialg))
            plt.plot(power_spectrum, c='b')
            plt.xticks([])
            plt.yticks([])
            margin = signal_length // 60
            plt.xlim([-margin, signal_length//2 + margin])
            if idat == 0:
                plt.title(plot_alg_names[alg], fontsize=12)
            elif idat == 15:
                plt.xlabel('frequencies')
            elif idat == 12 and ialg >= 5:
                plt.xlabel('frequencies')
            if ialg == 0:
                plt.ylabel(eb.get_dataset_name(env=env, ds=dataset), rotation=0, horizontalalignment='right', verticalalignment='top')
        
    plt.subplot2grid(shape=(16,8), loc=(0,2))
    plt.title('ForeCA', fontsize=12)
    plt.gca().axis('off')

    plt.subplot2grid(shape=(16,8), loc=(0,5))
    plt.title('hSFA', fontsize=12)
    plt.gca().axis('off')

    plt.subplot2grid(shape=(16,8), loc=(0,6))
    plt.title('hPFA', fontsize=12)
    plt.gca().axis('off')

    plt.subplot2grid(shape=(16,8), loc=(0,7))
    plt.title('hGPFA', fontsize=12)
    plt.gca().axis('off')

    plt.subplots_adjust(left=0.1, right=.99, bottom=0.04, top=.96, wspace=.05)
    plt.savefig('fig_spectra.eps')
    plt.show()



if __name__ == '__main__':
    main()
    
