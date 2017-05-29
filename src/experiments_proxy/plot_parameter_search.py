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
            eb.Algorithms.PFA,
            eb.Algorithms.GPFA2
            ]

    p_range = {eb.Algorithms.PFA:   [1,2,4,6,8,10],
               eb.Algorithms.GPFA2: [1,2,4,6]}
    
    line_styles = ['-', '--']
    
    results = {}
    
    for alg in algs:
        results[alg] = {}
        for alg2 in [eb.Algorithms.SFA, alg]:
            results[alg][alg2] = parameters.get_results(alg=alg, overide_args={'algorithm': alg2, 'p': p_range[alg]})#, 'repetitions': 5})
        
    for _, alg in enumerate(algs):  # determines the predictable algorithm and measure
        
        figsize = (10,4.5) if alg is eb.Algorithms.ForeCA else (10,6)
        plt.figure(figsize=figsize)
        plt.suptitle(plot_alg_names[alg])
        
        for i, dataset_args in enumerate(parameters.dataset_args):
                
            env = dataset_args['env']
            dataset = dataset_args['dataset']
            
            plt.subplot(4, 4, i+1)
            plt.title(eb.get_dataset_name(env=env, ds=dataset), fontsize=12)
            print dataset
            
            for j, alg2 in enumerate([alg, eb.Algorithms.SFA]): # plots predictable algorithm or SFA
            
                values = results[alg][alg2][dataset].values
                values = np.mean(values, axis=0)  # output_dim
                print alg2, values.shape, np.mean(values, axis=-1)
                plt.errorbar(x=p_range[alg], y=np.mean(values, axis=-1), yerr=np.std(values, axis=-1), label=plot_alg_names[alg2], ls=line_styles[j])
                #bars[-1][0].set_linestyle(line_styles[j])
                
                #xlim_max = 5.5 if alg is eb.Algorithms.ForeCA else 10.5 
                #plt.xlim(.5, xlim_max)
                #plt.ylim(-.2, np.pi/2+.2)
                #if idx % 4 == 0:
                #    plt.ylabel('angle')
                #else:
                #    plt.gca().set_yticklabels([])
                #if (alg is eb.Algorithms.ForeCA and idx in [5,6,7,8]) or idx >= 12:
                #    plt.xlabel('# dimensions')
                #else:
                #    plt.gca().set_xticklabels([])
                
            plt.xlim(.5, p_range[alg][-1]+.5)
            #plt.locator_params(axis='y', nbins=2)
            plt.gca().set_yticks([plt.gca().get_yticks()[0], plt.gca().get_yticks()[-1]])
            if i < 12:
                plt.gca().set_xticklabels([])
            else:
                plt.xlabel('p')
            if i == 15:
                plt.legend(prop={'size': 9})
                    
        plt.subplots_adjust(hspace=.4, wspace=.4, left=0.04, right=.98, bottom=.08, top=.92)
        plt.savefig('fig_search_p_%s.eps' % plot_alg_names[alg].lower())
        
    plt.show()



if __name__ == '__main__':
    main()
    
