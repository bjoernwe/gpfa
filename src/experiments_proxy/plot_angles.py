import matplotlib.pyplot as plt
import numpy as np

import experiments_proxy.experiment_base as eb
import parameters



def main():

    algs = [eb.Algorithms.Random,
            eb.Algorithms.ForeCA,
            #eb.Algorithms.SFA,
            #eb.Algorithms.SFFA,
            eb.Algorithms.PFA,
            eb.Algorithms.GPFA2
            ]

    results_angle = {}
    
    for alg in algs:
        results_angle[alg] = parameters.get_results(alg, overide_args={'measure': eb.Measures.angle_to_sfa})
        
    for _, alg in enumerate(algs):
            
        plt.figure()
        plt.suptitle(alg)
            
        for ids, dataset_args in enumerate(parameters.dataset_args):

            dataset = dataset_args['dataset']
            if not dataset in results_angle[alg]:
                continue
            
            # angles
            plt.subplot(4, 4, ids+1)
            #plt.subplot2grid(shape=(n_algs,4), loc=(a,3))
            values = results_angle[alg][dataset].values
            d, _ = values.shape
            plt.errorbar(x=range(1,d+1), y=np.mean(values, axis=1), yerr=np.std(values, axis=1))
            plt.xlim(.5, d+.5)
            plt.ylim(-.1, np.pi/2+.1)
                
            # title
            #dataset_str = '%s<%s>' % (dataset_args['env'], dataset_args['dataset'])
            dataset_str = '%s' % (dataset_args['dataset'])
            plt.title(dataset_str)

        plt.subplots_adjust(hspace=.4, wspace=.3, left=0.02, right=.98, bottom=.02, top=.95)
            
    plt.show()



if __name__ == '__main__':
    main()
    
