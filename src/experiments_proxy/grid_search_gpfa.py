

import matplotlib.pyplot as plt
import numpy as np
import mkl
#import sys

import experiments_proxy.experiment_base as eb

import parameters
    


def main():
    
    mkl.set_num_threads(1)
    
    search_args = {'p': [1,2,4,6],
                   'k': [1,2,5,10]}

    # run cross-validation        
    results = parameters.get_results(alg=eb.Algorithms.GPFA2, overide_args=search_args)

    # dict to store results in
    result_dict = {}

    for dataset, result in results.items():

        result_averaged = np.mean(result.values, axis=(0, -1)) # average out 1st axis = output_dim and last axis = repetitions

        if len(result.iter_args) == 3:
            # grid search
            params = result.iter_args.items()[1:] # iter_args w/o output_dim
            idc_min = np.unravel_index(np.argmin(result_averaged), result_averaged.shape) # convert to 2D index
            # assemble result dict
            result_dict[dataset] = {}
            for i, idx in enumerate(idc_min):
                result_dict[dataset][params[i][0]] = params[i][1][idx]
        elif len(result.iter_args) == 2:
            # plot results for the one changing parameter
            iter_arg_values = result.iter_args.values()[1]
            plt.figure()
            plt.plot(iter_arg_values, result_averaged)
            plt.xlabel(result.iter_args.keys()[1])
            plt.title(dataset)
        else:
            assert False
    
    # print results as code ready to copy & paste
    print '{'
    for dataset_dict in parameters.dataset_args:
        dataset = dataset_dict['dataset']
        print '                       env_data.%s: %s,' % (dataset.__str__(), result_dict.get(dataset, None))
    print '}'
    
    plt.show()


if __name__ == '__main__':
    main()
    
    