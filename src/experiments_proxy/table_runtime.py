from __future__ import print_function

import mkl
import numpy as np
import scipy.stats

import experiments_proxy.experiment_base as eb
import parameters
import parameters_hi



def main():

    mkl.set_num_threads(1)
    
    algs = [eb.Algorithms.SFA,
            eb.Algorithms.ForeCA,
            eb.Algorithms.PFA,
            eb.Algorithms.GPFA2
            ]
    
    results = {}
    for alg in algs:
        print(alg)
        results[alg] = parameters.get_results(alg, overide_args={'measure': eb.Measures.ndims, 'output_dim': 5, 'output_dim_max': 5,})

    f = open('table_runtime.tex', 'w+')
    print("""
\\begin{center}
\\begin{tabular}{L{5cm} C{1.6cm} C{1.6cm} C{1.6cm} C{1.6cm}}
\\toprule 
Dataset & SFA & ForeCA & PFA & GPFA \\\\
\\midrule""", file=f)
    for dataset_args in parameters.dataset_args:
        env = dataset_args['env']
        dataset = dataset_args['dataset']
        print('\\texttt{%s}' % eb.get_dataset_name(env=env, ds=dataset, latex=True), end='', file=f)
        # linear
        for alg in algs:
            print(' & ', end='', file=f)
            if not dataset in results[alg]:
                print(' ', end='', file=f)
            else:
                #
                time = np.mean(results[alg][dataset].elapsed_times) # axis 0 = output_dim
                #time = np.mean(time, axis=-1)
                print('\\texttt{%E}' % (time/1000.), end='', file=f)
        print(' \\\\\n', file=f)
    print('\\bottomrule\n', file=f)
    print('\\end{tabular}\n\\end{center}', file=f)
    
    

if __name__ == '__main__':
    main()
    
