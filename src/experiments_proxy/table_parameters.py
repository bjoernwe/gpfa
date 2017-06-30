from __future__ import print_function

import mkl
import numpy as np
import scipy.stats

import experiments_proxy.experiment_base as eb
import parameters
import parameters_hi



def main():

    #f = open('/home/weghebvc/Documents/2016-09 - NC2/paper/table_sfa_comparison.tex', 'w+')
    f = open('table_parameters.tex', 'w+')
    print("""
\\begin{center}
\\begin{tabular}{L{5cm} C{1cm} C{1cm} C{1cm} C{1cm}}
\\toprule 
 & \multicolumn{2}{c}{PFA} & \multicolumn{2}{c}{GPFA} \\\\
Dataset & $p$ & $K$ & $p$ & $k$ \\\\
\\midrule""", file=f)
    for dataset_args in parameters.dataset_args:
        env = dataset_args['env']
        dataset = dataset_args['dataset']
        args_pfa = parameters.algorithm_parameters[eb.Algorithms.PFA].get(dataset, {})
        args_gpfa = parameters.algorithm_parameters[eb.Algorithms.GPFA2].get(dataset, {})
        print('\\texttt{%s} & ' % eb.get_dataset_name(env=env, ds=dataset, latex=True), end='', file=f)
        print('%s & %s & %s & %s' % (args_pfa.get('p', dataset_args.get('p', None)), 
                                     args_pfa.get('K', dataset_args.get('K', None)), 
                                     args_gpfa.get('p', dataset_args.get('p', None)), 
                                     args_gpfa.get('k', dataset_args.get('k', None))), end='', file=f)
        print(' \\\\\n', file=f)
    print('\\bottomrule\n', file=f)
    print('\\end{tabular}\n\\end{center}', file=f)
   
    

if __name__ == '__main__':
    main()
    
