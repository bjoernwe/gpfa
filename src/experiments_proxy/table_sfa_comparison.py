from __future__ import print_function

import mkl
import numpy as np
import scipy.stats

import experiments_proxy.experiment_base as eb
import parameters
import parameters_hi



def main():

    mkl.set_num_threads(1)
    
    use_test_set = True

    algs = [eb.Algorithms.ForeCA,
            eb.Algorithms.PFA,
            eb.Algorithms.GPFA2
            ]
    
    algs_hi = [eb.Algorithms.HiPFA,
               eb.Algorithms.HiGPFA,
               ]

    results = {}
    results_sfa = {}
    #results_sffa = {}
    for alg in algs:

        print(alg)
        results[alg] = parameters.get_results(alg, overide_args={'use_test_set': use_test_set})
        results_sfa[alg] = parameters.get_results(alg, overide_args={'algorithm': eb.Algorithms.SFA, 'use_test_set': use_test_set})
        #results_sffa[alg] = parameters.get_results(alg, overide_args={'algorithm': eb.Algorithms.SFFA, 'use_test_set': use_test_set})

    results_hi = {}
    results_hisfa = {}
    #results_hisffa = {}
    for alg in algs_hi:

        print(alg)
        results_hi[alg] = parameters_hi.get_results(alg, overide_args={'use_test_set': use_test_set})
        results_hisfa[alg] = parameters_hi.get_results(alg, overide_args={'algorithm': eb.Algorithms.HiSFA, 'use_test_set': use_test_set})
        #results_hisffa[alg] = parameters_hi.get_results(alg, overide_args={'algorithm': eb.Algorithms.HiSFFA, 'use_test_set': use_test_set})

    symbol_stats = {}

    #f = open('/home/weghebvc/Documents/2016-09 - NC2/paper/table_sfa_comparison.tex', 'w+')
    f = open('table_sfa_comparison%s.tex' % ('' if use_test_set else '_training'), 'w+')
    print("""
\\begin{center}
\\begin{tabular}{L{3.5cm} C{1.3cm} C{1.3cm} C{1.3cm} C{1.3cm} C{1.3cm}}
\\toprule 
Dataset & ForeCA & PFA & GPFA & hPFA & hGPFA \\\\
\\midrule""", file=f)
    for dataset_args in parameters.dataset_args:
        env = dataset_args['env']
        dataset = dataset_args['dataset']
        print('\\texttt{%s}' % eb.get_dataset_name(env=env, ds=dataset, latex=True), end='', file=f)
        # linear
        for alg in results.keys():
            print(' & ', end='', file=f)
            if not dataset in results[alg]:
                print(' ', end='', file=f)
            else:
                #
                samples_alg  = np.mean(results[alg][dataset].values, axis=0) # axis 0 = output_dim
                samples_sfa  = np.mean(results_sfa[alg][dataset].values, axis=0)
                #samples_sffa = np.mean(results_sffa[alg][dataset].values, axis=0)
                symbol = evaluate(samples0=samples_sfa, samples1=samples_alg, inverse=alg is eb.Algorithms.ForeCA)
                symbol_stats[symbol] = symbol_stats.get(symbol, 0) + 1
                #symbol_sffa = evaluate(samples0=samples_sffa, samples1=samples_alg, inverse=alg is eb.Algorithms.ForeCA)
                print(symbol, end='', file=f)
                #if symbol_sffa != symbol:
                #    print('/'+symbol_sffa, end='', file=f)
        # hierarchical
        for alg in results_hi.keys():
            print(' & ', end='', file=f)
            if not dataset in results_hi[alg]:
                print(' ', end='', file=f)
            else:
                samples_hialg  = np.mean(results_hi[alg][dataset].values, axis=0) # axis 0 = output_dim
                samples_hisfa  = np.mean(results_hisfa[alg][dataset].values, axis=0)
                #samples_hisffa = np.mean(results_hisffa[alg][dataset].values, axis=0)
                symbol_hi = evaluate(samples0=samples_hisfa, samples1=samples_hialg)
                symbol_stats[symbol_hi] = symbol_stats.get(symbol_hi, 0) + 1
                #symbol_hisffa = evaluate(samples0=samples_hisffa, samples1=samples_hialg)
                print(symbol_hi, end='', file=f)
                #if symbol_hisffa != symbol_hi:
                #    print('/'+symbol_hisffa, end='', file=f)
        print(' \\\\\n', file=f)
    print('\\bottomrule\n', file=f)
    print('\\end{tabular}\n\\end{center}', file=f)
    
    for symbol, counts in symbol_stats.items():
        print('%s : %d' % (symbol, counts))
    
    
def evaluate(samples0, samples1, inverse=False):
    
    _, pvalue = scipy.stats.wilcoxon(samples0, samples1)
    significance = pvalue/2. < .01
    advantage0 = np.mean(samples0) < np.mean(samples1)
    if inverse:
        advantage0 = not advantage0
    large_difference = np.abs(np.mean(samples1) - np.mean(samples0)) > np.std(samples0)

    if large_difference:
        if advantage0:
            return '$++$'
        else:
            return '$--$'
    else:
        if significance:
            if advantage0:
                return '$+$'
            else:
                return '$-$'
        else:
            return '$\circ$'



if __name__ == '__main__':
    main()
    
