from __future__ import print_function

import numpy as np
import scipy.stats

import experiments_proxy.experiment_base as eb
import parameters
import parameters_hi



def main():

    results = {}
    results_sfa = {}
    results_sffa = {}
    for alg in [eb.Algorithms.ForeCA,
                eb.Algorithms.PFA,
                eb.Algorithms.GPFA2
                ]:

        print(alg)
        results[alg] = parameters.get_results(alg)
        results_sfa[alg] = parameters.get_results(alg, overide_args={'algorithm': eb.Algorithms.SFA})
        results_sffa[alg] = parameters.get_results(alg, overide_args={'algorithm': eb.Algorithms.SFFA})

    results_hi = {}
    results_hisfa = {}
    for alg in [eb.Algorithms.HiPFA,
                eb.Algorithms.HiGPFA,
                ]:

        print(alg)
        results_hi[alg] = parameters_hi.get_results(alg)
        results_hisfa[alg] = parameters_hi.get_results(alg, overide_args={'algorithm': eb.Algorithms.HiSFA})

    f = open('/home/weghebvc/Documents/2016-09 - NC2/paper/table_sfa_comparison.tex', 'w+')
    print("""
\\begin{center}
\\begin{tabular}{L{3.5cm} C{1.3cm} C{1.3cm} C{1.3cm} C{1.3cm} C{1.3cm}}
\\toprule 
Dataset & ForeCA & PFA & GPFA & HiPFA & HiGPFA \\\\
\\midrule""", file=f)
    for dataset_args in parameters.dataset_args:
        env = dataset_args['env']
        dataset = dataset_args['dataset']
        print('\\texttt{%s}' % eb.get_dataset_name(env=env, ds=dataset, latex=True), end='', file=f)
        for alg in results.keys():
            # linear
            print(' & ', end='', file=f)
            if not dataset in results[alg]:
                print(' ', end='', file=f)
            else:
                x = np.mean(results[alg][dataset].values, axis=0) # axis 0 = output_dim
                y = np.mean(results_sfa[alg][dataset].values, axis=0)
                z = np.mean(results_sffa[alg][dataset].values, axis=0)
                _, pvalue = scipy.stats.wilcoxon(x, y)
                _, pvalue_sffa = scipy.stats.wilcoxon(x, z)
                if alg is eb.Algorithms.ForeCA:
                    sfa_advantage = np.mean(y) > np.mean(x)
                    sffa_advantage = np.mean(z) > np.mean(x)
                    sfa_advantage_soft = np.mean(y) + np.std(y) > np.mean(x)
                else:
                    sfa_advantage = np.mean(x) > np.mean(y)
                    sffa_advantage = np.mean(x) > np.mean(z)
                    sfa_advantage_soft = np.mean(x) > np.mean(y) - np.std(y) 
                if sfa_advantage and pvalue/2. < .01:
                    print('+', end='', file=f)
                elif sfa_advantage_soft:
                    print('$\circ$', end='', file=f)
                    if sffa_advantage and pvalue_sffa/2. < .01:
                        print('/+', end='', file=f)
                else:
                    print('-', end='', file=f)
            # hierarchical
        for alg in results_hi.keys():
            print(' & ', end='', file=f)
            if not dataset in results_hi[alg]:
                print(' ', end='', file=f)
            else:
                x = results_hi[alg][dataset].values
                y = results_hisfa[alg][dataset].values
                _, pvalue = scipy.stats.wilcoxon(x, y)
                hisfa_advantage = np.mean(x) > np.mean(y)
                hisfa_advantage_soft = np.mean(x) > np.mean(y) - np.std(y) 
                if hisfa_advantage and pvalue/2. < .01:
                    print('+', end='', file=f)
                elif hisfa_advantage_soft:
                    print('$\circ$', end='', file=f)
                else:
                    print('-', end='', file=f)
        print(' \\\\\n', file=f)
    print('\\bottomrule\n', file=f)
    print('\\end{tabular}\n\\end{center}', file=f)



if __name__ == '__main__':
    main()
    