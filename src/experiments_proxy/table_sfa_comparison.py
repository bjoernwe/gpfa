from __future__ import print_function

import mkl
import numpy as np
import scipy.stats

import experiments_proxy.experiment_base as eb
import parameters
import parameters_hi



def main():

    mkl.set_num_threads(1)

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
                sfa_advantage = np.mean(x) > np.mean(y)
                sffa_advantage = np.mean(x) > np.mean(z)
                if alg is eb.Algorithms.ForeCA:
                    sfa_advantage = not sfa_advantage
                    sffa_advantage = not sffa_advantage  
                soft_advantage = np.abs(np.mean(x) - np.mean(y)) < np.std(y)
                if soft_advantage:
                    if pvalue/2. < .01:
                        if sfa_advantage:
                            print('$+$', end='', file=f)
                        else:
                            print('$-$', end='', file=f)
                    else:
                        print('$\circ$', end='', file=f)
                else:
                    if sfa_advantage:
                        print('$++$', end='', file=f)
                    else:
                        print('$--$', end='', file=f)
        # hierarchical
        for alg in results_hi.keys():
            print(' & ', end='', file=f)
            if not dataset in results[alg]:
                print(' ', end='', file=f)
            else:
                x = np.mean(results_hi[alg][dataset].values, axis=0) # axis 0 = output_dim
                y = np.mean(results_hisfa[alg][dataset].values, axis=0)
                #z = np.mean(results_hi_sffa[alg][dataset].values, axis=0)
                _, pvalue = scipy.stats.wilcoxon(x, y)
                #_, pvalue_sffa = scipy.stats.wilcoxon(x, z)
                sfa_advantage = np.mean(x) > np.mean(y)
                sffa_advantage = np.mean(x) > np.mean(z)
                soft_advantage = np.abs(np.mean(x) - np.mean(y)) - np.std(y)
                if soft_advantage:
                    if pvalue/2. < .01:
                        if sfa_advantage:
                            print('+', end='', file=f)
                        else:
                            print('-', end='', file=f)
                    else:
                        print('$\circ$', end='', file=f)
                else:
                    if sfa_advantage:
                        print('++', end='', file=f)
                    else:
                        print('--', end='', file=f)
        print(' \\\\\n', file=f)
    print('\\bottomrule\n', file=f)
    print('\\end{tabular}\n\\end{center}', file=f)



if __name__ == '__main__':
    main()
    