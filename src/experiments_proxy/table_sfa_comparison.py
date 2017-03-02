from __future__ import print_function

import numpy as np

import experiments_proxy.experiment_base as eb
import parameters



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

    f = open('/home/weghebvc/Documents/2016-09 - NC2/paper/table_sfa_comparison.tex', 'w+')
    print("""
\\begin{center}
\\begin{tabular}{|c|c|c|c|}
\\hline 
Dataset & ForeCA & PFA & GPFA \\\\
\\hline""", file=f)
    for dataset_args in parameters.dataset_args:
        env = dataset_args['env']
        dataset = dataset_args['dataset']
        print('\\texttt{%s}' % eb.get_dataset_name(env=env, ds=dataset, latex=True), end='', file=f)
        for alg in results.keys():
            print(' & ', end='', file=f)
            if not dataset in results[alg]:
                print(' n/a ', end='', file=f)
            else:
                x = np.mean(results[alg][dataset].values, axis=0) # axis 0 = output_dim
                y = np.mean(results_sfa[alg][dataset].values, axis=0)
                z = np.mean(results_sffa[alg][dataset].values, axis=0)
                if alg is eb.Algorithms.ForeCA:
                    sfa_advantage = np.mean(y) > np.mean(x)
                    sffa_advantage = np.mean(z) > np.mean(x)
                    sfa_advantage_soft = np.mean(y) + 1*np.std(y) > np.mean(x)
                else:
                    sfa_advantage = np.mean(x) > np.mean(y)
                    sffa_advantage = np.mean(x) > np.mean(z)
                    sfa_advantage_soft = np.mean(x) + 1*np.std(x) > np.mean(y)
                if sfa_advantage:
                    print('+', end='', file=f)
                elif sfa_advantage_soft:
                    print('-', end='', file=f)
                    if sffa_advantage:
                        print('/+', end='', file=f)
                else:
                    print('', end='', file=f)
        print(' \\\\\n\\hline', file=f)
    print('\\end{tabular}\n\\end{center}', file=f)



if __name__ == '__main__':
    main()
    