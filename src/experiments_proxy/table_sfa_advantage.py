from __future__ import print_function

import numpy as np

import experiments_proxy.experiment_base as eb
import parameters



def main():

    results = {}
    results_sfa = {}
    for alg in [eb.Algorithms.ForeCA,
                eb.Algorithms.PFA,
                eb.Algorithms.GPFA2
                ]:

        print(alg)
        only_low_dimensional = alg is eb.Algorithms.ForeCA
        results[alg] = parameters.get_results(alg, only_low_dimensional=only_low_dimensional)
        results_sfa[alg] = parameters.get_results(alg, overide_args={'algorithm': eb.Algorithms.SFA}, only_low_dimensional=only_low_dimensional)

    print("""
\\begin{center}
\\begin{tabular}{|c|c|c|c|}
\\hline 
- & ForeCA & PFA & GPFA \\\\
\\hline""")
    for dataset_args in parameters.dataset_args:
        env = dataset_args['env']
        dataset = dataset_args['dataset']
        print('\\texttt{%s}' % eb.get_dataset_name(env=env, ds=dataset, latex=True), end='') 
        for alg in results.keys():
            print(' & ', end='')
            if not dataset in results[alg]:
                print(' n/a ', end='')
            else:
                x = np.mean(results[alg][dataset].values, axis=0) # axis 0 = output_dim
                y = np.mean(results_sfa[alg][dataset].values, axis=0)
                if alg is eb.Algorithms.ForeCA:
                    sfa_advantage = np.mean(y) > np.mean(x)
                    sfa_advantage_soft = np.mean(y) + 1*np.std(y) > np.mean(x)
                else:
                    sfa_advantage = np.mean(x) > np.mean(y)
                    sfa_advantage_soft = np.mean(x) + 1*np.std(x) > np.mean(y)
                if sfa_advantage:
                    print('**', end='')
                elif sfa_advantage_soft:
                    print('*', end='')
                else:
                    print('', end='')
        print(' \\\\\n\\hline')
    print('\\end{tabular}\n\\end{center}')



if __name__ == '__main__':
    main()
    