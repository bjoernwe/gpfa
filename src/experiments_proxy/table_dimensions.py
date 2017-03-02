from __future__ import print_function

import numpy as np

import experiments_proxy.experiment_base as eb
import parameters



def main():

    results = parameters.get_results(alg=eb.Algorithms.SFA, overide_args={'measure': eb.Measures.ndims})

    f = open('/home/weghebvc/Documents/2016-09 - NC2/paper/table_dims.tex', 'w+')
    print("""
\\begin{center}
\\begin{tabular}{|l|r|r|r|c|c|}
\\hline 
Dataset & $S$ & $S_{train}$ & $S_{test}$ & $N$ & $N'$ \\\\
\\hline""", file=f)
    for dataset_args in parameters.dataset_args:

        env = dataset_args['env']
        dataset = dataset_args['dataset']
        
        kwargs = dict(parameters.default_args_global)
        #kwargs['algorithm'] = alg
        #kwargs['measure'] = algorithm_measures[alg]
        kwargs.update(dataset_args)
        kwargs.update(parameters.dataset_default_args.get(dataset, {}))
        #kwargs.update(algorithm_parameters.get(alg, {}).get(dataset, {}))
        #kwargs.update(algorithm_args.get(alg, {}))
        #kwargs.update(overide_args)
        
        print('\\texttt{%s}' % eb.get_dataset_name(env=env, ds=dataset, latex=True), end='', file=f)
        print(' & ', end='', file=f)
        e = env(dataset, limit_data=parameters.default_args_global['limit_data'])
        N, M = e.data.shape
        print('%d' % N, end='', file=f)
        print(' & ', end='', file=f)
        print('%d' % kwargs['n_train'], end='', file=f)
        print(' & ', end='', file=f)
        print('%d' % kwargs['n_test'], end='', file=f)
        print(' & ', end='', file=f)
        print('%d' % M, end='', file=f)
        print(' & ', end='', file=f)
        dim_avg = np.mean(results[dataset].values) # axis 0 = output_dim
        dim_std = np.std(results[dataset].values) # axis 0 = output_dim
        if kwargs['pca'] < 1.0:
            print('%0.1f $\pm$ %0.1f' % (dim_avg, dim_std), end='', file=f)
        else:
            print('%d' % dim_avg, end='', file=f)
        print(' \\\\\n\\hline', file=f)
    print('\\end{tabular}\n\\end{center}', file=f)



if __name__ == '__main__':
    main()
    