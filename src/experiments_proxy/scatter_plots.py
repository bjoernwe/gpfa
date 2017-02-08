from __future__ import print_function

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

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
        #only_low_dimensional = alg is eb.Algorithms.ForeCA
        results[alg] = parameters.get_results(alg)
        results_sfa[alg] = parameters.get_results(alg, overide_args={'algorithm': eb.Algorithms.SFA})
    #results_sfa = parameters.get_results(eb.Algorithms.SFA)

    for alg in results.keys():
        
        colors = iter(matplotlib.cm.get_cmap('Set1')(np.linspace(0, 1, len(parameters.dataset_args))))
        markers = iter(['*', 'o', '^', 'v', '<', '>', 'd', 's'] * 2)
        
        plt.figure()

        #print alg
        #only_low_dimensional = alg is eb.Algorithms.ForeCA
        #results = parameters.get_results(alg, only_low_dimensional=only_low_dimensional)
        #print eb.Algorithms.SFA
        #results_sfa = parameters.get_results(alg, overide_args={'algorithm': eb.Algorithms.SFA}, only_low_dimensional=only_low_dimensional)
        
        for dataset_args in parameters.dataset_args:
            
            dataset = dataset_args['dataset']
            if not dataset in results[alg]:
                continue
            result = results[alg][dataset].values
            result_sfa = results_sfa[alg][dataset].values
            
            if True:
                # average over first dim (output_dim)
                result = np.mean(result, axis=0, keepdims=True) 
                result_sfa = np.mean(result_sfa, axis=0, keepdims=True) 
            
            # point cloud
            color = next(colors)
            marker = next(markers)
            #for i in range(result.shape[0]):
            #    plt.scatter(result[i], result_sfa[i], c=color, marker=marker, label=None, s=80, alpha=.2, linewidths=0, zorder=1)
    
            # plot
            mu = np.mean(result, axis=-1) # last axis = repetitions
            values0 = (result.T - mu).T
            values0_dummy = np.array(values0, copy=True)
            values0_dummy[values0 < 0] = np.NaN
            errors_pos = np.sqrt(np.nanmean(values0_dummy**2, axis=-1))
            values0_dummy = np.array(values0, copy=True)
            values0_dummy[values0 > 0] = np.NaN
            errors_neg = np.sqrt(np.nanmean(values0_dummy**2, axis=-1))
    
            mu_sfa = np.mean(result_sfa, axis=-1) # 1st axis = output_dim, last axis = repetitions
            values0_sfa = (result_sfa.T - mu_sfa).T
            values0_sfa_dummy = np.array(values0_sfa, copy=True)
            values0_sfa_dummy[values0_sfa < 0] = np.NaN
            errors_sfa_pos = np.sqrt(np.nanmean(values0_sfa_dummy**2, axis=-1))
            values0_sfa_dummy = np.array(values0_sfa, copy=True)
            values0_sfa_dummy[values0_sfa > 0] = np.NaN
            errors_sfa_neg = np.sqrt(np.nanmean(values0_sfa_dummy**2, axis=-1))
    
            label = '%s<%s>' % (dataset_args['env'], dataset_args['dataset'])
            xerr = np.vstack([errors_neg, errors_pos])
            yerr = np.vstack([errors_sfa_neg, errors_sfa_pos])
            plt.errorbar(mu, mu_sfa, xerr=xerr, yerr=yerr, c=color, marker=marker, markersize=10, label=label, zorder=2)
            
            # Wilcoxon signed-rank test
            x = np.mean(result, axis=0) # axis 0 = output_dim
            y = np.mean(result_sfa, axis=0)
            _, pvalue = scipy.stats.wilcoxon(x, y)
            if np.mean(x) > np.mean(y):
                p = pvalue / 2.
            else:
                p = 1 - pvalue / 2.
            print('%s\np-value for X > Y: %0.4f (%0.4f)' % (dataset, p, 1-p))
            #print 'p-value for X < Y:', 1 - pvalue / 2.

        # 
        plt.plot([1e-6, 1e2], [1e-6, 1e2], '-', zorder=3)
        plt.xlabel('error of %s' % alg)
        plt.ylabel('error of SFA')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='best', prop={'size': 8})

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
        
    plt.show()



if __name__ == '__main__':
    main()
    