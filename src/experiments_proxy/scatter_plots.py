import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

import experiments_proxy.experiment_base as eb
import parameters



def main():

    for alg in [eb.Algorithms.ForeCA,
                eb.Algorithms.SFFA,
                eb.Algorithms.PFA,
                eb.Algorithms.GPFA2
                ]:
        
        plt.figure()
        colors = iter(matplotlib.cm.get_cmap('Set1')(np.linspace(0, 1, len(parameters.dataset_args))))
        markers = iter(['*', 'o', '^', 'v', '<', '>', 'd', 's'] * 2)
        
        print alg
        only_low_dimensional = alg is eb.Algorithms.ForeCA
        results = parameters.get_results(alg, only_low_dimensional=only_low_dimensional)
        print eb.Algorithms.SFA
        results_sfa = parameters.get_results(alg, overide_args={'algorithm': eb.Algorithms.SFA}, only_low_dimensional=only_low_dimensional)
        
        for dataset_args in parameters.dataset_args:
            
            dataset = dataset_args['dataset']
            if not dataset in results:
                continue
            result = results[dataset]
            result_sfa = results_sfa[dataset]
            
            # point cloud
            color = next(colors)
            marker = next(markers)
            for i in range(result.values.shape[0]):
                plt.scatter(result.values[i], result_sfa.values[i], c=color, marker=marker, label=None, s=80, alpha=.3, linewidths=0, zorder=1)
    
            # plot
            mu = np.mean(result.values, axis=-1) # last axis = repetitions
            values0 = (result.values.T - mu).T
            values0_dummy = np.array(values0, copy=True)
            values0_dummy[values0 < 0] = np.NaN
            errors_pos = np.sqrt(np.nanmean(values0_dummy**2, axis=-1))
            values0_dummy = np.array(values0, copy=True)
            values0_dummy[values0 > 0] = np.NaN
            errors_neg = np.sqrt(np.nanmean(values0_dummy**2, axis=-1))
    
            mu_sfa = np.mean(result_sfa.values, axis=-1) # 1st axis = output_dim, last axis = repetitions
            values0_sfa = (result_sfa.values.T - mu_sfa).T
            values0_sfa_dummy = np.array(values0_sfa, copy=True)
            values0_sfa_dummy[values0_sfa < 0] = np.NaN
            errors_sfa_pos = np.sqrt(np.nanmean(values0_sfa_dummy**2, axis=-1))
            values0_sfa_dummy = np.array(values0_sfa, copy=True)
            values0_sfa_dummy[values0_sfa > 0] = np.NaN
            errors_sfa_neg = np.sqrt(np.nanmean(values0_sfa_dummy**2, axis=-1))
    
            label = '%s<%s>' % (dataset_args['env'], dataset_args['dataset'])
            xerr = np.vstack([errors_neg, errors_pos])
            yerr = np.vstack([errors_sfa_neg, errors_sfa_pos])
            plt.errorbar(mu, mu_sfa, xerr=xerr, yerr=yerr, c=color, marker=marker, markersize=7, label=label, zorder=2)
            
            # Wilcoxon signed-rank test
            x = np.mean(result.values, axis=0) # axis 0 = output_dim
            y = np.mean(result_sfa.values, axis=0)
            _, pvalue = scipy.stats.wilcoxon(x, y)
            print 'p-value for X > Y:', pvalue / 2.
            print 'p-value for X < Y:', 1 - pvalue / 2.

        # 
        plt.plot([1e-6, 1e1], [1e-6, 1e1], '-', zorder=3)
        plt.xlabel('error of %s' % alg)
        plt.ylabel('error of SFA')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='best', prop={'size': 8})
        
    plt.show()



if __name__ == '__main__':
    main()
    