import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import experiments_proxy.experiment_base as eb
import parameters



def main():

    for alg in [eb.Algorithms.ForeCA,
                eb.Algorithms.SFA,
                eb.Algorithms.SFFA,
                eb.Algorithms.PFA,
                eb.Algorithms.GPFA2
                ]:
        
        plt.figure()
        colors = iter(matplotlib.cm.get_cmap('Set1')(np.linspace(0, 1, len(parameters.dataset_args))))
        markers = iter(['*', 'o', '^', 'v', '<', '>', 'd', 's'] * 2)
        
        print alg
        only_low_dimensional = alg is eb.Algorithms.ForeCA
        results_x = parameters.get_results(alg, only_low_dimensional=only_low_dimensional)
        results_y = parameters.get_results(alg, overide_args={'use_test_set': False}, only_low_dimensional=only_low_dimensional)
        
        for dataset_args in parameters.dataset_args:
            
            dataset = dataset_args['dataset']
            if not dataset in results_x:
                continue
            result_x = results_x[dataset].values
            result_y = results_y[dataset].values
            
            if False:
                # average over first dim (output_dim)
                result_x = np.mean(result_x, axis=0, keepdims=True) 
                result_y = np.mean(result_y, axis=0, keepdims=True) 
            
            # point cloud
            color = next(colors)
            marker = next(markers)
            for i in range(result_x.shape[0]):
                plt.scatter(result_x[i], result_y[i], c=color, marker=marker, label=None, s=80, alpha=.3, linewidths=0, zorder=1)
    
            # plot
            mu_x = np.mean(result_x, axis=-1) # last axis = repetitions
            values0 = (result_x.T - mu_x).T
            values0_dummy = np.array(values0, copy=True)
            values0_dummy[values0 < 0] = np.NaN
            errors_pos = np.sqrt(np.nanmean(values0_dummy**2, axis=-1))
            values0_dummy = np.array(values0, copy=True)
            values0_dummy[values0 > 0] = np.NaN
            errors_neg = np.sqrt(np.nanmean(values0_dummy**2, axis=-1))
    
            mu_y = np.mean(result_y, axis=-1) # 1st axis = output_dim, last axis = repetitions
            values0_sfa = (result_y.T - mu_y).T
            values0_sfa_dummy = np.array(values0_sfa, copy=True)
            values0_sfa_dummy[values0_sfa < 0] = np.NaN
            errors_sfa_pos = np.sqrt(np.nanmean(values0_sfa_dummy**2, axis=-1))
            values0_sfa_dummy = np.array(values0_sfa, copy=True)
            values0_sfa_dummy[values0_sfa > 0] = np.NaN
            errors_sfa_neg = np.sqrt(np.nanmean(values0_sfa_dummy**2, axis=-1))
    
            label = '%s<%s>' % (dataset_args['env'], dataset_args['dataset'])
            xerr = np.vstack([errors_neg, errors_pos])
            yerr = np.vstack([errors_sfa_neg, errors_sfa_pos])
            plt.errorbar(mu_x, mu_y, xerr=xerr, yerr=yerr, c=color, marker=marker, markersize=7, label=label, zorder=2)
            
        # 
        plt.plot([1e-6, 1e2], [1e-6, 1e2], '-', zorder=3)
        plt.xlabel('error of %s on test' % alg)
        plt.ylabel('error of %s on train' % alg)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='best', prop={'size': 8})
        
    plt.show()



if __name__ == '__main__':
    main()
    