from __future__ import print_function

import matplotlib
import matplotlib.pyplot as plt
import mkl
import numpy as np
import scipy.stats

import experiments_proxy.experiment_base as eb
import parameters



def main():
    
    mkl.set_num_threads(1)
    
    plot_alg_names = {eb.Algorithms.Random: 'Random',
                      eb.Algorithms.SFA:    'SFA',
                      eb.Algorithms.SFFA:   "SFA'",
                      eb.Algorithms.ForeCA: 'ForeCA',
                      eb.Algorithms.PFA:    'PFA',
                      eb.Algorithms.GPFA2:  'GPFA',
                      }

    results_x = {}
    results_y = {}
    for alg in [eb.Algorithms.Random,
                eb.Algorithms.SFA,
                eb.Algorithms.SFFA,
                eb.Algorithms.ForeCA,
                eb.Algorithms.PFA,
                eb.Algorithms.GPFA2
                ]:

        print(alg)
        #only_low_dimensional = alg is eb.Algorithms.ForeCA
        results_x[alg] = parameters.get_results(alg, overide_args={})
        results_y[alg] = parameters.get_results(alg, overide_args={'use_test_set': False})

    for alg in results_x.keys():
        
        colors = iter(matplotlib.cm.get_cmap('pink')(np.linspace(0, 1, int(1.25*len(parameters.dataset_args)))))
        markers = iter(['*', 'o', '^', 'v', '<', '>', 'd', 'D', 's'] * 2)
        
        plt.figure(figsize=(10,6))

        for dataset_args in parameters.dataset_args:
            
            env = dataset_args['env']
            dataset = dataset_args['dataset']
            if not dataset in results_x[alg]:
                continue
            result_x = results_x[alg][dataset].values
            result_y = results_y[alg][dataset].values
            
            if True:
                # average over first dim (output_dim)
                result_x = np.mean(result_x, axis=0, keepdims=True) 
                result_y = np.mean(result_y, axis=0, keepdims=True) 
            
            # point cloud
            color = next(colors)
            marker = next(markers)
#             for i in range(result_x.shape[0]):
#                 plt.scatter(result_x[i], result_y[i], c=color, marker=marker, label=None, s=80, alpha=.3, linewidths=0, zorder=1)
    
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
    
            label = '%s' % eb.get_dataset_name(env=env, ds=dataset, latex=False) #%s<%s>' % (dataset_args['env'], dataset_args['dataset'])
            xerr = np.vstack([errors_neg, errors_pos])
            yerr = np.vstack([errors_sfa_neg, errors_sfa_pos])
            plt.errorbar(mu_x, mu_y, xerr=xerr, yerr=yerr, c=color, marker=marker, markersize=9, label=label, zorder=2)
            
        # 
        measure_label = 'forcastability' if alg is eb.Algorithms.ForeCA else 'prediction error'
        measure_limits = [1e0, 1e2] if alg is eb.Algorithms.ForeCA else [1e-4, 1e2]
        plt.plot(measure_limits, measure_limits, '-', c='gray', zorder=3)
        plt.suptitle(plot_alg_names[alg])
        plt.xlabel('%s on test set' % (measure_label))
        plt.ylabel('%s on training set' % (measure_label))
        plt.xscale('log')
        plt.yscale('log')
        handles, labels = plt.gca().get_legend_handles_labels()
        handles = [h[0] for h in handles]
        plt.legend(handles, labels, loc='best', prop={'size': 9}, numpoints=1, borderpad=1, handlelength=0)
        plt.tight_layout()
        plt.savefig('fig_generalization_%s.eps' % plot_alg_names[alg].lower())

    plt.show()



if __name__ == '__main__':
    main()
    