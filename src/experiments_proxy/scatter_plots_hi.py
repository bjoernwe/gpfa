from __future__ import print_function

import matplotlib
import matplotlib.pyplot as plt
import math
import mkl
import numpy as np

import experiments_proxy.experiment_base as eb
import parameters_hi



def main():

    mkl.set_num_threads(1)
    
    plot_alg_names = {eb.Algorithms.HiSFA:  'HiSFA',
                      eb.Algorithms.HiPFA:  'HiPFA',
                      eb.Algorithms.HiGPFA: 'HiGPFA',
                      }

    results = {}
    results_sfa = {}
    for alg in [eb.Algorithms.HiPFA,
                eb.Algorithms.HiGPFA
                ]:

        print(alg)
        results[alg] = parameters_hi.get_results(alg, overide_args={})
        results_sfa[alg] = parameters_hi.get_results(alg, overide_args={'algorithm': eb.Algorithms.HiSFA})
    #results_sfa = parameters.get_results(eb.Algorithms.SFA)

    for alg in results.keys():
        
        colors = iter(matplotlib.cm.get_cmap('pink')(np.linspace(0, 1, math.ceil(1.25*len(parameters_hi.dataset_args_hi)))))
        markers = iter(['*', 'o', '^', 'v', '<', '>', 'd', 'D', 's'] * 2)
        
        plt.figure(figsize=(10,6))

        for dataset_args in parameters_hi.dataset_args_hi:
            
            env = dataset_args['env']
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
    
            label = '%s' % eb.get_dataset_name(env=env, ds=dataset, latex=False) #%s<%s>' % (dataset_args['env'], dataset_args['dataset'])
            xerr = np.vstack([errors_neg, errors_pos])
            yerr = np.vstack([errors_sfa_neg, errors_sfa_pos])
            plt.errorbar(mu, mu_sfa, xerr=xerr, yerr=yerr, c=color, marker=marker, markersize=10, label=label, zorder=2)
            
        # 
        plt.plot([1e-2, 1e1], [1e-2, 1e1], '-', c='gray', zorder=3)
        plt.xlabel('prediction error on %s features' % plot_alg_names[alg])
        plt.ylabel('prediction error on HiSFA features')
        plt.xscale('log')
        plt.yscale('log')
        handles, labels = plt.gca().get_legend_handles_labels()
        handles = [h[0] for h in handles]
        plt.legend(handles, labels, loc='best', prop={'size': 9}, numpoints=1, borderpad=1, handlelength=0)
        plt.tight_layout()
        plt.savefig('fig_results_%s.eps' % plot_alg_names[alg].lower())

    plt.show()



if __name__ == '__main__':
    main()
    