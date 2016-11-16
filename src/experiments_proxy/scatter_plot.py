import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import explot as ep

import experiments_proxy.experiment_base as eb



def scatter_plot(default_args_global, default_args_low, default_args_high, datasets_low, datasets_high, parameters_low, parameters_high):
    
    colors = iter(matplotlib.cm.get_cmap('Set1')(np.linspace(0, 1, len(datasets_low) + len(datasets_high))))
    markers = iter(['*', 'o', '^', 'v', '<', '>', 'd', 's'] * 2)
    for default_args, datasets, parameters in zip([default_args_low, default_args_high], 
                                                  [datasets_low, datasets_high], 
                                                  [parameters_low, parameters_high]):
        
        for _, dataset_args in enumerate(datasets):
            
            print dataset_args['dataset']
            
            # PFA/GPFA signals
            kwargs = dict(default_args_global)
            kwargs.update(default_args)
            kwargs.update(dataset_args)
            dataset = dataset_args['dataset']
            if dataset in parameters:
                kwargs.update(parameters[dataset])
            result = ep.evaluate(eb.prediction_error, argument_order=['output_dim'], ignore_arguments=['window'], **kwargs)
            
            # SFA signals for comparison
            kwargs.update({'algorithm': eb.Algorithms.SFA})
            result_sfa = ep.evaluate(eb.prediction_error, argument_order=['output_dim'], ignore_arguments=['window'], **kwargs)
    
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
            
    return


