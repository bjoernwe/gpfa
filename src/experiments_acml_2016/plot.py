import numpy as np
import sys

from matplotlib import pyplot as plt
from matplotlib.colors import ColorConverter

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

import experiments.experiment_base as eb


def _get_values(result, plot_time=False):
    if plot_time:
        return result.elapsed_times / 1000.
    return result.values


def plot_experiment(dataset, N, k, k_eval, p, K, noisy_dims, keep_variance, iterations, 
                    output_dim, repetitions, include_random, include_sfa, include_sffa, 
                    include_foreca, include_gfa1, include_gfa2, N2, cachedir, processes,
                    causal_features=True, generalized_eigen_problem=True, use_test_set=True, x_offset=0, 
                    y_label=True, legend=True, plot_time=False, whitening=True, 
                    manage_seed='external', legend_loc='best', seed=0):
    
    results = {}
    
    if plot_time:
        eb.set_cachedir(cachedir=None)
    else:
        eb.set_cachedir(cachedir=cachedir)
    
    legends = []
    cc = ColorConverter()
    
    use_test_set = use_test_set
    ecolor_alpha = .5
    
    result = ep.evaluate(eb.prediction_error,
                         algorithm=eb.Algorithms.Random, 
                         N=N, 
                         k=k,
                         k_eval=k_eval, 
                         p=p, 
                         K=K, 
                         whitening=whitening,
                         seed=seed,
                         num_steps_test=N2,
                         iterations=iterations,
                         noisy_dims=noisy_dims,
                         keep_variance=keep_variance, 
                         neighborhood_graph=False,
                         weighted_edges=True,
                         #generalized_eigen_problem=generalized_eigen_problem, 
                         output_dim=output_dim,
                         use_test_set=use_test_set, 
                         dataset=dataset, 
                         measure=eb.Measures.gpfa, 
                         repetitions=repetitions, 
                         processes=processes,
                         manage_seed=manage_seed, 
                         cachedir=cachedir)
    results[eb.Algorithms.Random] = result
 
    # determine iter_arg
    iter_arg = result.iter_args.keys()[0]
    
    if include_random:
        values = _get_values(result, plot_time=plot_time)
        m = np.mean(values, axis=-1)
        s = np.std(values, axis=-1)
        x = np.array(result.iter_args[iter_arg]) - 2 * x_offset
        x = x+2 if dataset == eb.Datasets.Kai and type(noisy_dims) == list else x
        ecolor = tuple(1-((1-c)*.25) for c in list(cc.to_rgb('green')))
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='green', ecolor=ecolor, marker=None, linestyle=':')
        legends.append('Random')
        
    if include_sfa:
        result = ep.evaluate(eb.prediction_error,
                             algorithm=eb.Algorithms.SFA, 
                             N=N, 
                             k=k,
                             k_eval=k_eval, 
                             p=p, 
                             K=K, 
                             whitening=whitening,
                             seed=seed,
                             num_steps_test=N2,
                             iterations=iterations,
                             noisy_dims=noisy_dims,
                             keep_variance=keep_variance, 
                             #neighborhood_graph=False,
                             #weighted_edges=True,
                             #generalized_eigen_problem=generalized_eigen_problem, 
                             output_dim=output_dim, 
                             use_test_set=use_test_set, 
                             dataset=dataset,
                             measure=eb.Measures.gpfa, 
                             repetitions=repetitions, 
                             processes=processes, 
                             manage_seed=manage_seed,
                             cachedir=cachedir)
        results[eb.Algorithms.SFA] = result
        values = _get_values(result, plot_time=plot_time)
        m = np.mean(values, axis=-1)
        s = np.std(values, axis=-1)
        x = np.array(result.iter_args[iter_arg]) - x_offset
        x = x+2 if dataset == eb.Datasets.Kai and type(noisy_dims) == list else x
        ecolor = tuple(1-((1-c)*.25) for c in list(cc.to_rgb('green')))
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='green', ecolor=ecolor, marker=None, linestyle='-')
        legends.append('SFA')
    else:
        plt.errorbar(x=1, y=0, linewidth=1.2, elinewidth=.5, color='green', ecolor=cc.to_rgba('green', alpha=ecolor_alpha), marker=None, linestyle='-')
        legends.append('SFA')
    
    if include_sffa:
        result = ep.evaluate(eb.prediction_error,
                             algorithm=eb.Algorithms.SFFA, 
                             N=N, 
                             k=k,
                             k_eval=k_eval, 
                             p=p, 
                             K=K, 
                             whitening=whitening,
                             seed=seed,
                             num_steps_test=N2,
                             iterations=iterations,
                             noisy_dims=noisy_dims,
                             keep_variance=keep_variance, 
                             #neighborhood_graph=False,
                             #weighted_edges=True, 
                             #generalized_eigen_problem=generalized_eigen_problem,
                             output_dim=output_dim, 
                             use_test_set=use_test_set, 
                             dataset=dataset,
                             measure=eb.Measures.gpfa, 
                             repetitions=repetitions, 
                             processes=processes, 
                             manage_seed=manage_seed,
                             cachedir=cachedir)
        results[eb.Algorithms.SFFA] = result
        values = _get_values(result, plot_time=plot_time)
        m = np.mean(values, axis=-1)
        s = np.std(values, axis=-1)
        x = np.array(result.iter_args[iter_arg]) - x_offset
        x = x+2 if dataset == eb.Datasets.Kai and type(noisy_dims) == list else x
        ecolor = tuple(1-((1-c)*.25) for c in list(cc.to_rgb('green')))
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='green', ecolor=ecolor, marker='^', linestyle='-')
        legends.append('SFFA')
#    else:
#        plt.errorbar(x=1, y=0, linewidth=1.2, elinewidth=.5, color='green', ecolor=cc.to_rgba('green', alpha=ecolor_alpha), marker=None, linestyle='-')
#        legends.append('SFFA')
    
    if include_foreca:
        N_foreca = N
        if type(N) is list:
            N_foreca = [n for n in N if n <= 800]
        noisy_dims_foreca = noisy_dims
        if type(noisy_dims) is list:
            noisy_dims_foreca = [d for d in noisy_dims if d <= 30]
        keep_variance_foreca = keep_variance
        if type(keep_variance) is list:
            keep_variance_foreca = [v for v in keep_variance if v <= .92]
        result = ep.evaluate(eb.prediction_error,
                             algorithm=eb.Algorithms.ForeCA, 
                             N=N_foreca, 
                             k=k,
                             k_eval=k_eval,
                             p=p, 
                             K=K, 
                             whitening=whitening,
                             seed=seed,
                             num_steps_test=N2,
                             iterations=iterations,
                             noisy_dims=noisy_dims_foreca,
                             keep_variance=keep_variance_foreca, 
                             neighborhood_graph=False,
                             weighted_edges=True, 
                             #generalized_eigen_problem=generalized_eigen_problem,
                             output_dim=output_dim, 
                             use_test_set=use_test_set, 
                             dataset=dataset,
                             measure=eb.Measures.gpfa, 
                             repetitions=repetitions, 
                             processes=processes,# if processes else 16, 
                             manage_seed=manage_seed,
                             cachedir=cachedir)
        results[eb.Algorithms.ForeCA] = result
        values = _get_values(result, plot_time=plot_time)
        m = np.mean(values, axis=-1)
        s = np.std(values, axis=-1)
        x = np.array(result.iter_args[iter_arg]) - x_offset
        x = x+2 if dataset == eb.Datasets.Kai and type(noisy_dims) == list else x
        ecolor = tuple(1-((1-c)*.25) for c in list(cc.to_rgb('red')))
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='red', ecolor=ecolor, marker=None, linestyle='-.')
        legends.append('ForeCA')
#    else:
#        plt.errorbar(x=1, y=0, linewidth=1.2, elinewidth=.5, color='red', ecolor=cc.to_rgba('red', alpha=ecolor_alpha), marker=None, linestyle='-.')
#        legends.append('ForeCA')

    result = ep.evaluate(eb.prediction_error,
                         algorithm=eb.Algorithms.PFA, 
                         N=N, 
                         k=k,
                         k_eval=k_eval,
                         p=p, 
                         K=K,
                         whitening=whitening,
                         seed=seed, 
                         num_steps_test=N2,
                         iterations=iterations,
                         noisy_dims=noisy_dims,
                         keep_variance=keep_variance, 
                         neighborhood_graph=False,
                         weighted_edges=True, 
                         #generalized_eigen_problem=generalized_eigen_problem,
                         output_dim=output_dim, 
                         dataset=dataset,
                         use_test_set=use_test_set,
                         measure=eb.Measures.gpfa, 
                         repetitions=repetitions, 
                         processes=processes,
                         manage_seed=manage_seed,
                         argument_order=['algorithm'], 
                         cachedir=cachedir)
    results[eb.Algorithms.PFA] = result
    linestyles = ['--']
    #colors = ['red']
    markers = [None]
    facecolors = [None]
    #for _, _ in enumerate(result.iter_args['algorithm']):
    values = _get_values(result, plot_time=plot_time)
    m = np.mean(values, axis=-1)
    s = np.std(values, axis=-1)
    x = np.array(result.iter_args[iter_arg]) + 0 * x_offset
    x = x+2 if dataset == eb.Datasets.Kai and type(noisy_dims) == list else x
    ecolor = tuple(1-((1-c)*.25) for c in list(cc.to_rgb('red')))
    plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='red', ecolor=ecolor, markerfacecolor=facecolors[0], marker=markers[0], linestyle=linestyles[0], markersize=10)
    legends += ['PFA']

    if include_gfa1:
        k_gpfa1 = k
        if type(k_gpfa1) is list:
            k_gpfa1 = [ik for ik in k if ik <= 20]
        result = ep.evaluate(eb.prediction_error,
                             algorithm=eb.Algorithms.GPFA1, 
                             N=N, 
                             k=k_gpfa1,
                             k_eval=k_eval,
                             p=p, 
                             K=K,
                             whitening=whitening,
                             seed=seed, 
                             num_steps_test=N2,
                             iterations=iterations,
                             noisy_dims=noisy_dims,
                             keep_variance=keep_variance,
                             causal_features=causal_features, 
                             neighborhood_graph=False,
                             weighted_edges=True, 
                             generalized_eigen_problem=generalized_eigen_problem,
                             output_dim=output_dim, 
                             dataset=dataset,
                             measure=eb.Measures.gpfa, 
                             use_test_set=use_test_set,
                             repetitions=repetitions, 
                             processes=processes,
                             manage_seed=manage_seed,
                             argument_order=['algorithm'], 
                             cachedir=cachedir)
        results[eb.Algorithms.GPFA1] = result
        linestyles = ['-']
        #colors = ['blue']
        markers = ['^']
        facecolors = ['blue']
        #for _, _ in enumerate(result.iter_args['algorithm']):
        values = _get_values(result, plot_time=plot_time)
        m = np.mean(values, axis=-1)
        s = np.std(values, axis=-1)
        x = np.array(result.iter_args[iter_arg]) + x_offset
        x = x+2 if dataset == eb.Datasets.Kai and type(noisy_dims) == list else x
        ecolor = tuple(1-((1-c)*.25) for c in list(cc.to_rgb('blue')))
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='blue', ecolor=ecolor, markerfacecolor=facecolors[0], marker=markers[0], linestyle=linestyles[0], markersize=7)
        legends += ['GPFA (1)']

    if include_gfa2:
        result = ep.evaluate(eb.prediction_error,
                             algorithm=eb.Algorithms.GPFA2, 
                             N=N, 
                             k=k,
                             k_eval=k_eval,
                             p=p, 
                             K=K,
                             whitening=whitening,
                             seed=seed, 
                             num_steps_test=N2,
                             iterations=iterations,
                             noisy_dims=noisy_dims,
                             keep_variance=keep_variance,
                             causal_features=causal_features, 
                             neighborhood_graph=False,
                             weighted_edges=True, 
                             generalized_eigen_problem=generalized_eigen_problem,
                             output_dim=output_dim, 
                             dataset=dataset,
                             measure=eb.Measures.gpfa, 
                             use_test_set=use_test_set,
                             repetitions=repetitions, 
                             processes=processes,
                             manage_seed=manage_seed,
                             argument_order=['algorithm'], 
                             cachedir=cachedir)
        results[eb.Algorithms.GPFA2] = result
        linestyles = ['-']
        #colors = ['blue']
        markers = ['^']
        facecolors = ['white']
        #for i, _ in enumerate(result.iter_args['algorithm']):
        values = _get_values(result, plot_time=plot_time)
        m = np.mean(values, axis=-1)
        s = np.std(values, axis=-1)
        x = np.array(result.iter_args[iter_arg]) + 2 * x_offset
        x = x+2 if dataset == eb.Datasets.Kai and type(noisy_dims) == list else x
        ecolor = tuple(1-((1-c)*.25) for c in list(cc.to_rgb('blue')))
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='blue', ecolor=ecolor, markerfacecolor=facecolors[0], marker=markers[0], linestyle=linestyles[0], markersize=7)
        legends += ['GPFA (2)']

    if legend:
        plt.legend(legends, loc=legend_loc, prop={'size':12})
        #plt.legend(legends, loc='upper center', prop={'size':12})
        #plt.legend(legends, loc='lower center', prop={'size':12})
        #plt.legend(legends, loc='center', prop={'size':12})

    x_label = iter_arg
    x_label = x_label if x_label != 'keep_variance' else 'variance preserved'   
    x_label = x_label if x_label != 'iterations' else 'R'   
    x_label = x_label if x_label != 'N' else 'S_train'   
    x_label = x_label if x_label != 'k_eval' else 'q'   
    x_label = x_label if x_label != 'output_dim' else 'M'   
    x_label = x_label if x_label != 'noisy_dims' else 'N'   
    plt.xlabel(x_label)
    #plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(-3,3))
    
    if plot_time:
        plt.gca().set_yscale('log')
    
    if y_label:
        if plot_time:
            plt.ylabel('elapsed time in sec. (log-scale)')
        else:
            plt.ylabel('prediction error')
            
    return results



if __name__ == '__main__':
    pass
