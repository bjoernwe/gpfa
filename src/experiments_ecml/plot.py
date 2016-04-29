import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

import experiments.experiment_base as eb


def _get_values(result, plot_time=False):
    if plot_time:
        return result.elapsed_times / 1000.
    return result.values


def plot_experiment(dataset, N, k, p, K, noisy_dims, keep_variance, iterations, output_dim, repetitions, include_random, include_sfa, include_foreca, include_gcfa, N2, x_offset=0, y_label=True, legend=True, plot_time=False, cachedir='/scratch/weghebvc', seed=0):
    
    legends = []
    
    processes = None
    
    result = ep.evaluate(eb.prediction_error,
                         algorithm=eb.Algorithms.Random, 
                         N=N, 
                         k=k, 
                         p=p, 
                         K=K, 
                         seed=seed,
                         num_steps_test=N2,
                         iterations=iterations,
                         noisy_dims=noisy_dims,
                         keep_variance=keep_variance, 
                         neighborhood_graph=False,
                         weighted_edges=True, 
                         output_dim=output_dim,
                         use_test_set=True, 
                         dataset=dataset, 
                         measure=eb.Measures.gpfa, 
                         repetitions=repetitions, 
                         processes=processes,
                         manage_seed='external', 
                         cachedir=cachedir)
 
    # determine iter_arg
    iter_arg = result.iter_args.keys()[0]
    
    if include_random:
        values = _get_values(result, plot_time=plot_time)
        m = np.mean(values, axis=-1)
        s = np.std(values, axis=-1)
        x = np.array(result.iter_args[iter_arg]) - 2 * x_offset
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='green', marker=None, linestyle=':')
        legends.append('Random')
        
    if include_sfa:
        result = ep.evaluate(eb.prediction_error,
                             algorithm=eb.Algorithms.SFA, 
                             N=N, 
                             k=k,
                             p=p, 
                             K=K, 
                             seed=seed,
                             num_steps_test=N2,
                             iterations=iterations,
                             noisy_dims=noisy_dims,
                             keep_variance=keep_variance, 
                             #neighborhood_graph=False,
                             #weighted_edges=True, 
                             output_dim=output_dim, 
                             use_test_set=True, 
                             dataset=dataset,
                             measure=eb.Measures.gpfa, 
                             repetitions=repetitions, 
                             processes=processes, 
                             manage_seed='external',
                             cachedir=cachedir)
        values = _get_values(result, plot_time=plot_time)
        m = np.mean(values, axis=-1)
        s = np.std(values, axis=-1)
        x = np.array(result.iter_args[iter_arg]) - x_offset
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='green', marker=None, linestyle='-')
        legends.append('SFA')
    else:
        plt.errorbar(x=1, y=0, linewidth=1.2, elinewidth=.5, color='green', marker=None, linestyle='-')
        legends.append('SFA')
    
    if include_foreca:
        noisy_dims_foreca = noisy_dims
        if type(noisy_dims) is list:
            noisy_dims_foreca = [d for d in noisy_dims if d <= 100]
        keep_variance_foreca = keep_variance
        if type(keep_variance) is list:
            keep_variance_foreca = [v for v in keep_variance if v <= .92]
        result = ep.evaluate(eb.prediction_error,
                             algorithm=eb.Algorithms.ForeCA, 
                             N=N, 
                             k=k,
                             p=p, 
                             K=K, 
                             seed=seed,
                             num_steps_test=N2,
                             iterations=iterations,
                             noisy_dims=noisy_dims_foreca,
                             keep_variance=keep_variance_foreca, 
                             neighborhood_graph=False,
                             weighted_edges=True, 
                             output_dim=output_dim, 
                             use_test_set=True, 
                             dataset=dataset,
                             measure=eb.Measures.gpfa, 
                             repetitions=repetitions, 
                             processes=processes if processes else 16, 
                             manage_seed='external',
                             cachedir=cachedir)
        values = _get_values(result, plot_time=plot_time)
        m = np.mean(values, axis=-1)
        s = np.std(values, axis=-1)
        x = np.array(result.iter_args[iter_arg]) - x_offset
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='red', marker=None, linestyle='-')
        legends.append('ForeCA')
    else:
        plt.errorbar(x=1, y=0, linewidth=1.2, elinewidth=.5, color='red', marker=None, linestyle='-')
        legends.append('ForeCA')

    result = ep.evaluate(eb.prediction_error,
                         algorithm=[eb.Algorithms.PFA], 
                         N=N, 
                         k=k,
                         p=p, 
                         K=K,
                         seed=seed, 
                         num_steps_test=N2,
                         iterations=iterations,
                         noisy_dims=noisy_dims,
                         keep_variance=keep_variance, 
                         neighborhood_graph=False,
                         weighted_edges=True, 
                         output_dim=output_dim, 
                         dataset=dataset,
                         use_test_set=True,
                         measure=eb.Measures.gpfa, 
                         repetitions=repetitions, 
                         processes=processes,
                         manage_seed='external',
                         argument_order=['algorithm'], 
                         cachedir=cachedir)
    linestyles = ['--']
    colors = ['red']
    markers = [None]
    facecolors = [None]
    for i, _ in enumerate(result.iter_args['algorithm']):
        values = _get_values(result, plot_time=plot_time)
        m = np.mean(values[i], axis=-1)
        s = np.std(values[i], axis=-1)
        x = np.array(result.iter_args[iter_arg]) + i * x_offset
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color=colors[i], markerfacecolor=facecolors[i], marker=markers[i], linestyle=linestyles[i], markersize=10)
    legends += ['PFA']

    if include_gcfa:
        result = ep.evaluate(eb.prediction_error,
                             algorithm=[eb.Algorithms.GPFA2],#[eb.Algorithms.GPFA1, eb.Algorithms.GPFA2], 
                             N=N, 
                             k=k,
                             p=p, 
                             K=K,
                             seed=seed, 
                             num_steps_test=N2,
                             iterations=iterations,
                             noisy_dims=noisy_dims,
                             keep_variance=keep_variance, 
                             neighborhood_graph=False,
                             weighted_edges=True, 
                             output_dim=output_dim, 
                             dataset=dataset,
                             measure=eb.Measures.gpfa, 
                             use_test_set=True,
                             repetitions=repetitions, 
                             processes=processes,
                             manage_seed='external',
                             argument_order=['algorithm'], 
                             cachedir=cachedir)
        linestyles = ['-', '-']
        colors = ['blue', 'blue']
        markers = ['^', '^']
        facecolors = ['blue', 'white']
        for i, _ in enumerate(result.iter_args['algorithm']):
            values = _get_values(result, plot_time=plot_time)
            m = np.mean(values[i], axis=-1)
            s = np.std(values[i], axis=-1)
            x = np.array(result.iter_args[iter_arg]) + i * x_offset
            plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color=colors[i], markerfacecolor=facecolors[i], marker=markers[i], linestyle=linestyles[i], markersize=10)
        legends += ['GPFA (1)', 'GPFA (2)']

    if legend:
        plt.legend(legends, loc='best', prop={'size':12})

    x_label = iter_arg
    x_label = x_label if x_label != 'keep_variance' else 'variance preserved'   
    x_label = x_label if x_label != 'iterations' else 'R'   
    x_label = x_label if x_label != 'N' else 'S'   
    plt.xlabel(x_label)
    
    if y_label:
        if plot_time:
            plt.ylabel('elapsed time in sec. (log-scale)')
            plt.gca().set_yscale('log')
        else:
            plt.ylabel('prediction error')
    return



if __name__ == '__main__':
    pass
