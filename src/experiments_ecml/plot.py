import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

import experiments.experiment_base as eb


def plot_experiment(data, N, k, p, K, keep_variance, iterations, output_dim, include_random, include_foreca, include_gcfa, x_offset=0, y_label=True, legend=True):
    
    legends = []
    
    result = ep.evaluate(eb.prediction_error,
                         algorithm='random', 
                         N=N, 
                         k=k, 
                         p=p, 
                         K=K, 
                         seed=0,
                         iterations=iterations,
                         noisy_dims=0,
                         keep_variance=keep_variance, 
                         neighborhood_graph=False,
                         weighted_edges=True, 
                         output_dim=output_dim, 
                         data=data, 
                         measure='trace_of_avg_cov', 
                         repetitions=1, 
                         processes=None, 
                         cachedir='/scratch/weghebvc')
 
    # determine iter_arg
    iter_arg = result.iter_args.keys()[0]
    
    if include_random:
        m = np.mean(result.values, axis=-1)
        s = np.std(result.values, axis=-1)
        x = np.array(result.iter_args[iter_arg]) - 2 * x_offset
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='green', marker=None, linestyle=':')
        legends.append('Random')
    
    if include_foreca:
        keep_variance_foreca = keep_variance
        if type(keep_variance) is list:
            keep_variance_foreca = [v for v in keep_variance if v <= .92]
        result = ep.evaluate(eb.prediction_error,
                             algorithm='foreca', 
                             N=N, 
                             k=k,
                             p=p, 
                             K=K, 
                             seed=0,
                             iterations=iterations,
                             noisy_dims=0,
                             keep_variance=keep_variance_foreca, 
                             neighborhood_graph=False,
                             weighted_edges=True, 
                             output_dim=output_dim, 
                             data=data,
                             measure='trace_of_avg_cov', 
                             repetitions=1, 
                             processes=None, 
                             cachedir='/scratch/weghebvc')
        m = np.mean(result.values, axis=-1)
        s = np.std(result.values, axis=-1)
        x = np.array(result.iter_args[iter_arg]) - x_offset
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='red', marker=None, linestyle='-')
        legends.append('ForeCA')

    result = ep.evaluate(eb.prediction_error,
                         algorithm=['pfa', 'gpfa-1', 'gpfa-2'], 
                         N=N, 
                         k=k,
                         p=p, 
                         K=K,
                         seed=0, 
                         iterations=iterations,
                         noisy_dims=0,
                         keep_variance=keep_variance, 
                         neighborhood_graph=False,
                         weighted_edges=True, 
                         output_dim=output_dim, 
                         data=data,
                         measure='trace_of_avg_cov', 
                         repetitions=1, 
                         processes=None,
                         argument_order=['algorithm'], 
                         cachedir='/scratch/weghebvc')
    linestyles = ['--', '-', '-']
    colors = ['red', 'blue', 'blue']
    markers = [None, 'o', 'o']
    facecolors = [None, 'blue', 'white']
    for i, _ in enumerate(result.iter_args['algorithm']):
        m = np.mean(result.values[i], axis=-1)
        s = np.std(result.values[i], axis=-1)
        x = np.array(result.iter_args[iter_arg]) + i * x_offset
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color=colors[i], markerfacecolor=facecolors[i], marker=markers[i], linestyle=linestyles[i], markersize=10)
    legends += ['PFA', 'GPFA (1)', 'GPFA (2)']

    if include_gcfa:
        result = ep.evaluate(eb.prediction_error,
                             algorithm=['gcfa-1', 'gcfa-2'], 
                             N=N, 
                             k=k,
                             p=p, 
                             K=0,
                             seed=0, 
                             iterations=iterations,
                             noisy_dims=0,
                             keep_variance=keep_variance, 
                             neighborhood_graph=False,
                             weighted_edges=True, 
                             output_dim=output_dim, 
                             data=data,
                             measure='trace_of_avg_cov', 
                             repetitions=1, 
                             processes=None,
                             argument_order=['algorithm'], 
                             cachedir='/scratch/weghebvc')
        linestyles = ['-', '-']
        colors = ['blue', 'blue']
        markers = ['^', '^']
        facecolors = ['blue', 'white']
        for i, _ in enumerate(result.iter_args['algorithm']):
            m = np.mean(result.values[i], axis=-1)
            s = np.std(result.values[i], axis=-1)
            x = np.array(result.iter_args[iter_arg]) + i * x_offset
            plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color=colors[i], markerfacecolor=facecolors[i], marker=markers[i], linestyle=linestyles[i], markersize=10)
        legends += ['GPFA* (1)', 'GPFA* (2)']

    if legend:
        plt.legend(legends, loc='best', prop={'size':12})
    
    plt.xlabel(iter_arg if iter_arg != 'keep_variance' else 'variance preserved')
    if False:
        plt.gca().set_yscale('log')
        if y_label:
            plt.ylabel('prediction error (log-scale)')
    else:
        if y_label:
            plt.ylabel('prediction error')
    return



if __name__ == '__main__':
    pass