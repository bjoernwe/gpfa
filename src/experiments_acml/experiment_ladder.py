import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/easyexplot/src/')
import easyexplot as eep

import experiments.experiment_base as eb



def experiment():
    
    repeptitions = 5
    
    plt.figure()
    eep.plot(eb.prediction_error,
             algorithm='gpfa-1', 
             N=2000,#[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], 
             k=40,#[3, 5, 10, 15, 20, 30, 40, 50], 
             p=1, 
             K=1, 
             iterations=[1, 20, 50, 100],#, 200, 300, 400, 500], 
             noisy_dims=25,# 400, 500],#, 600], 
             neighborhood_graph=False,
             weighted_edges=True, 
             iteration_dim=1, 
             output_dim=1, 
             data='ladder', 
             measure='trace_of_avg_cov', 
             repetitions=repeptitions, 
             processes=None, 
             argument_order=None, 
             cachedir='/scratch/weghebvc',
             #ipython_profile='ssh', 
             plot_elapsed_time=False, 
             show_plot=False, 
             save_plot=True)
    #plt.gca().set_yscale('log')
    plt.show()
    
    
    
def plot_experiment(N=2000, k=40, noisy_dims=20, iterations=50, repetitions=50, ipython_profile=None, include_foreca=True, x_offset=0, y_label=True, legend=False):
    
    #plt.figure()
    result = eep.evaluate(eb.prediction_error,
                          algorithm='random', 
                          N=N, 
                          k=k, 
                          p=2, 
                          K=1, 
                          iterations=iterations,
                          noisy_dims=noisy_dims, 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=1, 
                          output_dim=1, 
                          data='ladder', 
                          measure='trace_of_avg_cov', 
                          repetitions=repetitions, 
                          processes=None, 
                          cachedir='/scratch/weghebvc',
                          ipython_profile=ipython_profile)

    # determine iter_arg
    iter_arg = result.iter_args.keys()[0]
    
    # plot error bars
    m = np.mean(result.values, axis=-1)
    s = np.std(result.values, axis=-1)
    x = np.array(result.iter_args[iter_arg]) - 2 * x_offset
    plt.errorbar(x=x, y=m, yerr=s, linewidth=1., color='green', marker=None, linestyle=':')
    
    if include_foreca:
        noisy_dims_foreca = noisy_dims
        if type(noisy_dims) is list:
            noisy_dims_foreca = [d for d in noisy_dims if d <= 200]
        result = eep.evaluate(eb.prediction_error,
                              algorithm='foreca', 
                              N=N, 
                              k=k,
                              p=2, 
                              K=1, 
                              iterations=iterations,
                              noisy_dims=noisy_dims_foreca, 
                              neighborhood_graph=False,
                              weighted_edges=True, 
                              iteration_dim=1, 
                              output_dim=1, 
                              data='ladder',
                              measure='trace_of_avg_cov', 
                              repetitions=repetitions, 
                              processes=8, 
                              cachedir='/scratch/weghebvc',
                              ipython_profile=ipython_profile)
        m = np.mean(result.values, axis=-1)
        s = np.std(result.values, axis=-1)
        x = np.array(result.iter_args[iter_arg]) - x_offset
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1., color='red', marker=None, linestyle='-')
     
    result = eep.evaluate(eb.prediction_error,
                          algorithm=['pfa', 'gpfa-1', 'gpfa-2'], 
                          N=N, 
                          k=k,
                          p=2, 
                          K=1, 
                          iterations=iterations,
                          noisy_dims=noisy_dims, 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=1, 
                          output_dim=1, 
                          data='ladder',
                          measure='trace_of_avg_cov', 
                          repetitions=repetitions, 
                          processes=None,
                          argument_order=['algorithm'], 
                          cachedir='/scratch/weghebvc',
                          ipython_profile=ipython_profile)
    linestyles = ['--', '-', '-']
    colors = ['red', 'blue', 'blue']
    markers = [None, 'o', 'o']
    facecolors = [None, 'blue', 'white']
    for i, _ in enumerate(result.iter_args['algorithm']):
        m = np.mean(result.values[i], axis=-1)
        s = np.std(result.values[i], axis=-1)
        x = np.array(result.iter_args[iter_arg]) + i * x_offset
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1., color=colors[i], markerfacecolor=facecolors[i], marker=markers[i], linestyle=linestyles[i], markersize=10)
    if legend:
        plt.legend(['random', 'ForeCA', 'PFA', 'GPFA (1)', 'GPFA (2)'], loc='best', prop={'size':12}) 
    
    plt.xlabel(iter_arg)
    if False:
        if y_label:
            plt.ylabel('prediction error (log-scale)')
        plt.gca().set_yscale('log')
    else:
        if y_label:
            plt.ylabel('prediction error')
    #plt.show()



def main():
    #experiment()
    plt.subplot(1, 2, 1)
    plt.title('(a)')
    plot_experiment(noisy_dims=[1, 5, 10, 15, 20, 25, 30, 35, 40], x_offset=0.)
    plt.subplot(1, 2, 2)
    plt.title('(b)')
    plot_experiment(k=[3, 5, 10, 15, 20, 30, 40, 50], x_offset=0., y_label=False, legend=True)
    plt.show()



if __name__ == '__main__':
    main()
