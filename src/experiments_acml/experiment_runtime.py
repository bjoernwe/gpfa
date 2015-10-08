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
             iterations=[1, 20, 50, 100, 200],#, 300, 400, 500], 
             noisy_dims=400, 
             neighborhood_graph=False,
             weighted_edges=True, 
             iteration_dim=2, 
             output_dim=2, 
             data='swiss_roll', 
             measure='trace_of_avg_cov', 
             repetitions=repeptitions, 
             processes=None, 
             argument_order=None, 
             cachedir='/scratch/weghebvc',
             #ipython_profile='ssh', 
             plot_elapsed_time=False, 
             show_plot=False, 
             save_plot=True)
    plt.gca().set_yscale('log')

    plt.show()
    
    
    
def plot_experiment(N=2000, k=40, noisy_dims=200, iterations=100, repetitions=20, ipython_profile=None, include_foreca=True, x_offset=0, y_label=True, legend=False):
    
    #plt.figure()
    result = eep.evaluate(eb.prediction_error,
                          algorithm='random', 
                          N=N, 
                          k=k, 
                          p=1, 
                          K=1, 
                          iterations=iterations,
                          noisy_dims=noisy_dims, 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='trace_of_avg_cov', 
                          repetitions=repetitions, 
                          processes=None, 
                          cachedir='/scratch/weghebvc/time2',
                          ipython_profile=ipython_profile)

    # determine iter_arg
    iter_arg = result.iter_args.keys()[0]
    
    # plot error bars
    m = np.mean(result.elapsed_times / 1000., axis=-1)
    s = np.std(result.elapsed_times / 1000., axis=-1)
    x = np.array(result.iter_args[iter_arg]) - 2 * x_offset
    #plt.errorbar(x=x, y=m, yerr=s, linewidth=1, color='green', marker=None, linestyle=':')
    
    if include_foreca:
        noisy_dims_foreca = noisy_dims
        if type(noisy_dims) is list:
            noisy_dims_foreca = [d for d in noisy_dims if d <= 200]
        result = eep.evaluate(eb.prediction_error,
                              algorithm='foreca', 
                              N=N, 
                              k=k,
                              p=1, 
                              K=1, 
                              iterations=iterations,
                              noisy_dims=noisy_dims_foreca, 
                              neighborhood_graph=False,
                              weighted_edges=True, 
                              iteration_dim=2, 
                              output_dim=2, 
                              data='swiss_roll', 
                              measure='trace_of_avg_cov', 
                              repetitions=repetitions, 
                              processes=10, 
                              cachedir='/scratch/weghebvc/time2',
                              ipython_profile=ipython_profile)
        m = np.mean(result.elapsed_times / 1000., axis=-1)
        s = np.std(result.elapsed_times / 1000., axis=-1)
        x = np.array(result.iter_args[iter_arg]) - x_offset
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1, color='red', marker=None, linestyle='-')
     
    result = eep.evaluate(eb.prediction_error,
                          algorithm=['pfa', 'gpfa-1', 'gpfa-2'], 
                          N=N, 
                          k=k,
                          p=1, 
                          K=1, 
                          iterations=iterations,
                          noisy_dims=noisy_dims, 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='trace_of_avg_cov', 
                          repetitions=repetitions, 
                          processes=None,
                          argument_order=['algorithm'], 
                          cachedir='/scratch/weghebvc/time2',
                          ipython_profile=ipython_profile)
    linestyles = ['--', '-', '-']
    colors = ['red', 'blue', 'blue']
    markers = [None, 'o', 'o']
    facecolors = [None, 'blue', 'white']
    for i, _ in enumerate(result.iter_args['algorithm']):
        m = np.mean(result.elapsed_times[i] / 1000., axis=-1)
        s = np.std(result.elapsed_times[i] / 1000., axis=-1)
        x = np.array(result.iter_args[iter_arg]) + i * x_offset
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1, color=colors[i], markerfacecolor=facecolors[i], marker=markers[i], linestyle=linestyles[i], markersize=10)
    if legend:
        plt.legend(['ForeCA', 'PFA', 'GPFA (1)', 'GPFA (2)'], loc='best', prop={'size':12}) 
    
    plt.xlabel(iter_arg)
    if True:
        if y_label:
            plt.ylabel('runtime in sec. (log-scale)')
        plt.gca().set_yscale('log')
    else:
        if y_label:
            plt.ylabel('runtime in sec.')
            
    #plt.gca().set_yscale('log')
    #plt.show()



def main():
    #experiment()
    plt.subplot(2, 2, 1)
    plt.gca().set_ylim([7e-1, 1e5])
    plt.title('(a)')
    plot_experiment(noisy_dims=[0, 50, 100, 200, 300, 400, 500], x_offset=0)
    plt.subplot(2, 2, 2)
    plt.gca().set_ylim([7e-1, 1e5])
    plt.title('(b)')
    plot_experiment(N=[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200], x_offset=0, y_label=False)
    plt.xlabel('S')
    plt.subplot(2, 2, 3)
    plt.gca().set_ylim([7e-1, 1e5])
    plt.title('(c)')
    plot_experiment(iterations=[1, 10, 20, 30, 40, 50, 100], x_offset=0)
    plt.subplot(2, 2, 4)
    plt.gca().set_ylim([7e-1, 1e5])
    plt.title('(d)')
    plot_experiment(k=[3, 5, 10, 15, 20, 30, 40, 50], x_offset=0, y_label=False, legend=True)
    plt.show()



if __name__ == '__main__':
    main()
