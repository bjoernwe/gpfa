import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

import experiments.experiment_base as eb



def experiment(N=2500, k=50, iterations=50, noisy_dims=40, data='ladder'):
    
    repeptitions = 20
    
    #plt.figure()
    ep.plot(eb.prediction_error,
            #algorithm=['random', 'pfa', 'gpfa-1', 'gpfa-2', 'gcfa-1', 'gcfa-2'], 
            algorithm=['random', 'pfa', 'gcfa-1', 'gcfa-2'], 
            N=N, 
            k=k, 
            p=2, 
            K=2, 
            seed=0,
            iterations=iterations, 
            noisy_dims=noisy_dims, 
            neighborhood_graph=False,
            weighted_edges=True, 
            iteration_dim=1, 
            output_dim=1, 
            data=data, 
            measure='trace_of_avg_cov', 
            repetitions=repeptitions, 
            processes=None, 
            argument_order=['N', 'iterations'], 
            cachedir='/scratch/weghebvc',
            plot_elapsed_time=False, 
            show_plot=False, 
            save_plot_path='./plots')
    #plt.gca().set_yscale('log')
    #plt.show()
    

def plot_experiment(N=2500, k=50, noisy_dims=40, iterations=50, repetitions=50, include_foreca=True, x_offset=0, y_label=True, legend=False):
    
    #plt.figure()
    result = ep.evaluate(eb.prediction_error,
                         algorithm='random', 
                         N=N, 
                         k=k, 
                         p=2, 
                         K=2, 
                         seed=0,
                         iterations=iterations,
                         noisy_dims=noisy_dims, 
                         neighborhood_graph=False,
                         weighted_edges=True, 
                         #iteration_dim=1, 
                         output_dim=1, 
                         data='ladder', 
                         measure='trace_of_avg_cov', 
                         repetitions=repetitions, 
                         processes=None, 
                         cachedir='/scratch/weghebvc')

    # determine iter_arg
    iter_arg = result.iter_args.keys()[0]
    
    # plot error bars
    m = np.mean(result.values, axis=-1)
    s = np.std(result.values, axis=-1)
    x = np.array(result.iter_args[iter_arg]) - 2 * x_offset
    plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='green', marker=None, linestyle=':')
    
    if include_foreca:
        noisy_dims_foreca = noisy_dims
        if type(noisy_dims) is list:
            noisy_dims_foreca = [d for d in noisy_dims if d <= 50]
        result = ep.evaluate(eb.prediction_error,
                             algorithm='foreca', 
                             N=N, 
                             k=k,
                             p=2, 
                             K=2, 
                             seed=0,
                             iterations=iterations,
                             noisy_dims=noisy_dims_foreca, 
                             neighborhood_graph=False,
                             weighted_edges=True, 
                             #iteration_dim=1, 
                             output_dim=1, 
                             data='ladder',
                             measure='trace_of_avg_cov', 
                             repetitions=repetitions, 
                             processes=None, 
                             cachedir='/scratch/weghebvc')
        m = np.mean(result.values, axis=-1)
        s = np.std(result.values, axis=-1)
        x = np.array(result.iter_args[iter_arg]) - x_offset
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='red', marker=None, linestyle='-')
     
    result = ep.evaluate(eb.prediction_error,
                         algorithm=['pfa', 'gcfa-1', 'gcfa-2'], 
                         N=N, 
                         k=k,
                         p=2, 
                         K=2,
                         seed=0, 
                         iterations=iterations,
                         noisy_dims=noisy_dims, 
                         neighborhood_graph=False,
                         weighted_edges=True, 
                         #iteration_dim=1, 
                         output_dim=1, 
                         data='ladder',
                         measure='trace_of_avg_cov', 
                         repetitions=repetitions, 
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
    if legend:
        plt.legend(['random', 'ForeCA', 'PFA', 'GPFA (1)', 'GPFA (2)'], loc='best', prop={'size':12}) 
    
    plt.xlabel(iter_arg if iter_arg != 'N' else 'S')
    plt.xlabel(iter_arg if iter_arg != 'noisy_dims' else '# noisy dimensions')
    if False:
        if y_label:
            plt.ylabel('prediction error (log-scale)')
        plt.gca().set_yscale('log')
    else:
        if y_label:
            plt.ylabel('prediction error')
    #plt.show()
    

def main():
    
    # ladder
    plt.figure()
    plt.subplot(2, 2, 1)
    experiment(noisy_dims=[0, 10, 20, 30, 40, 50])
    #plt.subplot(2, 2, 2)
    #experiment(N=[1500, 2000, 2500])
    #plt.subplot(2, 2, 3)
    #experiment(iterations=[1, 10, 30, 50, 100])
    plt.subplot(2, 2, 4)
    experiment(k=[1, 2, 5, 10, 15, 20, 30, 40, 50])

    plt.show()


def main_plot():
    #plt.subplot(1, 2, 1)
    #plt.title('(a)')
    plt.figure()
    plot_experiment(noisy_dims=[1, 10, 20, 30, 40, 50], x_offset=0.5)
    #plt.subplot(1, 2, 2)
    #plt.title('(b)')
    plt.figure()
    plot_experiment(k=[1, 2, 5, 10, 15, 20, 30, 40, 50], x_offset=0.5, legend=True)
    plt.show()



if __name__ == '__main__':
    #main()
    main_plot()
