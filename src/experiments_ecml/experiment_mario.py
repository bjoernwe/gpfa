import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

import experiments.experiment_base as eb



def experiment(N=2500, keep_variance=.99, k=20, iterations=150, output_dim=5):
    
    #plt.figure()
    ep.plot(eb.prediction_error,
            algorithm=['pfa', 'gcfa-1', 'gcfa-2'], 
            N=N, 
            keep_variance=keep_variance,
            k=k,
            p=2, 
            K=0, 
            seed=0,
            iterations=iterations,
            noisy_dims=0, 
            neighborhood_graph=False,
            weighted_edges=True, 
            iteration_dim=output_dim, 
            output_dim=output_dim, 
            data='mario_window',
            measure='trace_of_avg_cov', 
            reverse_error=False,
            repetitions=1, 
            processes=None,
            argument_order=['N', 'iterations'], 
            cachedir='/scratch/weghebvc',
            plot_elapsed_time=False, 
            show_plot=False, 
            save_plot_path='./plots')
    #plt.gca().set_yscale('log')
    #plt.show()
    
    
def plot_experiment(N=2500, k=20, keep_variance=.92, iterations=150, output_dim=5, include_foreca=True, x_offset=0, y_label=True, legend=False):
    
    #plt.figure()
    result = ep.evaluate(eb.prediction_error,
                         algorithm='random', 
                         N=N, 
                         k=k, 
                         p=2, 
                         K=0, 
                         seed=0,
                         iterations=iterations,
                         noisy_dims=0,
                         keep_variance=keep_variance, 
                         neighborhood_graph=False,
                         weighted_edges=True, 
                         iteration_dim=output_dim, 
                         output_dim=output_dim, 
                         data='mario_window', 
                         measure='trace_of_avg_cov', 
                         repetitions=1, 
                         processes=None, 
                         cachedir='/scratch/weghebvc')
 
    # determine iter_arg
    iter_arg = result.iter_args.keys()[0]
#     
#     # plot error bars
#     m = np.mean(result.values, axis=-1)
#     s = np.std(result.values, axis=-1)
#     x = np.array(result.iter_args[iter_arg]) - 2 * x_offset
#     plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='green', marker=None, linestyle=':')
    
    if include_foreca:
        keep_variance_foreca = keep_variance
        if type(keep_variance) is list:
            keep_variance_foreca = [v for v in keep_variance if v <= .92]
        result = ep.evaluate(eb.prediction_error,
                             algorithm='foreca', 
                             N=N, 
                             k=k,
                             p=2, 
                             K=0, 
                             seed=0,
                             iterations=iterations,
                             noisy_dims=0,
                             keep_variance=keep_variance_foreca, 
                             neighborhood_graph=False,
                             weighted_edges=True, 
                             iteration_dim=output_dim, 
                             output_dim=output_dim, 
                             data='mario_window',
                             measure='trace_of_avg_cov', 
                             repetitions=1, 
                             processes=None, 
                             cachedir='/scratch/weghebvc')
        m = np.mean(result.values, axis=-1)
        s = np.std(result.values, axis=-1)
        x = np.array(result.iter_args[iter_arg]) - x_offset
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='red', marker=None, linestyle='-')
    else:
        plt.plot(0, linewidth=1.2, color='red', marker=None, linestyle='-')
     
    result = ep.evaluate(eb.prediction_error,
                         algorithm=['pfa', 'gcfa-1', 'gcfa-2'], 
                         N=N, 
                         k=k,
                         p=2, 
                         K=0,
                         seed=0, 
                         iterations=iterations,
                         noisy_dims=0,
                         keep_variance=keep_variance, 
                         neighborhood_graph=False,
                         weighted_edges=True, 
                         iteration_dim=output_dim, 
                         output_dim=output_dim, 
                         data='mario_window',
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
        print x
        print m 
        print s
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color=colors[i], markerfacecolor=facecolors[i], marker=markers[i], linestyle=linestyles[i], markersize=10)
    if legend:
        #plt.legend(['random', 'ForeCA', 'PFA', 'GPFA (1)', 'GPFA (2)'], loc='best', prop={'size':12}) 
        plt.legend(['ForeCA', 'PFA', 'GPFA (1)', 'GPFA (2)'], loc='best', prop={'size':12}) 
    
    plt.xlabel(iter_arg if iter_arg != 'keep_variance' else 'kept variance')
    if False:
        if y_label:
            plt.ylabel('prediction error (log-scale)')
        plt.gca().set_yscale('log')
    else:
        if y_label:
            plt.ylabel('prediction error')
    #plt.show()
    
    
def calc_dimensions(keep_variance, N=2500):
    for variance in keep_variance:
        data_train, _ = eb.generate_training_data(N=N, 
                                                  keep_variance=variance, 
                                                  seed=0,
                                                  data='mario_window')
        print data_train.shape
        print 'variance %f <-> dimensions %d' % (variance, data_train.shape[1])
        
   
def main():
    
    # mario
    plt.figure()
    plt.subplot(2, 2, 1)
    experiment(keep_variance=list(np.arange(.85, 1., .01)))
    plt.subplot(2, 2, 2)
    experiment(N=[1500, 1750, 2000, 2250, 2500])
    plt.subplot(2, 2, 3)
    experiment(iterations=[20, 40, 60, 80, 100, 150])
    plt.subplot(2, 2, 4)
    experiment(k=range(5,51,5))

    plt.show()


def main_plot():
    #plt.subplot(1, 2, 1)
    #plt.title('(a)')
    plt.figure()
    plot_experiment(keep_variance=list(np.arange(.85, 1., .01)), x_offset=0.)
    #plt.subplot(2, 2, 2)
    #plt.title('(b)')
    #plot_experiment(N=[1500, 1750, 2000, 2250, 2500], x_offset=0., y_label=False)
    #plt.subplot(2, 2, 3)
    #plt.title('(c)')
    #plot_experiment(iterations=[20, 40, 60, 80, 100, 150], x_offset=0.)
    #plt.subplot(1, 2, 2)
    #plt.title('(b)')
    plt.figure()
    plot_experiment(k=range(5,51,5), include_foreca=True, x_offset=0., y_label=False, legend=True)
    plt.show()


if __name__ == '__main__':
    #main()
    main_plot()
    #calc_dimensions(keep_variance=list(np.arange(.85, 1., .01)))
