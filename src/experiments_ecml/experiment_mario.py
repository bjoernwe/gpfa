import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

import experiments.experiment_base as eb

import plot



def experiment(N=2500, keep_variance=.84, k=2, p=2, iterations=50, output_dim=5):
    
    #plt.figure()
    ep.plot(eb.prediction_error,
            algorithm=['pfa', 'gpfa-1', 'gpfa-2'],#, 'gcfa-1', 'gcfa-2'], 
            N=N, 
            keep_variance=keep_variance,
            k=k,
            p=p, 
            K=1, 
            seed=0,
            iterations=iterations,
            noisy_dims=0, 
            neighborhood_graph=True,
            weighted_edges=True, 
            output_dim=output_dim, 
            data='mario_window',
            measure='trace_of_avg_cov', 
            reverse_error=False,
            repetitions=1, 
            processes=None,
            argument_order=['keep_variance', 'p', 'N', 'iterations'], 
            cachedir='/scratch/weghebvc',
            plot_elapsed_time=False, 
            show_plot=False, 
            save_plot_path='./plots')
    #plt.gca().set_yscale('log')
    #plt.show()
    
    
def plot_experiment(N=2500, k=2, p=2, K=0, keep_variance=.86, iterations=50, output_dim=5, include_random=False, include_foreca=False, include_gcfa=True, x_offset=0, y_label=True, legend=False):
    plot.plot_experiment(data='mario_window', 
                         N=N, 
                         k=k, 
                         p=p, 
                         K=K, 
                         keep_variance=keep_variance, 
                         iterations=iterations, 
                         output_dim=output_dim, 
                         include_random=include_random, 
                         include_foreca=include_foreca, 
                         include_gcfa=include_gcfa, 
                         x_offset=x_offset, 
                         y_label=y_label, 
                         legend=legend)
    
    
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
    #plt.subplot(2, 2, 1)
    experiment(keep_variance=list(np.arange(.70, 1.01, .02)))
    #plt.subplot(2, 2, 2)
    #experiment(N=[1500, 1750, 2000, 2250, 2500])
    #plt.subplot(2, 2, 3)
    #experiment(iterations=[20, 40, 60, 80, 100, 150])
    #plt.subplot(2, 2, 4)
    #experiment(k=range(2,5) + range(5,21,5))
    #experiment(output_dim=range(1,11))
    plt.show()


def main_plot():
    #plt.subplot(1, 2, 1)
    #plt.title('(a)')
    plt.figure()
    plot_experiment(keep_variance=list(np.arange(.80, 1.01, .02)), x_offset=0.)
    #plt.subplot(2, 2, 2)
    #plt.title('(b)')
    #plot_experiment(N=[1500, 1750, 2000, 2250, 2500], x_offset=0., y_label=False)
    #plt.subplot(2, 2, 3)
    #plt.title('(c)')
    #plot_experiment(iterations=[20, 40, 60, 80, 100, 150], x_offset=0.)
    #plt.subplot(1, 2, 2)
    #plt.title('(b)')
    plt.figure()
    plot_experiment(k=range(2,5) + range(5,21,5), x_offset=0., legend=True)
    plt.show()
    
    
def plot_output_dims():
    for i in range(1,10):
        plt.subplot(3,3,i+1)
        plot_experiment(keep_variance=list(np.arange(.80, 1.01, .02)), output_dim=i, include_foreca=False)
    plt.show()


if __name__ == '__main__':
    #main()
    main_plot()
    #plot_output_dims()
    #calc_dimensions(keep_variance=list(np.arange(.8, 1.01, .01)))
