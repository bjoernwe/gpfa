import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

import experiments.experiment_base as eb

import plot



def experiment(N=10000, keep_variance=1., k=2, p=1, iterations=200, output_dim=9):
    
    #plt.figure()
    ep.plot(eb.prediction_error,
            algorithm=[eb.Algorithms.SFA, eb.Algorithms.PFA, eb.Algorithms.GPFA2], 
            N=N, 
            keep_variance=keep_variance,
            k=k,
            p=p, 
            K=0, 
            seed=0,
            iterations=iterations,
            noisy_dims=0, 
            neighborhood_graph=False,
            weighted_edges=True, 
            output_dim=output_dim, 
            dataset=eb.Datasets.Mario_window,
            use_test_set=True,
            measure=eb.Measures.gpfa, 
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
    
    
# def plot_experiment(N=10000, k=2, p=2, K=0, keep_variance=.98, iterations=100, output_dim=9, 
#                     include_random=False, include_sfa=True, include_foreca=False, 
#                     include_gcfa=True, x_offset=0, y_label=True, legend=False):
#     plot.plot_experiment(dataset=eb.Datasets.Mario_window, 
#                          N=N, 
#                          k=k, 
#                          p=p, 
#                          K=K, 
#                          noisy_dims=0,
#                          keep_variance=keep_variance, 
#                          iterations=iterations, 
#                          output_dim=output_dim,
#                          repetitions=1, 
#                          include_random=include_random,
#                          include_sfa=include_sfa, 
#                          include_foreca=include_foreca, 
#                          include_gcfa=include_gcfa, 
#                          x_offset=x_offset, 
#                          y_label=y_label, 
#                          legend=legend,
#                          seed=0)
    
    
    
def calc_dimensions(keep_variance, N=2500):
    for variance in keep_variance:
        data_train, _ = eb.generate_training_data(N=N, 
                                                  keep_variance=variance, 
                                                  seed=0,
                                                  data='mario_window')
        print data_train.shape
        print 'variance %f <-> dimensions %d' % (variance, data_train.shape[1])
        
   
def main():
    plt.figure()
    experiment(keep_variance=list(np.arange(.90, 1.01, .01)))
    plt.figure()
    experiment(N=[2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    plt.figure()
    experiment(iterations=[20, 50, 100, 150, 200])
    plt.figure()
    experiment(k=[2, 5, 10, 20])
    plt.figure()
    experiment(p=[1,2,3,4])
    plt.figure()
    experiment(output_dim=[3, 6, 9, 12])
    plt.show()


def main_plot():
#     plt.figure()
#     plot_experiment(keep_variance=list(np.arange(.80, 1.01, .02)), output_dim=3)
#     plt.figure()
#     plot_experiment(keep_variance=list(np.arange(.80, 1.01, .02)), output_dim=6)
    plt.figure()
    plot_experiment(keep_variance=list(np.arange(.80, 1.01, .02)), output_dim=9)
#     plt.figure()
#     plot_experiment(keep_variance=list(np.arange(.80, 1.01, .02)), output_dim=12, legend=True)
    plt.figure()
    plot_experiment(N=[2000, 3000, 4000, 5000])
    plt.figure()
    plot_experiment(iterations=[1, 10, 30, 50, 100])
    plt.figure()
    plot_experiment(k=[2, 5, 10, 20, 40])
    plt.figure()
    plot_experiment(p=[1,2])
    plt.show()
    
    
def plot_output_dims():
    for i in range(1,10):
        plt.subplot(3,3,i+1)
        plot_experiment(keep_variance=list(np.arange(.80, 1.01, .02)), output_dim=i, include_foreca=False)
    plt.show()


if __name__ == '__main__':
    main()
    #main_plot()
    #plot_output_dims()
    #calc_dimensions(keep_variance=list(np.arange(.8, 1.01, .01)))
