import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

import experiments.experiment_base as eb

import plot



def experiment(N=2500, k=50, p=1, iterations=50, noisy_dims=40, data='ladder'):
    
    repeptitions = 5
    
    #plt.figure()
    ep.plot(eb.prediction_error,
            #algorithm=['random', 'pfa', 'gpfa-1', 'gpfa-2', 'gcfa-1', 'gcfa-2'], 
            algorithm=['random', 'pfa', 'gcfa-1', 'gcfa-2'], 
            N=N, 
            k=k, 
            p=p, 
            P=2,
            K=2, 
            seed=0,
            iterations=iterations, 
            noisy_dims=noisy_dims, 
            neighborhood_graph=False,
            weighted_edges=True, 
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

    

def plot_experiment(N=2500, k=50, p=1, K=0, noisy_dims=40, iterations=50, output_dim=1, repetitions=25, include_random=True, include_foreca=True, include_gcfa=True, x_offset=0, y_label=True, legend=False):
    plot.plot_experiment(data='ladder', 
                         N=N, 
                         k=k, 
                         p=p, 
                         P=2,
                         K=K, 
                         noisy_dims=noisy_dims,
                         keep_variance=1., 
                         iterations=iterations, 
                         output_dim=output_dim,
                         repetitions=repetitions, 
                         include_random=include_random, 
                         include_foreca=include_foreca, 
                         include_gcfa=include_gcfa, 
                         x_offset=x_offset, 
                         y_label=y_label, 
                         legend=legend,
                         seed=0)



def main():
    # ladder
    plt.figure()
    plt.subplot(2, 2, 1)
    experiment(noisy_dims=[1, 10, 20, 30, 40, 50])
    plt.subplot(2, 2, 2)
    experiment(N=[1500, 2000, 2500])
    plt.subplot(2, 2, 3)
    experiment(iterations=[1, 10, 30, 50, 100, 150])
    plt.subplot(2, 2, 4)
    experiment(k=[2, 5, 10, 15, 20])#, 30, 40, 50])
    plt.show()


def main_plot():
    # ladder
    plt.figure()
    #plt.subplot(2, 2, 1)
    plot_experiment(noisy_dims=[1, 10, 20, 30, 40, 50])
    #plt.subplot(2, 2, 2)
    plt.figure()
    plot_experiment(N=[1500, 2000, 2500], include_foreca=False)
    #plt.subplot(2, 2, 3)
    plt.figure()
    plot_experiment(iterations=[1, 10, 30, 50, 100], include_foreca=False)
    #plt.subplot(2, 2, 4)
    plt.figure()
    plot_experiment(k=[2, 5, 10, 15, 20, 30, 40, 50], include_foreca=False, legend=True)
    plt.show()



if __name__ == '__main__':
    #main()
    main_plot()
