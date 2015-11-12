import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

import experiments.experiment_base as eb



def experiment(N=2000, keep_variance=1., k=20, iterations=50, output_dim=1):
    
    #plt.figure()
    ep.plot(eb.prediction_error,
            algorithm=['pfa', 'gpfa-1', 'gpfa-2', 'gcfa-1', 'gcfa-2'], 
            N=N, 
            keep_variance=keep_variance,
            k=k,
            p=1, 
            K=0, 
            seed=0,
            iterations=iterations,
            noisy_dims=0, 
            neighborhood_graph=False,
            weighted_edges=True, 
            iteration_dim=output_dim, 
            output_dim=output_dim, 
            data='eeg',
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
    
    
    
def main():
    
    # mario
    plt.figure()
    plt.subplot(2, 2, 1)
    experiment(keep_variance=list(np.arange(.85, 1., .01)))
    plt.subplot(2, 2, 2)
    experiment(N=[1500, 2000, 2500])
    plt.subplot(2, 2, 3)
    experiment(iterations=[20, 40, 60, 80, 100])
    plt.subplot(2, 2, 4)
    experiment(k=range(5,51,5))

    plt.show()



if __name__ == '__main__':
    main()
