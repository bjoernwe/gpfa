import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

import experiments.experiment_base as eb



def experiment(N=2500, keep_variance=1., k=2, p=2, iterations=60, output_dim=2):
    
    #plt.figure()
    ep.plot(eb.prediction_error,
            algorithm=['pfa', 'gcfa-1', 'gcfa-2'], 
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
    #plt.subplot(2,2,1)
    #experiment(p=[1,2,3,4])
    #plt.subplot(1,2,2)
    #experiment(output_dim=[1,2,3,4])
    #plt.subplot(2, 2, 1)
    experiment(keep_variance=list(np.arange(.85, 1., .01)))
    #plt.subplot(2, 2, 2)
    #experiment(N=[1500, 2000, 2500])
    #plt.subplot(2, 2, 3)
    #experiment(iterations=[20, 40, 60, 80, 100])
    #plt.subplot(2, 2, 4)
    #experiment(k=[2]+range(5,51,10))

    plt.show()



if __name__ == '__main__':
    main()
