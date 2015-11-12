import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

import experiments.experiment_base as eb



def experiment(N=2500, k=30, iterations=50, noisy_dims=40, data='ladder'):
    
    repeptitions = 20
    
    #plt.figure()
    ep.plot(eb.prediction_error,
            algorithm=['random', 'pfa', 'gpfa-1', 'gpfa-2', 'gcfa-1', 'gcfa-2'], 
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
    
    

def main():
    
    # ladder
    plt.figure()
    plt.subplot(2, 2, 1)
    experiment(noisy_dims=[0, 10, 20, 30, 40, 50])
    plt.subplot(2, 2, 2)
    experiment(N=[1500, 2000, 2500])
    plt.subplot(2, 2, 3)
    experiment(iterations=[1, 10, 30, 50, 100])
    plt.subplot(2, 2, 4)
    experiment(k=[1, 2, 5, 10, 15, 20, 30, 40, 50])

    plt.show()



if __name__ == '__main__':
    main()
