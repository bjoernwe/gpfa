import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/easyexplot/src/')
import easyexplot as eep

import experiments.experiment_base as eb



def experiment(N=2000, keep_variance=1., k=20, iterations=50, data='mario_window'):
    
    #plt.figure()
    eep.plot(eb.prediction_error,
             algorithm=['pfa', 'gcfa-1', 'gcfa-2'], 
             N=N, 
             keep_variance=keep_variance,
             k=k,
             p=1, 
             K=1, 
             iterations=iterations,
             noisy_dims=0, 
             neighborhood_graph=False,
             weighted_edges=True, 
             iteration_dim=1, 
             output_dim=1, 
             data=data,
             measure='trace_of_avg_cov', 
             reverse_error=False,
             repetitions=1, 
             processes=None,
             argument_order=['algorithm'], 
             cachedir=None,#'/scratch/weghebvc',
             plot_elapsed_time=False, 
             show_plot=False, 
             save_plot=True)
    #plt.gca().set_yscale('log')
    #plt.show()
    
    
    
def main():
    
    # mario
    plt.figure()
    plt.subplot(1, 2, 1)
    experiment(data='mario_window', keep_variance=list(np.arange(.85, 1., .05)))
    #experiment(data='mario_window', N=[1000, 1500, 2000, 2500])
    plt.subplot(1, 2, 2)
    experiment(data='mario_window', k=range(5,51,5))

    # face
    #plt.figure()
    #plt.subplot(1, 2, 1)
    #experiment(data='face', keep_variance=list(np.arange(.85, 1., .01)))
    #plt.subplot(1, 2, 2)
    #experiment(data='face', keep_variance=.95, k=[1, 2, 5, 10, 20, 30, 40, 50])
    
    plt.show()



if __name__ == '__main__':
    main()
