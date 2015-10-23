import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/easyexplot/src/')
import easyexplot as eep

import experiments.experiment_base as eb



def experiment(N=2000, k=40, iterations=50, noisy_dims=400, data='swiss_roll'):
    
    repeptitions = 5
    
    plt.figure()
    eep.plot(eb.prediction_error,
             algorithm=['pfa', 'gcfa-1', 'gcfa-2'], 
             N=N, 
             k=k, 
             p=1, 
             K=0, 
             iterations=iterations, 
             noisy_dims=noisy_dims, 
             neighborhood_graph=False,
             weighted_edges=True, 
             iteration_dim=2, 
             output_dim=2, 
             data=data, 
             measure='trace_of_avg_cov', 
             repetitions=repeptitions, 
             processes=None, 
             argument_order=None, 
             cachedir=None,
             plot_elapsed_time=False, 
             show_plot=False, 
             save_plot=True)
    #plt.gca().set_yscale('log')
    #plt.show()
    
    

def main():
    
    # swiss roll
    plt.figure()
    plt.subplot(2, 2, 1)
    experiment(data='swiss_roll', noisy_dims=[0, 50, 100, 200, 300, 400, 500])
    plt.subplot(2, 2, 2)
    experiment(data='swiss_roll', N=[500, 1000, 1500, 2000])
    plt.subplot(2, 2, 3)
    experiment(data='swiss_roll', iterations=[1, 10, 30, 50])
    plt.subplot(2, 2, 4)
    experiment(data='swiss_roll', k=[1, 2, 5, 10, 15, 20, 30, 40, 50])

    # dead corners
    plt.figure()
    plt.subplot(2, 2, 1)
    experiment(data='dead_corners', noisy_dims=[0, 50, 100, 200, 300, 400, 500])
    plt.subplot(2, 2, 2)
    experiment(data='dead_corners', N=[500, 1000, 1500, 2000])
    plt.subplot(2, 2, 3)
    experiment(data='dead_corners', iterations=[1, 10, 30, 50])
    plt.subplot(2, 2, 4)
    experiment(data='dead_corners', k=[1, 2, 5, 10, 15, 20, 30, 40, 50])

    plt.show()



if __name__ == '__main__':
    main()
