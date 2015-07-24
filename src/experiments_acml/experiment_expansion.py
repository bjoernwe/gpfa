import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/easyexplot/src/')
import easyexplot as eep

import experiments.experiment_base as eb



def experiment():
    
    repeptitions = 5
    
    plt.figure()
    eep.plot(eb.prediction_error,
             algorithm=['random', 'pfa', 'gpfa-1', 'gpfa-2'], 
             N=2000,#[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], 
             k=40,#[3, 5, 10, 15, 20, 30, 40, 50], 
             p=1, 
             K=4,#[1,2,4], 
             iterations=50,#[1, 20, 50, 75, 100, 200, 300, 400, 500], 
             noisy_dims=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45],# 30, 40, 50, 60],#[1, 10, 20, 30],#, 40, 50],#, 100, 200, 300],# 400, 500],#, 600], 
             neighborhood_graph=False,
             weighted_edges=True, 
             iteration_dim=2, 
             output_dim=2,
             expansion=2, 
             keep_variance=1.,
             data='swiss_roll_squared_noise', 
             measure='trace_of_avg_cov', 
             repetitions=repeptitions, 
             processes=None, 
             argument_order=None, 
             cachedir='/scratch/weghebvc',
             #ipython_profile='ssh', 
             plot_elapsed_time=False, 
             show_plot=False, 
             save_plot=False)
    #plt.gca().set_yscale('log')



def main():
    experiment()
    plt.show()



if __name__ == '__main__':
    main()
