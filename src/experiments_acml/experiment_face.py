import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/easyexplot/src/')
import easyexplot as eep

import experiments.experiment_base as eb



def main():
    eep.plot(eb.prediction_error,
             algorithm=['sfa', 'foreca', 'pfa', 'gpfa-1', 'gpfa-2', 'lpp'],#, 'random'], 
             N=2000, 
             k=[3, 5, 10, 15, 20, 30, 40, 50], 
             p=1, 
             K=1, 
             iterations=100,#[100, 200, 300, 400], 
             noisy_dims=0, 
             neighborhood_graph=True,
             weighted_edges=True, 
             keep_variance=.9,#np.arange(.85, .96, .01), 
             iteration_dim=1, 
             output_dim=1, 
             data='face', 
             measure='det_of_avg_cov', 
             repetitions=1, 
             processes=None, 
             argument_order=None, 
             cachedir='/scratch/weghebvc', 
             plot_elapsed_time=False, 
             show_plot=False, 
             save_plot=False)
    #plt.gca().set_yscale('log')
    plt.show()



if __name__ == '__main__':
    main()
