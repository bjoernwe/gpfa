import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/easyexplot/src/')
import easyexplot as eep

import experiments.experiment_base as eb



def main():
    eep.plot(eb.prediction_error,
             algorithm=['foreca', 'pfa', 'gpfa-1', 'gpfa-2', 'lpp', 'random'], 
             N=2000, 
             k=5, #[3, 5, 10, 15, 20, 30, 50], 
             p=2, 
             K=1, 
             iterations=10, 
             noisy_dims=0, 
             neighborhood_graph=True,
             weighted_edges=True, 
             keep_variance=np.arange(.85, .95, .01), 
             iteration_dim=2, 
             data='face', 
             measure='avg_det_of_cov', 
             repetitions=1, 
             processes=None, 
             argument_order=None, 
             cachedir='/tmp', 
             plot_elapsed_time=False, 
             show_plot=False, 
             save_plot=False)
    plt.gca().set_yscale('log')
    plt.show()



if __name__ == '__main__':
    main()
