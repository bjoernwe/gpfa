import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/easyexplot/src/')
import easyexplot as eep

import experiments.experiment_base as exp



def main():
    """
    Shows a problem with gPFA on the Oscillation data set.
    """

    algorithm = ['gpfa', 'foreca'] 
    N = 2000
    k = 50#[10, 20, 50, 100]
    p = 2
    K = 1
    iterations = 50
    noisy_dims = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50]#, 100, 200, 300, 400, 500]
    variance_graph = False
    neighborhood_graph = True 
    keep_variance = 1.
    iteration_dim = 2
     
    data = 'oscillation'
    measure = 'avg_det_of_cov'
    seed = 0
    repetitions = 20
    processes = None
    argument_order = ['noisy_dims']
    cachedir = '.'
    plot_elapsed_time = False
    show_plot = False
    save_plot = False
    
    eep.plot(exp.prediction_error,
             algorithm=algorithm, 
             N=N, 
             k=k, 
             p=p, 
             K=K, 
             iterations=iterations, 
             noisy_dims=noisy_dims, 
             variance_graph=variance_graph, 
             neighborhood_graph=neighborhood_graph, 
             keep_variance=keep_variance, 
             iteration_dim=iteration_dim, 
             data=data, 
             measure=measure, 
             seed=seed, 
             repetitions=repetitions, 
             processes=processes, 
             argument_order=argument_order, 
             cachedir=cachedir, 
             plot_elapsed_time=plot_elapsed_time, 
             show_plot=show_plot,
             save_plot=save_plot)
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.show()


if __name__ == '__main__':
    main()
