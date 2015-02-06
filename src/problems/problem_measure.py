import easyexplot as eep

import experiments.experiment_base as exp



def main():
    """
    Shows how the objective function grows because the whitening is calculated
    on the training data, not on the test data.
    """

    algorithm = 'random' 
    N = 2000
    k = 100
    p = 2
    K = 1
    iterations = 50
    noisy_dims = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 200, 300, 400, 500]
    variance_graph = False
    neighborhood_graph = True 
    keep_variance = 1.
    iteration_dim = 2
     
    data = 'random'
    measure = 'avg_det_of_cov'
    seed = 0
    repetitions = 10 
    processes = None
    argument_order = None 
    non_numeric_args = None
    cachedir = None
    plot_elapsed_time = False
    show_plot = True
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
             non_numeric_args=non_numeric_args, 
             cachedir=cachedir, 
             plot_elapsed_time=plot_elapsed_time, 
             show_plot=show_plot,
             save_plot=save_plot)



if __name__ == '__main__':
    main()
