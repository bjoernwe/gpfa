from matplotlib import pyplot as plt

import example_2014_11_plotter
import plotter



def main():
    
    # parameters
    p = 2
    K = 8
    k = 100
    noisy_dims = 0
    keep_variance = .9
    iterations = 60
    iteration_dim = 2
    neighborhood_graph=True
    measure = 'det_var'
    data = 'face'
    
    # plotter arguments
    processes = None
    repetitions = 50
    save_results = True

    plt.subplot(1, 2, 1)
    plotter.plot(example_2014_11_plotter.experiment,
                 algorithm=['pfa', 'gpfa'],
                 k=k,
                 N=None,
                 p=p,
                 K=K,
                 iterations=iterations,
                 noisy_dims=noisy_dims,
                 iteration_dim=iteration_dim,
                 variance_graph=False,
                 keep_variance=keep_variance,
                 neighborhood_graph=neighborhood_graph,
                 data=data,
                 processes=processes,
                 repetitions=1,
                 measure=measure,
                 save_result=save_results,
                 save_plot=False,
                 show_plot=False)
    plt.ylim(0, 0.05)
    
    plt.subplot(1, 2, 2)
    plotter.plot(example_2014_11_plotter.experiment,
                 algorithm=['foreca', 'random'],
                 k=k,
                 N=None,
                 p=p,
                 K=K,
                 iterations=iterations,
                 noisy_dims=noisy_dims,
                 iteration_dim=iteration_dim,
                 variance_graph=False,
                 keep_variance=keep_variance,
                 neighborhood_graph=neighborhood_graph,
                 data=data,
                 processes=processes,
                 repetitions=repetitions,
                 measure=measure,
                 save_result=save_results,
                 save_plot=True,
                 show_plot=False)
    plt.ylim(0, 0.5)
    
    plt.show()
    return



if __name__ == '__main__':
    main()
