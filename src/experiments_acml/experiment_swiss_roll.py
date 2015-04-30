import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/easyexplot/src/')
import easyexplot as eep

from envs.env_swiss_roll import EnvSwissRoll

import experiments.experiment_base as eb



def experiment():
    
    plt.figure()
    eep.plot(eb.prediction_error,
             #algorithm=['foreca', 'pfa', 'gpfa-1', 'gpfa-2', 'random'], 
             algorithm=['pfa', 'gpfa-1', 'gpfa-2'], 
             N=2000,#[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], 
             k=3,#range(3, 16),#[3, 5, 10, 15, 20, 30, 50], 
             p=2, 
             K=1, 
             iterations=[5, 10, 20, 30, 50, 100, 200, 300, 400, 500], 
             noisy_dims=400,#[100, 200, 300, 400, 500, 600], 
             neighborhood_graph=True,
             weighted_edges=True, 
             iteration_dim=2, 
             output_dim=2, 
             data='swiss_roll', 
             measure='avg_det_of_cov', 
             repetitions=25, 
             processes=None, 
             argument_order=None, 
             cachedir='/tmp', 
             plot_elapsed_time=False, 
             show_plot=False, 
             save_plot=True)
    #plt.gca().set_yscale('log')
    
    plt.figure()
    eep.plot(eb.prediction_error,
             #algorithm=['foreca', 'pfa', 'gpfa-1', 'gpfa-2', 'random'], 
             algorithm=['pfa', 'gpfa-1', 'gpfa-2'], 
             N=2000,#[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], 
             k=30,#range(3, 16),#[3, 5, 10, 15, 20, 30, 50], 
             p=2, 
             K=1, 
             iterations=[5, 10, 20, 30, 50, 100, 200, 300, 400, 500], 
             noisy_dims=400,#[100, 200, 300, 400, 500, 600], 
             neighborhood_graph=True,
             weighted_edges=True, 
             iteration_dim=2, 
             output_dim=2, 
             data='swiss_roll', 
             measure='avg_det_of_cov', 
             repetitions=25, 
             processes=None, 
             argument_order=None, 
             cachedir='/tmp', 
             plot_elapsed_time=False, 
             show_plot=False, 
             save_plot=True)
    #plt.gca().set_yscale('log')
    
    plt.figure()
    eep.plot(eb.prediction_error,
             #algorithm=['foreca', 'pfa', 'gpfa-1', 'gpfa-2', 'random'], 
             algorithm=['pfa', 'gpfa-1', 'gpfa-2'], 
             N=2000,#[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], 
             k=50,#range(3, 16),#[3, 5, 10, 15, 20, 30, 50], 
             p=2, 
             K=1, 
             iterations=[5, 10, 20, 30, 50, 100, 200, 300, 400, 500], 
             noisy_dims=400,#[100, 200, 300, 400, 500, 600], 
             neighborhood_graph=True,
             weighted_edges=True, 
             iteration_dim=2, 
             output_dim=2, 
             data='swiss_roll', 
             measure='avg_det_of_cov', 
             repetitions=25, 
             processes=None, 
             argument_order=None, 
             cachedir='/tmp', 
             plot_elapsed_time=False, 
             show_plot=False, 
             save_plot=True)
    #plt.gca().set_yscale('log')
    
    plt.figure()
    eep.plot(eb.prediction_error,
             #algorithm=['foreca', 'pfa', 'gpfa-1', 'gpfa-2', 'random'], 
             algorithm=['pfa', 'gpfa-1', 'gpfa-2'], 
             N=2000,#[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], 
             k=[3, 5, 10, 15, 20, 30, 50], 
             p=2, 
             K=1, 
             iterations=500,#[5, 10, 20, 30, 50, 100, 200, 300, 400, 500], 
             noisy_dims=400,#[100, 200, 300, 400, 500, 600], 
             neighborhood_graph=True,
             weighted_edges=True, 
             iteration_dim=2, 
             output_dim=2, 
             data='swiss_roll', 
             measure='avg_det_of_cov', 
             repetitions=25, 
             processes=None, 
             argument_order=None, 
             cachedir='/tmp', 
             plot_elapsed_time=False, 
             show_plot=False, 
             save_plot=True)
    #plt.gca().set_yscale('log')
    
    plt.figure()
    eep.plot(eb.prediction_error,
             #algorithm=['foreca', 'pfa', 'gpfa-1', 'gpfa-2', 'random'], 
             algorithm=['pfa', 'gpfa-1', 'gpfa-2'], 
             N=[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], 
             k=[3, 30, 50],#range(3, 16),#[3, 5, 10, 15, 20, 30, 50], 
             p=2, 
             K=1, 
             iterations=500,#[5, 10, 20, 30, 50, 100, 200, 300, 400, 500], 
             noisy_dims=400,#[100, 200, 300, 400, 500, 600], 
             neighborhood_graph=True,
             weighted_edges=True, 
             iteration_dim=2, 
             output_dim=2, 
             data='swiss_roll', 
             measure='avg_det_of_cov', 
             repetitions=25, 
             processes=None, 
             argument_order=None, 
             cachedir='/tmp', 
             plot_elapsed_time=False, 
             show_plot=False, 
             save_plot=True)
    #plt.gca().set_yscale('log')
    
    plt.show()
    
    
    
def visualize():
    k = 3
    #data_train, data_test = eb.generate_training_data_swiss_roll(N=2000, noisy_dims=400, seed=0)
    env = EnvSwissRoll(seed=1)
    data_train, data_test = env.generate_training_data(num_steps=2000, noisy_dims=400, whitening=True, chunks=2)
    data_test, _, data_test_labels = data_test
    data_train = data_train[0]

    plt.subplot(1, 2, 1)
    data_test_projected = eb.calc_projection_pfa(data_train=data_train, 
                                                 data_test=data_test, 
                                                 p=2, 
                                                 K=1,
                                                 output_dim=2)
    plt.scatter(data_test_projected[1:,0], data_test_projected[1:,1], c=data_test_labels)
    plt.title(eb.calc_error(data=data_test_projected, k=k, measure='avg_det_of_cov'))
    
    plt.subplot(1, 2, 2)
    data_test_projected = eb.calc_projection_gpfa(data_train=data_train, 
                                                  data_test=data_test, 
                                                  k=k, 
                                                  iterations=400, 
                                                  iteration_dim=2, 
                                                  variance_graph=False, 
                                                  neighborhood_graph=True,
                                                  weighted_edges=True,
                                                  output_dim=2)
    plt.scatter(data_test_projected[1:,0], data_test_projected[1:,1], c=data_test_labels)
    plt.title(eb.calc_error(data=data_test_projected, k=k, measure='avg_det_of_cov'))
    
    plt.show()



def main():
    experiment()
    #visualize()



if __name__ == '__main__':
    main()
