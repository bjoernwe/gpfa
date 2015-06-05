import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/easyexplot/src/')
import easyexplot as eep

sys.path.append('/home/weghebvc/workspace/Worldmodel/src/')
from envs.env_swiss_roll import EnvSwissRoll

import experiments.experiment_base as eb



def experiment():
    
    repeptitions = 20
    
    plt.figure()
    eep.plot(eb.prediction_error,
             algorithm='foreca', 
             N=2000,#[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], 
             k=40,#[3, 5, 10, 15, 20, 30, 40, 50], 
             p=1, 
             K=1, 
             iterations=300,#[1, 20, 50, 75, 100, 200, 300, 400, 500], 
             noisy_dims=[0, 10, 20, 50, 100, 200, 300],# 400, 500],#, 600], 
             neighborhood_graph=False,
             weighted_edges=True, 
             iteration_dim=2, 
             output_dim=2, 
             data='swiss_roll', 
             measure='det_of_avg_cov', 
             repetitions=repeptitions, 
             processes=4, 
             argument_order=None, 
             cachedir='/scratch/weghebvc', 
             plot_elapsed_time=False, 
             show_plot=False, 
             save_plot=True)
    plt.gca().set_yscale('log')

#     plt.figure()
#     eep.plot(eb.prediction_error,
#              algorithm=['pfa', 'gpfa-1', 'gpfa-2', 'random'], 
#              N=2000,#[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], 
#              k=40,#[3, 5, 10, 15, 20, 30, 40, 50], 
#              p=1, 
#              K=1, 
#              iterations=300,#[1, 20, 50, 75, 100, 200, 300, 400, 500], 
#              noisy_dims=[0, 10, 20, 50, 100, 200, 300, 400, 500],#, 600], 
#              neighborhood_graph=False,
#              weighted_edges=True, 
#              iteration_dim=2, 
#              output_dim=2, 
#              data='swiss_roll', 
#              measure='det_of_avg_cov', 
#              repetitions=repeptitions, 
#              processes=None, 
#              argument_order=None, 
#              cachedir='/scratch/weghebvc', 
#              plot_elapsed_time=False, 
#              show_plot=False, 
#              save_plot=True)
#     plt.gca().set_yscale('log')
#      
#     plt.figure()
#     eep.plot(eb.prediction_error,
#              algorithm=['pfa', 'gpfa-1', 'gpfa-2', 'random'], 
#              N=[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], 
#              k=40,#[3, 5, 10, 15, 20, 30, 40, 50], 
#              p=1, 
#              K=1, 
#              iterations=300,#[1, 20, 50, 75, 100, 200, 300, 400, 500], 
#              noisy_dims=300,#[0, 100, 200, 300, 400, 500],#, 600], 
#              neighborhood_graph=False,
#              weighted_edges=True, 
#              iteration_dim=2, 
#              output_dim=2, 
#              data='swiss_roll', 
#              measure='det_of_avg_cov', 
#              repetitions=repeptitions, 
#              processes=None, 
#              argument_order=None, 
#              cachedir='/scratch/weghebvc', 
#              plot_elapsed_time=False, 
#              show_plot=False, 
#              save_plot=True)
#     plt.gca().set_yscale('log')
#      
#     plt.figure()
#     eep.plot(eb.prediction_error,
#              algorithm=['pfa', 'gpfa-1', 'gpfa-2', 'random'], 
#              N=2000,#[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], 
#              k=40,#[3, 5, 10, 15, 20, 30, 40, 50], 
#              p=1, 
#              K=1, 
#              iterations=[1, 20, 50, 75, 100, 200, 300],#, 400, 500], 
#              noisy_dims=300,#[0, 100, 200, 300, 400, 500],#, 600], 
#              neighborhood_graph=False,
#              weighted_edges=True, 
#              iteration_dim=2, 
#              output_dim=2, 
#              data='swiss_roll', 
#              measure='det_of_avg_cov', 
#              repetitions=repeptitions, 
#              processes=None, 
#              argument_order=None, 
#              cachedir='/scratch/weghebvc', 
#              plot_elapsed_time=False, 
#              show_plot=False, 
#              save_plot=True)
#     plt.gca().set_yscale('log')
#  
#     plt.figure()
#     eep.plot(eb.prediction_error,
#              algorithm=['pfa', 'gpfa-1', 'gpfa-2', 'random'], 
#              N=2000,#[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], 
#              k=[3, 5, 10, 15, 20, 30, 40, 50], 
#              p=1, 
#              K=1, 
#              iterations=300,#[1, 20, 50, 75, 100, 200, 300, 400, 500], 
#              noisy_dims=300,#[0, 100, 200, 300, 400, 500],#, 600], 
#              neighborhood_graph=False,
#              weighted_edges=True, 
#              iteration_dim=2, 
#              output_dim=2, 
#              data='swiss_roll', 
#              measure='det_of_avg_cov', 
#              repetitions=repeptitions, 
#              processes=None, 
#              argument_order=None, 
#              cachedir='/scratch/weghebvc', 
#              plot_elapsed_time=False, 
#              show_plot=False, 
#              save_plot=True)
#     plt.gca().set_yscale('log')
    
    plt.show()
    
    
    
def visualize():

    repeptitions = 20
    
    plt.figure()
    result = eep.evaluate(eb.prediction_error,
                          algorithm='random', 
                          N=2000,#[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], 
                          k=40,#[3, 5, 10, 15, 20, 30, 40, 50], 
                          p=1, 
                          K=1, 
                          iterations=300,#[1, 20, 50, 75, 100, 200, 300, 400, 500], 
                          noisy_dims=[0, 20, 50, 100, 200, 300],# 400, 500],#, 600], 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='det_of_avg_cov', 
                          repetitions=repeptitions, 
                          processes=4, 
                          cachedir='/scratch/weghebvc')
    m = np.mean(result.values, axis=-1)
    s = np.std(result.values, axis=-1)
    plt.errorbar(x=result.iter_args['noisy_dims'], y=m, yerr=s, linewidth=1.5, color='green', marker=None, linestyle=':')
    
    result = eep.evaluate(eb.prediction_error,
                          algorithm='foreca', 
                          N=2000,#[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], 
                          k=40,#[3, 5, 10, 15, 20, 30, 40, 50], 
                          p=1, 
                          K=1, 
                          iterations=300,#[1, 20, 50, 75, 100, 200, 300, 400, 500], 
                          noisy_dims=[0, 20, 50, 100, 200, 300],# 400, 500],#, 600], 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='det_of_avg_cov', 
                          repetitions=repeptitions, 
                          processes=4, 
                          cachedir='/scratch/weghebvc')
    m = np.mean(result.values, axis=-1)
    s = np.std(result.values, axis=-1)
    plt.errorbar(x=result.iter_args['noisy_dims'], y=m, yerr=s, linewidth=1.5, color='red', marker=None, linestyle='-')
    
    result = eep.evaluate(eb.prediction_error,
                          algorithm=['pfa', 'gpfa-1', 'gpfa-2'], 
                          N=2000,#[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], 
                          k=40,#[3, 5, 10, 15, 20, 30, 40, 50], 
                          p=1, 
                          K=1, 
                          iterations=300,#[1, 20, 50, 75, 100, 200, 300, 400, 500], 
                          noisy_dims=[0, 20, 50, 100, 200, 300, 400, 500],#, 600], 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='det_of_avg_cov', 
                          repetitions=repeptitions, 
                          processes=4, 
                          cachedir='/scratch/weghebvc')
    linestyles = ['--', '-', '-']
    colors = ['red', 'blue', 'blue']
    markers = [None, 'o', 'o']
    facecolors = [None, 'blue', 'white']
    for i, _ in enumerate(result.iter_args['algorithm']):
        m = np.mean(result.values[:,i], axis=-1)
        s = np.std(result.values[:,i], axis=-1)
        plt.errorbar(x=result.iter_args['noisy_dims'], y=m, yerr=s, linewidth=1.5, color=colors[i], markerfacecolor=facecolors[i], marker=markers[i], linestyle=linestyles[i], markersize=10)
    plt.legend(['random', 'ForeCA', 'PFA', 'GPFA (1)', 'GPFA (2)']) 
    
#     eep.evaluate(eb.prediction_error,
#                  algorithm='foreca', 
#                  N=2000,#[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], 
#                  k=40,#[3, 5, 10, 15, 20, 30, 40, 50], 
#                  p=2, 
#                  K=1, 
#                  iterations=300,#[1, 20, 50, 75, 100, 200, 300, 400, 500], 
#                  noisy_dims=[0, 10, 20, 50, 100, 200, 300],# 400, 500],#, 600], 
#                  neighborhood_graph=False,
#                  weighted_edges=True, 
#                  iteration_dim=2, 
#                  output_dim=2, 
#                  data='swiss_roll', 
#                  measure='det_of_avg_cov', 
#                  repetitions=repeptitions, 
#                  processes=4, 
#                  cachedir='/scratch/weghebvc')
    plt.gca().set_yscale('log')
    plt.show()



def main():
    #experiment()
    visualize()



if __name__ == '__main__':
    main()
