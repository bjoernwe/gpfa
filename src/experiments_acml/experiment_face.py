import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/easyexplot/src/')
import easyexplot as eep

import experiments.experiment_base as eb



def experiment():
    
    eep.plot(eb.prediction_error,
             algorithm='pfa',#['foreca', 'pfa', 'gpfa-1', 'gpfa-2'],#, 'random'], 
             N=2000, 
             k=40, 
             p=1, 
             K=1, 
             iterations=100, 
             noisy_dims=0, 
             neighborhood_graph=False,
             weighted_edges=True, 
             keep_variance=np.arange(.85, .99, .01), 
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
    
    
    
def visualize_keep_variance():

    iter_arg = 'keep_variance'
    
    plt.figure()
    result = eep.evaluate(eb.prediction_error,
                          algorithm='random', 
                          N=2000, 
                          k=40, 
                          p=1, 
                          K=1, 
                          iterations=300, 
                          noisy_dims=0,
                          keep_variance=np.arange(.85, 1., .01),
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=1, 
                          output_dim=1, 
                          data='face', 
                          measure='det_of_avg_cov', 
                          processes=None, 
                          cachedir='/scratch/weghebvc')
    m = np.mean(result.values, axis=-1)
    s = np.std(result.values, axis=-1)
    variances = np.array(result.iter_args[iter_arg]) * 100
    plt.errorbar(x=variances, y=m, yerr=s, linewidth=1.5, color='green', marker=None, linestyle=':')
    
    result = eep.evaluate(eb.prediction_error,
                          algorithm='foreca', 
                          N=2000, 
                          k=40, 
                          p=1, 
                          K=1, 
                          iterations=300, 
                          noisy_dims=0,
                          keep_variance=np.arange(.85, .99, .01),
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=1, 
                          output_dim=1, 
                          data='face', 
                          measure='det_of_avg_cov', 
                          processes=4)
    m = np.mean(result.values, axis=-1)
    s = np.std(result.values, axis=-1)
    variances = np.array(result.iter_args[iter_arg]) * 100
    plt.errorbar(x=variances, y=m, yerr=s, linewidth=1.5, color='red', marker=None, linestyle='-')
    
    result = eep.evaluate(eb.prediction_error,
                          algorithm=['pfa', 'gpfa-1', 'gpfa-2'], 
                          N=2000, 
                          k=40, 
                          p=1, 
                          K=1, 
                          iterations=300, 
                          noisy_dims=0,
                          keep_variance=np.arange(.85, 1., .01),
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=1, 
                          output_dim=1, 
                          data='face', 
                          measure='det_of_avg_cov', 
                          processes=None, 
                          cachedir='/scratch/weghebvc')
    variances = np.array(result.iter_args[iter_arg]) * 100
    linestyles = ['--', '-', '-']
    colors = ['red', 'blue', 'blue']
    markers = [None, 'o', 'o']
    facecolors = [None, 'blue', 'white']
    for i, _ in enumerate(result.iter_args['algorithm']):
        m = np.mean(result.values[:,i], axis=-1)
        s = np.std(result.values[:,i], axis=-1)
        plt.errorbar(x=variances, y=m, yerr=s, linewidth=1.5, color=colors[i], markerfacecolor=facecolors[i], marker=markers[i], linestyle=linestyles[i], markersize=10)
    plt.legend(['random', 'ForeCA', 'PFA', 'GPFA (1)', 'GPFA (2)']) 
    
    plt.xlabel('% of variance kept')
    plt.ylabel('prediction error (log scale)')
    plt.gca().set_yscale('log')
    plt.show()



def visualize_k():

    iter_arg = 'k'
    
    plt.figure()
    result = eep.evaluate(eb.prediction_error,
                          algorithm='random', 
                          N=2000, 
                          k=[3, 5, 10, 15, 20, 30, 40, 50], 
                          p=1, 
                          K=1, 
                          iterations=300,
                          noisy_dims=0,
                          keep_variance=.95, 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=1, 
                          output_dim=1, 
                          data='face', 
                          measure='det_of_avg_cov', 
                          processes=None, 
                          cachedir='/scratch/weghebvc')
    m = np.mean(result.values, axis=-1)
    s = np.std(result.values, axis=-1)
    plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color='green', marker=None, linestyle=':')
    
    result = eep.evaluate(eb.prediction_error,
                          algorithm='foreca', 
                          N=2000, 
                          k=[3, 5, 10, 15, 20, 30, 40, 50],
                          p=1, 
                          K=1, 
                          iterations=300,
                          noisy_dims=0, 
                          keep_variance=.95, 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=1, 
                          output_dim=1, 
                          data='face', 
                          measure='det_of_avg_cov', 
                          processes=2, 
                          cachedir='/scratch/weghebvc')
    m = np.mean(result.values, axis=-1)
    s = np.std(result.values, axis=-1)
    plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color='red', marker=None, linestyle='-')
    
    result = eep.evaluate(eb.prediction_error,
                          algorithm=['pfa', 'gpfa-1', 'gpfa-2'], 
                          N=2000, 
                          k=[3, 5, 10, 15, 20, 30, 40, 50],
                          p=1, 
                          K=1, 
                          iterations=300,
                          noisy_dims=0, 
                          keep_variance=.95, 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=1, 
                          output_dim=1, 
                          data='face', 
                          measure='det_of_avg_cov', 
                          processes=None, 
                          cachedir='/scratch/weghebvc')
    linestyles = ['--', '-', '-']
    colors = ['red', 'blue', 'blue']
    markers = [None, 'o', 'o']
    facecolors = [None, 'blue', 'white']
    for i, _ in enumerate(result.iter_args['algorithm']):
        m = np.mean(result.values[:,i], axis=-1)
        s = np.std(result.values[:,i], axis=-1)
        plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color=colors[i], markerfacecolor=facecolors[i], marker=markers[i], linestyle=linestyles[i], markersize=10)
    plt.legend(['random', 'ForeCA', 'PFA', 'GPFA (1)', 'GPFA (2)']) 
    
    plt.xlabel(iter_arg)
    plt.ylabel('prediction error (log scale)')
    plt.gca().set_yscale('log')
    plt.show()



def main():
    experiment()
    #visualize_keep_variance()
    #visualize_k()



if __name__ == '__main__':
    main()
