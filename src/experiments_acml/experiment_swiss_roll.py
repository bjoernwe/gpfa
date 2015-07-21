import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/easyexplot/src/')
import easyexplot as eep

import experiments.experiment_base as eb



def experiment():
    
    repeptitions = 30
    
    plt.figure()
    eep.plot(eb.prediction_error,
             algorithm=['random', 'foreca'], 
             N=2000,#[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], 
             k=40,#[3, 5, 10, 15, 20, 30, 40, 50], 
             p=1, 
             K=1, 
             iterations=300,#[1, 20, 50, 75, 100, 200, 300, 400, 500], 
             noisy_dims=200,# 400, 500],#, 600], 
             neighborhood_graph=False,
             weighted_edges=True, 
             iteration_dim=2, 
             output_dim=2, 
             data='swiss_roll', 
             measure='trace_of_avg_cov', 
             repetitions=repeptitions, 
             processes=1, 
             argument_order=None, 
             cachedir='/scratch/weghebvc',
             ipython_profile='ssh', 
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
#              measure='trace_of_avg_cov', 
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
#              measure='trace_of_avg_cov', 
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
#              measure='trace_of_avg_cov', 
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
#              measure='trace_of_avg_cov', 
#              repetitions=repeptitions, 
#              processes=None, 
#              argument_order=None, 
#              cachedir='/scratch/weghebvc', 
#              plot_elapsed_time=False, 
#              show_plot=False, 
#              save_plot=True)
#     plt.gca().set_yscale('log')
    
    plt.show()
    
    
    
def visualize_noisy_dims(repetitions=1, ipython_profile=None, include_foreca=True):

    iter_arg = 'noisy_dims'

    plt.figure()
    result = eep.evaluate(eb.prediction_error,
                          algorithm='random', 
                          N=2000,#[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], 
                          k=40,#[3, 5, 10, 15, 20, 30, 40, 50], 
                          p=1, 
                          K=1, 
                          iterations=50,#[1, 20, 50, 75, 100, 200, 300, 400, 500], 
                          noisy_dims=[0, 20, 50, 100, 200, 300, 400, 500],#, 600], 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='trace_of_avg_cov', 
                          repetitions=repetitions, 
                          processes=None, 
                          cachedir='/scratch/weghebvc',
                          ipython_profile=ipython_profile)
    m = np.mean(result.values, axis=-1)
    s = np.std(result.values, axis=-1)
    plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color='green', marker=None, linestyle=':')
    
    if include_foreca:
        result = eep.evaluate(eb.prediction_error,
                              algorithm='foreca', 
                              N=2000,#[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], 
                              k=40,#[3, 5, 10, 15, 20, 30, 40, 50], 
                              p=1, 
                              K=1, 
                              iterations=50,#[1, 20, 50, 75, 100, 200, 300, 400, 500], 
                              noisy_dims=[0, 20, 50, 100, 200],#, 300],# 400, 500],#, 600], 
                              neighborhood_graph=False,
                              weighted_edges=True, 
                              iteration_dim=2, 
                              output_dim=2, 
                              data='swiss_roll', 
                              measure='trace_of_avg_cov', 
                              repetitions=repetitions, 
                              processes=2, 
                              cachedir='/scratch/weghebvc',
                              ipython_profile=ipython_profile)
        m = np.mean(result.values, axis=-1)
        s = np.std(result.values, axis=-1)
        plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color='red', marker=None, linestyle='-')
     
    result = eep.evaluate(eb.prediction_error,
                          algorithm=['pfa', 'gpfa-1', 'gpfa-2'], 
                          N=2000,#[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], 
                          k=40,#[3, 5, 10, 15, 20, 30, 40, 50], 
                          p=1, 
                          K=1, 
                          iterations=50,#[1, 20, 50, 75, 100, 200, 300, 400, 500], 
                          noisy_dims=[0, 20, 50, 100, 200, 300, 400, 500],#, 600], 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='trace_of_avg_cov', 
                          repetitions=repetitions, 
                          processes=None, 
                          cachedir='/scratch/weghebvc',
                          ipython_profile=ipython_profile)
    linestyles = ['--', '-', '-']
    colors = ['red', 'blue', 'blue']
    markers = [None, 'o', 'o']
    facecolors = [None, 'blue', 'white']
    for i, _ in enumerate(result.iter_args['algorithm']):
        m = np.mean(result.values[:,i], axis=-1)
        s = np.std(result.values[:,i], axis=-1)
        plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color=colors[i], markerfacecolor=facecolors[i], marker=markers[i], linestyle=linestyles[i], markersize=10)
    plt.legend(['random', 'ForeCA', 'PFA', 'GPFA (1)', 'GPFA (2)'], loc='best') 
    
    plt.xlabel('number of noisy dimensions')
    plt.ylabel('prediction error (log scale)')
    #plt.gca().set_yscale('log')
    #plt.show()



def visualize_n(repetitions=1, ipython_profile=None, include_foreca=True):

    iter_arg = 'N'
    
    plt.figure()
    result = eep.evaluate(eb.prediction_error,
                          algorithm='random', 
                          N=[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200], 
                          k=40, 
                          p=1, 
                          K=1, 
                          iterations=50, 
                          noisy_dims=200, 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='trace_of_avg_cov', 
                          repetitions=repetitions, 
                          processes=None, 
                          cachedir='/scratch/weghebvc',
                          ipython_profile=ipython_profile)
    m = np.mean(result.values, axis=-1)
    s = np.std(result.values, axis=-1)
    plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color='green', marker=None, linestyle=':')
    
    if include_foreca:
        result = eep.evaluate(eb.prediction_error,
                              algorithm='foreca', 
                              N=[600, 800, 1000, 1200, 1400, 1600, 1800, 2000], 
                              k=40,
                              p=1, 
                              K=1, 
                              iterations=50, 
                              noisy_dims=200, 
                              neighborhood_graph=False,
                              weighted_edges=True, 
                              iteration_dim=2, 
                              output_dim=2, 
                              data='swiss_roll', 
                              measure='trace_of_avg_cov', 
                              repetitions=repetitions, 
                              processes=2, 
                              cachedir='/scratch/weghebvc',
                              ipython_profile=ipython_profile)
        m = np.mean(result.values, axis=-1)
        s = np.std(result.values, axis=-1)
        plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color='red', marker=None, linestyle='-')
    
    result = eep.evaluate(eb.prediction_error,
                          algorithm=['pfa', 'gpfa-1', 'gpfa-2'], 
                          N=[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200], 
                          k=40,
                          p=1, 
                          K=1, 
                          iterations=50, 
                          noisy_dims=200, 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='trace_of_avg_cov', 
                          repetitions=repetitions, 
                          processes=None, 
                          cachedir='/scratch/weghebvc',
                          ipython_profile=ipython_profile)
    linestyles = ['--', '-', '-']
    colors = ['red', 'blue', 'blue']
    markers = [None, 'o', 'o']
    facecolors = [None, 'blue', 'white']
    for i, _ in enumerate(result.iter_args['algorithm']):
        m = np.mean(result.values[i], axis=-1)
        s = np.std(result.values[i], axis=-1)
        plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color=colors[i], markerfacecolor=facecolors[i], marker=markers[i], linestyle=linestyles[i], markersize=10)
    plt.legend(['random', 'ForeCA', 'PFA', 'GPFA (1)', 'GPFA (2)'], loc='best') 
    
    plt.xlabel('number of samples for training')
    plt.ylabel('prediction error (log scale)')
    #plt.gca().set_yscale('log')
    #plt.show()



def visualize_iterations(repetitions=1, ipython_profile=None, include_foreca=True):

    iter_arg = 'iterations'
    
    plt.figure()
    result = eep.evaluate(eb.prediction_error,
                          algorithm='random', 
                          N=2000, 
                          k=40, 
                          p=1, 
                          K=1, 
                          iterations=[1, 10, 20, 30, 40, 50],
                          noisy_dims=200, 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='trace_of_avg_cov', 
                          repetitions=repetitions, 
                          processes=None, 
                          cachedir='/scratch/weghebvc',
                          ipython_profile=ipython_profile)
    m = np.mean(result.values, axis=-1)
    s = np.std(result.values, axis=-1)
    plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color='green', marker=None, linestyle=':')

    if include_foreca:    
        result = eep.evaluate(eb.prediction_error,
                              algorithm='foreca', 
                              N=2000, 
                              k=40,
                              p=1, 
                              K=1, 
                              iterations=[1, 10, 20, 30, 40, 50],
                              noisy_dims=200, 
                              neighborhood_graph=False,
                              weighted_edges=True, 
                              iteration_dim=2, 
                              output_dim=2, 
                              data='swiss_roll', 
                              measure='trace_of_avg_cov', 
                              repetitions=repetitions, 
                              processes=2, 
                              cachedir='/scratch/weghebvc',
                              ipython_profile=ipython_profile)
        m = np.mean(result.values, axis=-1)
        s = np.std(result.values, axis=-1)
        plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color='red', marker=None, linestyle='-')
    
    result = eep.evaluate(eb.prediction_error,
                          algorithm=['pfa', 'gpfa-1', 'gpfa-2'], 
                          N=2000, 
                          k=40,
                          p=1, 
                          K=1, 
                          iterations=[1, 10, 20, 30, 40, 50],
                          noisy_dims=200, 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='trace_of_avg_cov', 
                          repetitions=repetitions, 
                          processes=None, 
                          cachedir='/scratch/weghebvc',
                          ipython_profile=ipython_profile)
    linestyles = ['--', '-', '-']
    colors = ['red', 'blue', 'blue']
    markers = [None, 'o', 'o']
    facecolors = [None, 'blue', 'white']
    for i, _ in enumerate(result.iter_args['algorithm']):
        m = np.mean(result.values[i], axis=-1)
        s = np.std(result.values[i], axis=-1)
        plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color=colors[i], markerfacecolor=facecolors[i], marker=markers[i], linestyle=linestyles[i], markersize=10)
    plt.legend(['random', 'ForeCA', 'PFA', 'GPFA (1)', 'GPFA (2)'], loc='best') 
    
    plt.xlabel(iter_arg)
    plt.ylabel('prediction error (log scale)')
    #plt.gca().set_yscale('log')
    #plt.show()



def visualize_k(repetitions=1, ipython_profile=None, include_foreca=True):

    iter_arg = 'k'
    
    plt.figure()
    result = eep.evaluate(eb.prediction_error,
                          algorithm='random', 
                          N=2000, 
                          k=[3, 5, 10, 15, 20, 30, 40, 50], 
                          p=1, 
                          K=1, 
                          iterations=50,
                          noisy_dims=200, 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='trace_of_avg_cov', 
                          repetitions=repetitions, 
                          processes=None, 
                          cachedir='/scratch/weghebvc',
                          ipython_profile=ipython_profile)
    m = np.mean(result.values, axis=-1)
    s = np.std(result.values, axis=-1)
    plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color='green', marker=None, linestyle=':')
    
    if include_foreca:
        result = eep.evaluate(eb.prediction_error,
                              algorithm='foreca', 
                              N=2000, 
                              k=[3, 5, 10, 15, 20, 30, 40, 50],
                              p=1, 
                              K=1, 
                              iterations=50,
                              noisy_dims=200, 
                              neighborhood_graph=False,
                              weighted_edges=True, 
                              iteration_dim=2, 
                              output_dim=2, 
                              data='swiss_roll', 
                              measure='trace_of_avg_cov', 
                              repetitions=repetitions, 
                              processes=2, 
                              cachedir='/scratch/weghebvc',
                              ipython_profile=ipython_profile)
        m = np.mean(result.values, axis=-1)
        s = np.std(result.values, axis=-1)
        plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color='red', marker=None, linestyle='-')
     
    result = eep.evaluate(eb.prediction_error,
                          algorithm=['pfa', 'gpfa-1', 'gpfa-2'], 
                          N=2000, 
                          k=[3, 5, 10, 15, 20, 30, 40, 50],
                          p=1, 
                          K=1, 
                          iterations=50,
                          noisy_dims=200, 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='trace_of_avg_cov', 
                          repetitions=repetitions, 
                          processes=None, 
                          cachedir='/scratch/weghebvc',
                          ipython_profile=ipython_profile)
    linestyles = ['--', '-', '-']
    colors = ['red', 'blue', 'blue']
    markers = [None, 'o', 'o']
    facecolors = [None, 'blue', 'white']
    for i, _ in enumerate(result.iter_args['algorithm']):
        m = np.mean(result.values[:,i], axis=-1)
        s = np.std(result.values[:,i], axis=-1)
        plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color=colors[i], markerfacecolor=facecolors[i], marker=markers[i], linestyle=linestyles[i], markersize=10)
    plt.legend(['random', 'ForeCA', 'PFA', 'GPFA (1)', 'GPFA (2)'], loc='best') 
    
    plt.xlabel(iter_arg)
    plt.ylabel('prediction error (log scale)')
    #plt.gca().set_yscale('log')
    #plt.show()



def plot_experiment(k=40, repetitions=5, ipython_profile=None, include_foreca=True):

    plt.figure()
    result = eep.evaluate(eb.prediction_error,
                          algorithm='random', 
                          N=2000, 
                          k=[3, 5, 10, 15, 20, 30, 40, 50], 
                          p=1, 
                          K=1, 
                          iterations=50,
                          noisy_dims=200, 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='trace_of_avg_cov', 
                          repetitions=repetitions, 
                          processes=None, 
                          cachedir='/scratch/weghebvc',
                          ipython_profile=ipython_profile)
    m = np.mean(result.values, axis=-1)
    s = np.std(result.values, axis=-1)
    plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color='green', marker=None, linestyle=':')
    
    if include_foreca:
        result = eep.evaluate(eb.prediction_error,
                              algorithm='foreca', 
                              N=2000, 
                              k=[3, 5, 10, 15, 20, 30, 40, 50],
                              p=1, 
                              K=1, 
                              iterations=50,
                              noisy_dims=200, 
                              neighborhood_graph=False,
                              weighted_edges=True, 
                              iteration_dim=2, 
                              output_dim=2, 
                              data='swiss_roll', 
                              measure='trace_of_avg_cov', 
                              repetitions=repetitions, 
                              processes=2, 
                              cachedir='/scratch/weghebvc',
                              ipython_profile=ipython_profile)
        m = np.mean(result.values, axis=-1)
        s = np.std(result.values, axis=-1)
        plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color='red', marker=None, linestyle='-')
     
    result = eep.evaluate(eb.prediction_error,
                          algorithm=['pfa', 'gpfa-1', 'gpfa-2'], 
                          N=2000, 
                          k=[3, 5, 10, 15, 20, 30, 40, 50],
                          p=1, 
                          K=1, 
                          iterations=50,
                          noisy_dims=200, 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='trace_of_avg_cov', 
                          repetitions=repetitions, 
                          processes=None, 
                          cachedir='/scratch/weghebvc',
                          ipython_profile=ipython_profile)
    linestyles = ['--', '-', '-']
    colors = ['red', 'blue', 'blue']
    markers = [None, 'o', 'o']
    facecolors = [None, 'blue', 'white']
    for i, _ in enumerate(result.iter_args['algorithm']):
        m = np.mean(result.values[:,i], axis=-1)
        s = np.std(result.values[:,i], axis=-1)
        plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color=colors[i], markerfacecolor=facecolors[i], marker=markers[i], linestyle=linestyles[i], markersize=10)
    plt.legend(['random', 'ForeCA', 'PFA', 'GPFA (1)', 'GPFA (2)'], loc='best') 
    
    plt.xlabel(iter_arg)
    plt.ylabel('prediction error (log scale)')
    #plt.gca().set_yscale('log')
    #plt.show()



def main():
    repetitions = 20
    ipython_profile = 'ssh'
    include_foreca = False
    #experiment()
    visualize_noisy_dims(repetitions=repetitions, ipython_profile=ipython_profile, include_foreca=include_foreca)
    visualize_n(repetitions=repetitions, ipython_profile=ipython_profile, include_foreca=include_foreca)
    visualize_iterations(repetitions=repetitions, ipython_profile=ipython_profile, include_foreca=include_foreca)
    visualize_k(repetitions=repetitions, ipython_profile=ipython_profile, include_foreca=include_foreca)
    plt.show()



if __name__ == '__main__':
    main()
