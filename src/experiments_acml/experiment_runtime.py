import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/easyexplot/src/')
import easyexplot as eep

import experiments.experiment_base as eb



def experiment():
    
    repeptitions = 20
    
    plt.figure()
    eep.plot(eb.prediction_error,
             algorithm=['random', 'foreca'],#['foreca', 'pfa', 'gpfa-1', 'gpfa-2'], 
             N=2000,#[600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000], 
             k=40,#[3, 5, 10, 15, 20, 30, 40, 50], 
             p=1, 
             K=1, 
             iterations=300,#[1, 20, 50, 75, 100, 200, 300, 400, 500], 
             noisy_dims=300,#[0, 20, 50, 100, 200, 300],#, 400, 500], 
             neighborhood_graph=False,
             weighted_edges=True, 
             iteration_dim=2, 
             output_dim=2, 
             data='swiss_roll', 
             measure='det_of_avg_cov', 
             repetitions=repeptitions, 
             processes=1, 
             argument_order=None, 
             cachedir='/scratch/weghebvc/time',
             ipython_profile='ssh', 
             plot_elapsed_time=True, 
             show_plot=False, 
             save_plot=False)
    plt.gca().set_yscale('log')
    


def visualize_noisy_dims():

    iter_arg = 'noisy_dims'
    repeptitions = 20
    
    plt.figure()
    result = eep.evaluate(eb.prediction_error,
                          algorithm='foreca', 
                          N=2000, 
                          k=40, 
                          p=1, 
                          K=1, 
                          iterations=300, 
                          noisy_dims=[0, 20, 50, 100, 200, 300], 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='det_of_avg_cov', 
                          repetitions=repeptitions, 
                          processes=3, 
                          cachedir='/scratch/weghebvc/time',
                          ipython_profile='ssh')
#     m = np.mean(values_from_result(result), axis=-1)
#     s = np.std(values_from_result(result), axis=-1)
#     plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color='red', marker=None, linestyle='-')
    eep.plot_result(result, plot_elapsed_time=True, save_plot=False, show_plot=False)
    
    result = eep.evaluate(eb.prediction_error,
                          algorithm=['pfa', 'gpfa-1', 'gpfa-2'], 
                          N=2000, 
                          k=40, 
                          p=1, 
                          K=1, 
                          iterations=300, 
                          noisy_dims=[0, 20, 50, 100, 200, 300], 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='det_of_avg_cov', 
                          repetitions=repeptitions, 
                          processes=None, 
                          cachedir='/scratch/weghebvc/time',
                          ipython_profile='ssh')
    eep.plot_result(result, plot_elapsed_time=True, save_plot=False, show_plot=False)
#     fig = plt.gcf()
#     ax = plt.gca()
#     ax.set_title('')
#     fig.subplots_adjust(left=.125, right=.9, bottom=.1, top=.9)
#     fig.suptitle('')
#     linestyles = ['--', '-', '-']
#     colors = ['red', 'blue', 'blue']
#     markers = [None, 'o', 'o']
#     facecolors = [None, 'blue', 'white']
#     for i, line in enumerate(ax.get_lines()[2::3]):
#         line.set_linestyle(linestyles[i])
#         line.set_color(colors[i])
#         line.set_marker(markers[i])
#         line.set_markerfacecolor(facecolors[i])
    #for i, _ in enumerate(result.iter_args['algorithm']):
    #    m = np.mean(values_from_result(result)[:,i], axis=-1)
    #    s = np.std(values_from_result(result)[:,i], axis=-1)
    #    plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color=colors[i], markerfacecolor=facecolors[i], marker=markers[i], linestyle=linestyles[i], markersize=10)
    #plt.legend(['ForeCA', 'PFA', 'GPFA (1)', 'GPFA (2)'], loc='best') 
    
    plt.xlabel('number of noisy dimensions')
    plt.ylabel('runtime in minutes (log scale)')
    plt.gca().set_yscale('log')
    #plt.show()



def visualize_n():

    iter_arg = 'N'
    repeptitions = 20
    
    plt.figure()
    result = eep.evaluate(eb.prediction_error,
                          algorithm='foreca', 
                          N=[500, 1000, 1500, 2000], 
                          k=40,
                          p=1, 
                          K=1, 
                          iterations=300, 
                          noisy_dims=300, 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='det_of_avg_cov', 
                          repetitions=repeptitions, 
                          processes=3, 
                          cachedir='/scratch/weghebvc/time',
                          ipython_profile='ssh')
#     m = np.mean(values_from_result(result), axis=-1)
#     s = np.std(values_from_result(result), axis=-1)
#     plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color='red', marker=None, linestyle='-')
    eep.plot_result(result, plot_elapsed_time=True, save_plot=False, show_plot=False)
    
    result = eep.evaluate(eb.prediction_error,
                          algorithm=['pfa', 'gpfa-1', 'gpfa-2'], 
                          N=[500, 1000, 1500, 2000], 
                          k=40,
                          p=1, 
                          K=1, 
                          iterations=300, 
                          noisy_dims=300, 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='det_of_avg_cov', 
                          repetitions=repeptitions, 
                          processes=None, 
                          cachedir='/scratch/weghebvc/time',
                          ipython_profile='ssh')
    eep.plot_result(result, plot_elapsed_time=True, save_plot=False, show_plot=False)
#     linestyles = ['--', '-', '-']
#     colors = ['red', 'blue', 'blue']
#     markers = [None, 'o', 'o']
#     facecolors = [None, 'blue', 'white']
#     for i, _ in enumerate(result.iter_args['algorithm']):
#         m = np.mean(values_from_result(result)[:,i], axis=-1)
#         s = np.std(values_from_result(result)[:,i], axis=-1)
#         print i
#         print result.iter_args
#         print result.values.shape
#         plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color=colors[i], markerfacecolor=facecolors[i], marker=markers[i], linestyle=linestyles[i], markersize=10)
#     plt.legend(['ForeCA', 'PFA', 'GPFA (1)', 'GPFA (2)'], loc='best') 
#     
#     plt.xlabel('number of samples for training')
#     plt.ylabel('runtime in minutes (log scale)')
    plt.gca().set_yscale('log')
    #plt.show()



def visualize_iterations():

    iter_arg = 'iterations'
    repeptitions = 20
    
    plt.figure()
    result = eep.evaluate(eb.prediction_error,
                          algorithm='foreca', 
                          N=2000, 
                          k=40,
                          p=1, 
                          K=1, 
                          iterations=[1, 20, 50, 75, 100, 200, 300], 
                          noisy_dims=300, 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='det_of_avg_cov', 
                          repetitions=repeptitions, 
                          processes=3, 
                          cachedir='/scratch/weghebvc/time',
                          ipython_profile='ssh')
#     m = np.mean(values_from_result(result), axis=-1)
#     s = np.std(values_from_result(result), axis=-1)
#     plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color='red', marker=None, linestyle='-')
    eep.plot_result(result, plot_elapsed_time=True, save_plot=False, show_plot=False)
    
    result = eep.evaluate(eb.prediction_error,
                          algorithm=['pfa', 'gpfa-1', 'gpfa-2'], 
                          N=2000, 
                          k=40,
                          p=1, 
                          K=1, 
                          iterations=[1, 20, 50, 75, 100, 200, 300], 
                          noisy_dims=300, 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='det_of_avg_cov', 
                          repetitions=repeptitions, 
                          processes=None, 
                          cachedir='/scratch/weghebvc/time',
                          ipython_profile='ssh')
    eep.plot_result(result, plot_elapsed_time=True, save_plot=False, show_plot=False)
#     linestyles = ['--', '-', '-']
#     colors = ['red', 'blue', 'blue']
#     markers = [None, 'o', 'o']
#     facecolors = [None, 'blue', 'white']
#     for i, _ in enumerate(result.iter_args['algorithm']):
#         m = np.mean(values_from_result(result)[:,i], axis=-1)
#         s = np.std(values_from_result(result)[:,i], axis=-1)
#         plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color=colors[i], markerfacecolor=facecolors[i], marker=markers[i], linestyle=linestyles[i], markersize=10)
#     plt.legend(['ForeCA', 'PFA', 'GPFA (1)', 'GPFA (2)'], loc='best') 
#     
#     plt.xlabel('iterations')
#     plt.ylabel('runtime in minutes (log scale)')
    plt.gca().set_yscale('log')
    #plt.show()



def visualize_k():

    iter_arg = 'k'
    repeptitions = 20
    
    plt.figure()
    result = eep.evaluate(eb.prediction_error,
                          algorithm='foreca', 
                          N=2000, 
                          k=[3, 5, 10, 15, 20, 30, 40, 50],
                          p=1, 
                          K=1, 
                          iterations=300, 
                          noisy_dims=300, 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='det_of_avg_cov', 
                          repetitions=repeptitions, 
                          processes=3, 
                          cachedir='/scratch/weghebvc/time',
                          ipython_profile='ssh')
#     m = np.mean(values_from_result(result), axis=-1)
#     s = np.std(values_from_result(result), axis=-1)
#     plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color='red', marker=None, linestyle='-')
    eep.plot_result(result, plot_elapsed_time=True, save_plot=False, show_plot=False)
    
    result = eep.evaluate(eb.prediction_error,
                          algorithm=['pfa', 'gpfa-1', 'gpfa-2'], 
                          N=2000, 
                          k=[3, 5, 10, 15, 20, 30, 40, 50],
                          p=1, 
                          K=1, 
                          iterations=300, 
                          noisy_dims=300, 
                          neighborhood_graph=False,
                          weighted_edges=True, 
                          iteration_dim=2, 
                          output_dim=2, 
                          data='swiss_roll', 
                          measure='det_of_avg_cov', 
                          repetitions=repeptitions, 
                          processes=None, 
                          cachedir='/scratch/weghebvc/time',
                          ipython_profile='ssh')
    eep.plot_result(result, plot_elapsed_time=True, save_plot=False, show_plot=False)
#     linestyles = ['--', '-', '-']
#     colors = ['red', 'blue', 'blue']
#     markers = [None, 'o', 'o']
#     facecolors = [None, 'blue', 'white']
#     for i, _ in enumerate(result.iter_args['algorithm']):
#         m = np.mean(values_from_result(result)[:,i], axis=-1)
#         s = np.std(values_from_result(result)[:,i], axis=-1)
#         plt.errorbar(x=result.iter_args[iter_arg], y=m, yerr=s, linewidth=1.5, color=colors[i], markerfacecolor=facecolors[i], marker=markers[i], linestyle=linestyles[i], markersize=10)
#     plt.legend(['ForeCA', 'PFA', 'GPFA (1)', 'GPFA (2)'], loc='best') 
#     
#     plt.xlabel('k')
#     plt.ylabel('runtime in minutes (log scale)')
    plt.gca().set_yscale('log')
    #plt.show()



def main():
    #experiment()
    visualize_noisy_dims()
    visualize_n()
    visualize_iterations()
    visualize_k()
    plt.show()



if __name__ == '__main__':
    main()
