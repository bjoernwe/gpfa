import numpy as np
import sys

from matplotlib import pyplot as plt
from matplotlib.colors import ColorConverter

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

import experiments_proxy.experiment_base as eb

#sys.path.append('/home/weghebvc/workspace/git/environments_new/src/')
#from envs.env_data import EnvData
#from envs.env_data2d import EnvData2D
from envs.env_kai import EnvKai



def _get_values(result, plot_time=False):
    if plot_time:
        return result.elapsed_times / 1000.
    return result.values


def plot_experiment(env, dataset, n_train, n_test, k, k_eval, p, K, noisy_dims, 
                    pca, iterations, output_dim, repetitions, include_random, 
                    include_sfa, include_sffa, include_foreca, include_pfa, 
                    include_gfa1, include_gfa2, measure, cachedir, processes,
                    window=None, pca_after_expansion=1., additive_noise=0, 
                    causal_features=True, generalized_eigen_problem=True, 
                    use_test_set=True, x_offset=0, y_label=True, legend=True, 
                    plot_time=False, whitening=True, manage_seed='external', 
                    legend_loc='best', seed=0):
    
    results = {}
    
    if plot_time:
        eb.set_cachedir(cachedir=None)
    else:
        eb.set_cachedir(cachedir=cachedir)
    
    legends = []
    cc = ColorConverter()
    
    ecolor_alpha = .5
    
    result = ep.evaluate(eb.prediction_error,
                         algorithm=eb.Algorithms.Random, 
                         n_train=n_train,
                         n_test=n_test, 
                         k=k,
                         k_eval=k_eval, 
                         p=p, 
                         #K=K, 
                         whitening=whitening,
                         seed=seed,
                         #iterations=iterations,
                         noisy_dims=noisy_dims,
                         additive_noise=additive_noise,
                         pca=pca, 
                         pca_after_expansion=pca_after_expansion,
                         #neighborhood_graph=False,
                         #weighted_edges=True,
                         #generalized_eigen_problem=generalized_eigen_problem, 
                         output_dim=output_dim,
                         use_test_set=use_test_set, 
                         env=env,
                         dataset=dataset, 
                         window=window,
                         measure=measure, 
                         repetitions=repetitions, 
                         processes=processes,
                         manage_seed=manage_seed, 
                         cachedir=cachedir,
                         ignore_arguments=['window'])
    results[eb.Algorithms.Random] = result
 
    # determine iter_arg
    iter_arg = result.iter_args.keys()[0]
    
    if include_random:
        values = _get_values(result, plot_time=plot_time)
        m = np.mean(values, axis=-1)
        s = np.std(values, axis=-1)
        x = np.array(result.iter_args[iter_arg]) - 2 * x_offset
        x = x+2 if env is EnvKai and type(noisy_dims) == list else x
        ecolor = tuple(1-((1-c)*.25) for c in list(cc.to_rgb('green')))
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='green', ecolor=ecolor, marker=None, linestyle=':')
        legends.append('Random')
        
    if include_sfa:
        result = ep.evaluate(eb.prediction_error,
                             algorithm=eb.Algorithms.SFA, 
                             n_train=n_train,
                             n_test=n_test, 
                             k=k,
                             k_eval=k_eval, 
                             p=p, 
                             #K=K, 
                             whitening=whitening,
                             seed=seed,
                             #iterations=iterations,
                             noisy_dims=noisy_dims,
                             additive_noise=additive_noise,
                             pca=pca, 
                             pca_after_expansion=pca_after_expansion,
                             #neighborhood_graph=False,
                             #weighted_edges=True,
                             #generalized_eigen_problem=generalized_eigen_problem, 
                             output_dim=output_dim, 
                             use_test_set=use_test_set,
                             env=env, 
                             dataset=dataset,
                             window=window,
                             measure=measure, 
                             repetitions=repetitions, 
                             processes=processes, 
                             manage_seed=manage_seed,
                             cachedir=cachedir,
                             ignore_arguments=['window'])
        results[eb.Algorithms.SFA] = result
        values = _get_values(result, plot_time=plot_time)
        m = np.mean(values, axis=-1)
        s = np.std(values, axis=-1)
        x = np.array(result.iter_args[iter_arg]) - x_offset
        x = x+2 if env is EnvKai and type(noisy_dims) == list else x
        ecolor = tuple(1-((1-c)*.25) for c in list(cc.to_rgb('green')))
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='green', ecolor=ecolor, marker=None, linestyle='-')
        legends.append('SFA')
    else:
        plt.errorbar(x=1, y=0, linewidth=1.2, elinewidth=.5, color='green', ecolor=cc.to_rgba('green', alpha=ecolor_alpha), marker=None, linestyle='-')
        legends.append('SFA')
    
    if include_sffa:
        result = ep.evaluate(eb.prediction_error,
                             algorithm=eb.Algorithms.SFFA, 
                             n_train=n_train,
                             n_test=n_test,
                             k=k,
                             k_eval=k_eval, 
                             p=p, 
                             #K=K, 
                             whitening=whitening,
                             seed=seed,
                             #iterations=iterations,
                             noisy_dims=noisy_dims,
                             additive_noise=additive_noise,
                             pca=pca, 
                             pca_after_expansion=pca_after_expansion,
                             #neighborhood_graph=False,
                             #weighted_edges=True, 
                             #generalized_eigen_problem=generalized_eigen_problem,
                             output_dim=output_dim, 
                             use_test_set=use_test_set,
                             env=env, 
                             dataset=dataset,
                             window=window,
                             measure=measure, 
                             repetitions=repetitions, 
                             processes=processes, 
                             manage_seed=manage_seed,
                             cachedir=cachedir,
                             ignore_arguments=['window'])
        results[eb.Algorithms.SFFA] = result
        values = _get_values(result, plot_time=plot_time)
        m = np.mean(values, axis=-1)
        s = np.std(values, axis=-1)
        x = np.array(result.iter_args[iter_arg]) - x_offset
        x = x+2 if env is EnvKai and type(noisy_dims) == list else x
        ecolor = tuple(1-((1-c)*.25) for c in list(cc.to_rgb('green')))
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='green', ecolor=ecolor, marker='^', linestyle='-')
        legends.append('SFFA')
#    else:
#        plt.errorbar(x=1, y=0, linewidth=1.2, elinewidth=.5, color='green', ecolor=cc.to_rgba('green', alpha=ecolor_alpha), marker=None, linestyle='-')
#        legends.append('SFFA')
    
    if include_foreca:
        n_train_foreca = n_train
        if type(n_train) is list:
            n_train_foreca = [n for n in n_train if n <= 800]
        noisy_dims_foreca = noisy_dims
        if type(noisy_dims) is list:
            noisy_dims_foreca = [d for d in noisy_dims if d <= 30]
        pca_foreca = pca
        if type(pca) is list:
            pca_foreca = [v for v in pca if v <= .92]
        pca_after_expansion_foreca = pca_after_expansion
        if type(pca_after_expansion) is list:
            pca_after_expansion_foreca = [v for v in pca_after_expansion if v <= .92]
        result = ep.evaluate(eb.prediction_error,
                             algorithm=eb.Algorithms.ForeCA, 
                             n_train=n_train_foreca,
                             n_test=n_test, 
                             k=k,
                             k_eval=k_eval,
                             p=p, 
                             #K=K, 
                             whitening=whitening,
                             seed=seed,
                             #iterations=iterations,
                             noisy_dims=noisy_dims_foreca,
                             additive_noise=additive_noise,
                             pca=pca_foreca,
                             pca_after_expansion=pca_after_expansion_foreca, 
                             #neighborhood_graph=False,
                             #weighted_edges=True, 
                             #generalized_eigen_problem=generalized_eigen_problem,
                             output_dim=output_dim, 
                             use_test_set=use_test_set, 
                             env=env,
                             dataset=dataset,
                             window=window,
                             measure=measure, 
                             repetitions=repetitions, 
                             processes=processes,# if processes else 16, 
                             manage_seed=manage_seed,
                             cachedir=cachedir,
                             ignore_arguments=['window'])
        results[eb.Algorithms.ForeCA] = result
        values = _get_values(result, plot_time=plot_time)
        m = np.mean(values, axis=-1)
        s = np.std(values, axis=-1)
        x = np.array(result.iter_args[iter_arg]) - x_offset
        x = x+2 if env is EnvKai and type(noisy_dims) == list else x
        ecolor = tuple(1-((1-c)*.25) for c in list(cc.to_rgb('red')))
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='red', ecolor=ecolor, marker=None, linestyle='-.')
        legends.append('ForeCA')
#    else:
#        plt.errorbar(x=1, y=0, linewidth=1.2, elinewidth=.5, color='red', ecolor=cc.to_rgba('red', alpha=ecolor_alpha), marker=None, linestyle='-.')
#        legends.append('ForeCA')

    if include_pfa:
        result = ep.evaluate(eb.prediction_error,
                             algorithm=eb.Algorithms.PFA, 
                             n_train=n_train,
                             n_test=n_test,
                             k=k,
                             k_eval=k_eval,
                             p=p, 
                             K=K,
                             whitening=whitening,
                             seed=seed, 
                             #iterations=iterations,
                             noisy_dims=noisy_dims,
                             additive_noise=additive_noise,
                             pca=pca, 
                             pca_after_expansion=pca_after_expansion,
                             neighborhood_graph=False,
                             weighted_edges=True, 
                             #generalized_eigen_problem=generalized_eigen_problem,
                             output_dim=output_dim, 
                             env=env,
                             dataset=dataset,
                             window=window,
                             use_test_set=use_test_set,
                             measure=measure, 
                             repetitions=repetitions, 
                             processes=processes,
                             manage_seed=manage_seed,
                             argument_order=['algorithm'], 
                             cachedir=cachedir,
                             ignore_arguments=['window'])
        results[eb.Algorithms.PFA] = result
        linestyles = ['--']
        #colors = ['red']
        markers = [None]
        facecolors = [None]
        #for _, _ in enumerate(result.iter_args['algorithm']):
        values = _get_values(result, plot_time=plot_time)
        m = np.mean(values, axis=-1)
        s = np.std(values, axis=-1)
        x = np.array(result.iter_args[iter_arg]) + 0 * x_offset
        x = x+2 if env is EnvKai and type(noisy_dims) == list else x
        ecolor = tuple(1-((1-c)*.25) for c in list(cc.to_rgb('red')))
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='red', ecolor=ecolor, markerfacecolor=facecolors[0], marker=markers[0], linestyle=linestyles[0], markersize=10)
        legends += ['PFA']

    if include_gfa1:
        k_gpfa1 = k
        if type(k_gpfa1) is list:
            k_gpfa1 = [ik for ik in k if ik <= 20]
        result = ep.evaluate(eb.prediction_error,
                             algorithm=eb.Algorithms.GPFA1, 
                             n_train=n_train,
                             n_test=n_test,
                             k=k_gpfa1,
                             k_eval=k_eval,
                             p=p, 
                             #K=K,
                             whitening=whitening,
                             seed=seed, 
                             iterations=iterations,
                             noisy_dims=noisy_dims,
                             additive_noise=additive_noise,
                             pca=pca,
                             pca_after_expansion=pca_after_expansion,
                             causal_features=causal_features, 
                             neighborhood_graph=False,
                             weighted_edges=True, 
                             generalized_eigen_problem=generalized_eigen_problem,
                             output_dim=output_dim, 
                             env=env,
                             dataset=dataset,
                             window=window,
                             measure=measure, 
                             use_test_set=use_test_set,
                             repetitions=repetitions, 
                             processes=processes,
                             manage_seed=manage_seed,
                             argument_order=['algorithm'], 
                             cachedir=cachedir,
                             ignore_arguments=['window'])
        results[eb.Algorithms.GPFA1] = result
        linestyles = ['-']
        #colors = ['blue']
        markers = ['^']
        facecolors = ['blue']
        #for _, _ in enumerate(result.iter_args['algorithm']):
        values = _get_values(result, plot_time=plot_time)
        m = np.mean(values, axis=-1)
        s = np.std(values, axis=-1)
        x = np.array(result.iter_args[iter_arg]) + x_offset
        x = x+2 if env is EnvKai and type(noisy_dims) == list else x
        ecolor = tuple(1-((1-c)*.25) for c in list(cc.to_rgb('blue')))
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='blue', ecolor=ecolor, markerfacecolor=facecolors[0], marker=markers[0], linestyle=linestyles[0], markersize=7)
        legends += ['GPFA (1)']

    if include_gfa2:
        result = ep.evaluate(eb.prediction_error,
                             algorithm=eb.Algorithms.GPFA2, 
                             n_train=n_train,
                             n_test=n_test,
                             k=k,
                             k_eval=k_eval,
                             p=p, 
                             #K=K,
                             whitening=whitening,
                             seed=seed, 
                             iterations=iterations,
                             noisy_dims=noisy_dims,
                             additive_noise=additive_noise,
                             pca=pca,
                             pca_after_expansion=pca_after_expansion,
                             causal_features=causal_features, 
                             neighborhood_graph=False,
                             weighted_edges=True, 
                             generalized_eigen_problem=generalized_eigen_problem,
                             output_dim=output_dim,
                             env=env, 
                             dataset=dataset,
                             window=window,
                             measure=measure, 
                             use_test_set=use_test_set,
                             repetitions=repetitions, 
                             processes=processes,
                             manage_seed=manage_seed,
                             argument_order=['algorithm'], 
                             cachedir=cachedir,
                             ignore_arguments=['window'])
        results[eb.Algorithms.GPFA2] = result
        linestyles = ['-']
        #colors = ['blue']
        markers = ['^']
        facecolors = ['white']
        #for i, _ in enumerate(result.iter_args['algorithm']):
        values = _get_values(result, plot_time=plot_time)
        m = np.mean(values, axis=-1)
        s = np.std(values, axis=-1)
        x = np.array(result.iter_args[iter_arg]) + 2 * x_offset
        x = x+2 if env is EnvKai and type(noisy_dims) == list else x
        ecolor = tuple(1-((1-c)*.25) for c in list(cc.to_rgb('blue')))
        plt.errorbar(x=x, y=m, yerr=s, linewidth=1.2, elinewidth=.5, color='blue', ecolor=ecolor, markerfacecolor=facecolors[0], marker=markers[0], linestyle=linestyles[0], markersize=7)
        legends += ['GPFA (2)']

    if legend:
        plt.legend(legends, loc=legend_loc, prop={'size':12})
        #plt.legend(legends, loc='upper center', prop={'size':12})
        #plt.legend(legends, loc='lower center', prop={'size':12})
        #plt.legend(legends, loc='center', prop={'size':12})

    x_label = iter_arg
    x_label = x_label if x_label != 'keep_variance' else 'variance preserved'   
    x_label = x_label if x_label != 'iterations' else 'R'   
    x_label = x_label if x_label != 'N' else 'S_train'   
    x_label = x_label if x_label != 'k_eval' else 'q'   
    x_label = x_label if x_label != 'output_dim' else 'M'   
    x_label = x_label if x_label != 'noisy_dims' else 'N'   
    plt.xlabel(x_label)
    #plt.gca().ticklabel_format(axis='x', style='sci', scilimits=(-3,3))
    
    if plot_time:
        plt.gca().set_yscale('log')
    
    if y_label:
        if plot_time:
            plt.ylabel('elapsed time in sec. (log-scale)')
        else:
            plt.ylabel('prediction error')
            
    return results



if __name__ == '__main__':
    pass
