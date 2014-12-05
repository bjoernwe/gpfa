import numpy as np
import sys

from matplotlib import pyplot as plt

import mdp

sys.path.append('/home/weghebvc/workspace/PFAStefan/src/')
import PFANodeMDP

import foreca_node
import gpfa
import plotter

sys.path.append('/home/weghebvc/workspace/Worldmodel/src/')
from envs.env_face import EnvFace
from envs.env_oscillator import EnvOscillator
from envs.env_swiss_roll import EnvSwissRoll



def experiment(algorithm, N, k, p, K, iterations, noisy_dims, variance_graph, neighborhood_graph=False, keep_variance=1., iteration_dim=2, data='swiss_roll', measure='var'):
    
    assert algorithm in ['foreca', 'sfa', 'pfa', 'gpfa', 'random']
    assert data in ['oscillator', 'swiss_roll', 'face']
    assert measure in ['graph_var', 'det_var', 'graph_star']
    
    # generate data
    if data == 'swiss_roll':
        env = EnvSwissRoll()
        data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=True, chunks=2)
        data_train = data_train[0]
        data_test = data_test[0]
    elif data == 'oscillator':
        env = EnvOscillator()
        data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=True, chunks=2)
        data_train = data_train[0]
        data_test = data_test[0]
    elif data == 'face':
        env = EnvFace()
        data_train, data_test = env.generate_training_data(num_steps=[1500, 465], noisy_dims=noisy_dims, whitening=False, chunks=2)
        pca = mdp.nodes.PCANode(output_dim=keep_variance)
        pca.train(data_train[0])
        data_train = pca.execute(data_train[0])
        data_test = pca.execute(data_test[0])
        whitening = mdp.nodes.WhiteningNode()
        whitening.train(data_train)
        data_train = whitening.execute(data_train)
        data_test = whitening.execute(data_test)
    else:
        assert False
    
    # train algorithm
    if algorithm == 'foreca':
        model = foreca_node.ForeCA(output_dim=2)
    elif algorithm == 'sfa':
        model = mdp.nodes.SFANode(output_dim=2)
    elif algorithm == 'pfa':
        model = PFANodeMDP.PFANode(p=p, k=K, affine=False, output_dim=2)
    elif algorithm == 'gpfa':
        model = gpfa.gPFA(k=k, 
                          output_dim=2, 
                          iterations=iterations, 
                          iteration_dim=iteration_dim, 
                          variance_graph=variance_graph,
                          neighborhood_graph=neighborhood_graph)
    elif algorithm == 'random':
        model = gpfa.RandomProjection(output_dim=2)
    else:
        return
    
    model.train(data_train)
    
    # evaluate solution
    result = model.execute(data_test)
    if measure == 'det_var':
        return gpfa.calc_predictability_det_var(result, k)
    elif measure == 'graph_var':
        return gpfa.calc_predictability_graph_var(result, k)
    elif measure == 'graph_star':
        return gpfa.calc_predictability_graph_star(result, k)
    else:
        assert False



def calc_baseline(N, k, data='swiss_roll', measure='var'):
    
    assert data in ['oscillator', 'swiss_roll']
    assert measure in ['graph_var', 'det_var', 'graph_star']
    
    # generate data
    if data == 'swiss_roll':
        env = EnvSwissRoll()
        data, _, _ = env.generate_training_data(num_steps=N, noisy_dims=0, whitening=True, chunks=1)[0]
    elif data == 'oscillator':
        env = EnvOscillator()
        data, _, _ = env.generate_training_data(num_steps=N, noisy_dims=0, whitening=True, chunks=1)[0]
    else:
        assert False
    
    # evaluate solution
    if measure == 'det_var':
        return gpfa.calc_predictability_det_var(data, k)
    elif measure == 'graph_var':
        return gpfa.calc_predictability_graph_var(data, k)
    elif measure == 'graph_star':
        return gpfa.calc_predictability_graph_star(data, k)
    else:
        assert False



def main():
    
    # parameters
    algorithms = ['random', 'foreca', 'pfa', 'gpfa']
    p = 2
    K = 8
    k = 100 # [2, 3, 5, 10, 20, 30, 40, 50, 100, 200]
    N = 2000 #[1000, 2000, 3000, 4000, 5000] 1965
    noisy_dims = [1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500] #[0, 50, 100, 150, 200, 250, 300, 350, 400]
    keep_variance = 1. #[.98, .95, .90, .80]
    iterations = 100 # [1, 10, 20, 30, 40, 50, 100]
    iteration_dim = 2 # [2, 5, 10, 20, 50, 100, 200]
    neighborhood_graph=True
    data = 'swiss_roll'
    measure = 'det_var'
    
    # plotter arguments
    processes = None
    repetitions = 50
    save_results = False

    # plot results from algorithms
    for i, a in enumerate(algorithms):
        is_last_iteration = (i==len(algorithms)-1)
        result = plotter.plot(experiment,
                              algorithm=a,
                              k=k,
                              N=N,
                              p=p,
                              K=K,
                              iterations=iterations,
                              noisy_dims=noisy_dims[:6] if a == 'foreca' else noisy_dims,
                              keep_variance=keep_variance,
                              iteration_dim=iteration_dim,
                              variance_graph=False,
                              neighborhood_graph=neighborhood_graph,
                              data=data,
                              processes=processes,
                              repetitions=repetitions,
                              measure='det_var',
                              save_result=save_results,
                              save_plot=False,#is_last_iteration,
                              show_plot=False)#is_last_iteration)
  
    # plot a baseline
    result_baseline = plotter.evaluate(calc_baseline, N=2000, k=k, data=data, measure=measure, repetitions=repetitions)
    baseline = np.mean(result_baseline.values, axis=1)
    plt.plot([1, noisy_dims[-1]], [baseline, baseline], '--', color='black')

    # show plot
    plt.legend(algorithms + ['baseline'], loc='best')
    plt.gca().set_xscale('log')
    plt.savefig('plotter_results/%s.png' % result.result_prefix)
    plt.show()

    return



if __name__ == '__main__':
    main()
