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
from envs.env_swiss_roll import EnvSwissRoll



def experiment(algorithm, N, k, p, K, iterations, noisy_dims, variance_graph, neighborhood_graph=False, keep_variance=1., iteration_dim=2, data='swiss_roll', measure='var'):
    
    assert algorithm in ['foreca', 'sfa', 'pfa', 'gpfa', 'random']
    assert data in ['face']
    assert measure in ['graph_var', 'det_var', 'graph_star']
    
    # generate data
    if data == 'swiss_roll':
        env = EnvSwissRoll()
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
        return
    
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



def main():
    
    # parameters
    algorithms = ['sfa', 'pfa', 'gpfa', 'random', 'foreca']
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
    repetitions = 2
    save_results = True

    plt.subplot(1, 2, 1)
    plotter.plot(experiment,
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
                 legend=algorithms,
                 save_result=save_results,
                 save_plot=False,
                 show_plot=False)
    plt.ylim(0, 0.05)
    
    plt.subplot(1, 2, 2)
    plotter.plot(experiment,
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
                 legend=algorithms,
                 save_result=save_results,
                 save_plot=True,
                 show_plot=False)
    plt.ylim(0, 0.5)
    
    plt.show()
    return



if __name__ == '__main__':
    main()
