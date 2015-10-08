import joblib
import numpy as np
import sys

from matplotlib import pyplot as plt

import mdp

sys.path.append('/home/weghebvc/workspace/git/GNUPFA/src/')
import PFANodeMDP

sys.path.append('/home/weghebvc/workspace/git/easyexplot/src/')
import easyexplot as eep

sys.path.append('/home/weghebvc/workspace/git/Environments/src/')
from envs.env_event import EnvEvent
from envs.env_face import EnvFace
from envs.env_kai import EnvKai
from envs.env_ladder import EnvLadder
from envs.env_oscillator import EnvOscillator
from envs.env_random import EnvRandom
from envs.env_ribbon import EnvRibbon
from envs.env_swiss_roll import EnvSwissRoll

import foreca.foreca_node as foreca_node
import gpfa
import NonlinearNoiseNode


# prepare joblib.Memory
cachedir = '/scratch/weghebvc'
mem = joblib.Memory(cachedir=cachedir, verbose=1)



def generate_training_data(N, noisy_dims=0, expansion=1, keep_variance=1., event_prob=.1, num_states=10, max_steps=4, data='swiss_roll', seed=None, repetition_index=None):
    
    assert data in ['random', 'oscillation', 'swiss_roll', 'face', 'event', 'ladder', 'ribbon', 'swiss_roll_squared_noise', 'kai']
    
    # generate data
    if data == 'random':
        data_train, data_test = generate_training_data_random(N=N, 
                                                              noisy_dims=noisy_dims, 
                                                              seed=seed, 
                                                              repetition_index=repetition_index)
    elif data == 'swiss_roll':
        data_train, data_test = generate_training_data_swiss_roll(N=N, 
                                                                  noisy_dims=noisy_dims, 
                                                                  seed=seed, 
                                                                  repetition_index=repetition_index)
    elif data == 'oscillation':
        data_train, data_test = generate_training_data_oscillation(N=N, 
                                                                   noisy_dims=noisy_dims, 
                                                                   seed=seed, 
                                                                   repetition_index=repetition_index)
    elif data == 'face':
        data_train, data_test = generate_training_data_face(N=N, 
                                                            noisy_dims=noisy_dims)
    elif data == 'event':
        data_train, data_test = generate_training_data_event(N=N,
                                                             noisy_dims=noisy_dims,
                                                             seed=seed,
                                                             prob=event_prob, 
                                                             repetition_index=repetition_index)
    elif data == 'ribbon':
        data_train, data_test = generate_training_data_ribbon(N=N,
                                                              noisy_dims=noisy_dims,
                                                              seed=seed,
                                                              repetition_index=repetition_index)
    elif data == 'ladder':
        data_train, data_test = generate_training_data_ladder(N=N,
                                                              noisy_dims=noisy_dims,
                                                              seed=seed,
                                                              num_states=num_states,
                                                              max_steps=max_steps, 
                                                              repetition_index=repetition_index)
    elif data == 'swiss_roll_squared_noise':
        data_train, data_test = generate_training_data_swiss_roll(N=N, 
                                                                  noisy_dims=noisy_dims, 
                                                                  seed=seed, 
                                                                  repetition_index=repetition_index)
        noise_node = NonlinearNoiseNode.NonlinearNoiseNode(dims_modified=2, seed=seed)
        data_train = noise_node.execute(data_train)
        data_test = noise_node.execute(data_test)
    elif data == 'kai':
        data_train, data_test = generate_training_data_kai(N=N, 
                                                           noisy_dims=noisy_dims, 
                                                           seed=seed, 
                                                           repetition_index=repetition_index)
    else:
        assert False

    # expansion        
    if expansion > 1:
        ex = mdp.nodes.PolynomialExpansionNode(degree=expansion)
        data_train = ex.execute(data_train)
        data_test = ex.execute(data_test)
        
    # PCA
    if keep_variance < 1.:
        pca = mdp.nodes.PCANode(output_dim=keep_variance)
        pca.train(data_train)
        data_train = pca.execute(data_train)
        data_test = pca.execute(data_test)
        
    # whitening
    whitening = mdp.nodes.WhiteningNode(reduce=True)
    whitening.train(data_train)
    data_train = whitening.execute(data_train)
    data_test = whitening.execute(data_test)
    
    return data_train, data_test



@mem.cache
def generate_training_data_random(N, noisy_dims, seed=None, repetition_index=None):
    unique_seed = abs(hash(joblib.hash((N, noisy_dims, seed, repetition_index))))
    env = EnvRandom(ndim=2, seed=unique_seed)
    #data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=True, chunks=2)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test



@mem.cache
def generate_training_data_swiss_roll(N, noisy_dims, seed=None, repetition_index=None):
    unique_seed = abs(hash(joblib.hash((N, noisy_dims, seed, repetition_index))))
    env = EnvSwissRoll(seed=unique_seed)
    #data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=True, chunks=2)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test



@mem.cache
def generate_training_data_ribbon(N, noisy_dims, sigma_noise=.05, seed=None, repetition_index=None):
    unique_seed = abs(hash(joblib.hash((N, noisy_dims, sigma_noise, seed, repetition_index))))
    env = EnvRibbon(seed=unique_seed, step_size=.1, sigma_noise=sigma_noise)
    #data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=True, chunks=2)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test



@mem.cache
def generate_training_data_oscillation(N, noisy_dims, seed=None, repetition_index=None):
    unique_seed = abs(hash(joblib.hash((N, noisy_dims, seed, repetition_index))))
    env = EnvOscillator(transition_prob=.9, seed=unique_seed)
    #data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=True, chunks=2)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test



@mem.cache
def generate_training_data_face(N, noisy_dims):
    env = EnvFace()
    data_train, data_test = env.generate_training_data(num_steps=[1500, 465], noisy_dims=noisy_dims, whitening=False, chunks=2)
    #pca = mdp.nodes.PCANode(output_dim=keep_variance)
    #pca.train(data_train[0])
    #data_train = pca.execute(data_train[0])
    #data_test = pca.execute(data_test[0])
    #whitening = mdp.nodes.WhiteningNode()
    #whitening.train(data_train)
    #data_train = whitening.execute(data_train)
    #data_test = whitening.execute(data_test)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test



@mem.cache
def generate_training_data_event(N, noisy_dims, prob=.1, seed=None, repetition_index=None):
    unique_seed = abs(hash(joblib.hash((N, noisy_dims, seed, prob, repetition_index))))
    env = EnvEvent(prob=prob, seed=unique_seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=True, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test
    


@mem.cache
def generate_training_data_kai(N, noisy_dims, seed=None, repetition_index=None):
    unique_seed = abs(hash(joblib.hash((N, noisy_dims, seed, repetition_index))))
    env = EnvKai(seed=unique_seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=True, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test
    


@mem.cache
def generate_training_data_ladder(N, noisy_dims, num_states=10, max_steps=4, seed=None, repetition_index=None):
    unique_seed = abs(hash(joblib.hash((N, num_states, max_steps, noisy_dims, seed, repetition_index))))
    env = EnvLadder(num_states=num_states, max_steps=max_steps, seed=unique_seed)
    #data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=True, chunks=2)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test
    


@mem.cache
def calc_projection_random(data_train, data_test, output_dim=1, seed=None, repetition_index=None):

    unique_seed = abs(hash(joblib.hash((data_train, data_test, output_dim, seed, repetition_index))))
    
    model = gpfa.RandomProjection(output_dim=output_dim, seed=unique_seed)
    model.train(data_train)
    
    result = model.execute(data_test)
    return result



@mem.cache
def calc_projection_sfa(data_train, data_test, output_dim=1):
    
    model = mdp.nodes.SFANode(output_dim=output_dim)
    model.train(data_train)
    
    result = model.execute(data_test)
    return result



@mem.cache
def calc_projection_foreca(data_train, data_test, output_dim=1, seed=None, repetition_index=None):
    # rv: 2
    
    unique_seed = abs(hash(joblib.hash((data_train, data_test, output_dim, seed, repetition_index))))
    
    model = foreca_node.ForeCA(output_dim=output_dim, seed=unique_seed)
    model.train(data_train)
    
    result = model.execute(data_test)
    return result



@mem.cache
def calc_projection_pfa(data_train, data_test, p, K, output_dim=1):
    
    model = PFANodeMDP.PFANode(p=p, k=K, affine=False, output_dim=output_dim)
    model.train(data_train)
    
    result = model.execute(data_test)
    return result



@mem.cache
def calc_projection_gpfa(data_train, data_test, k, iterations, iteration_dim, variance_graph, neighborhood_graph, weighted_edges, output_dim=1):
    
    model = gpfa.gPFA(k=k, 
                      output_dim=output_dim, 
                      iterations=iterations, 
                      iteration_dim=iteration_dim, 
                      variance_graph=variance_graph,
                      neighborhood_graph=neighborhood_graph,
                      weighted_edges=weighted_edges)    
    model.train(data_train)
    
    result = model.execute(data_test)
    return result



@mem.cache
def calc_projection_gpfa_kernelized(data_train, data_test, k, kernel_poly_degree, iterations, iteration_dim, variance_graph, weighted_edges, output_dim=1):
    
    model = gpfa.gPFAkernelized(k=k,
                                kernel_poly_degree=kernel_poly_degree, 
                                output_dim=output_dim, 
                                iterations=iterations, 
                                iteration_dim=iteration_dim, 
                                variance_graph=variance_graph,
                                weighted_edges=weighted_edges)    
    model.train(data_train)
    
    result = model.execute(data_test)
    return result



@mem.cache
def calc_projection_lpp(data_train, data_test, k, weighted_edges, output_dim=1):
    
    model = gpfa.LPP(k=k, weighted_edges=weighted_edges, output_dim=output_dim)
    model.train(data_train)
    
    result = model.execute(data_test)
    return result



@mem.cache
def calc_error(data, k=10, measure='trace_of_avg_cov'):
    
    if measure == 'trace_of_avg_cov':
        return gpfa.calc_predictability_trace_of_avg_cov(data, k)
    elif measure == 'det_of_avg_cov':
        return gpfa.calc_predictability_det_of_avg_cov(data, k)
    elif measure == 'avg_det_of_cov':
        return gpfa.calc_predictability_avg_det_of_cov(data, k)
    elif measure == 'graph_full':
        return gpfa.calc_predictability_graph_full(data, k)
    elif measure == 'graph_star':
        return gpfa.calc_predictability_graph_star(data, k)
    else:
        assert False



def prediction_error(algorithm, N, k, p, K, iterations, noisy_dims, kernel_poly_degree=2,
                     expansion=1, neighborhood_graph=None, weighted_edges=True, 
                     keep_variance=1., iteration_dim=2, event_prob=.1, num_states=10, 
                     max_steps=4, data='swiss_roll', measure='det_var', output_dim=1, 
                     seed=None, repetition_index=None):
    # rv: 7
    assert algorithm in ['random', 'foreca', 'sfa', 'pfa', 'gpfa-1', 'gpfa-2', 'gpfa-1-kernelized', 'gpfa-2-kernelized', 'lpp']
    
    # generate training data
    data_train, data_test = generate_training_data(N=N, 
                                                   noisy_dims=noisy_dims, 
                                                   expansion=expansion,
                                                   keep_variance=keep_variance,
                                                   event_prob=event_prob,
                                                   num_states=num_states,
                                                   max_steps=max_steps, 
                                                   data=data, 
                                                   seed=seed,
                                                   repetition_index=repetition_index)
    
    # train algorithm
    if algorithm == 'random':
        data_test_projected = calc_projection_random(data_train=data_train, 
                                                     data_test=data_test, 
                                                     output_dim=output_dim,
                                                     seed=seed,
                                                     repetition_index=repetition_index)
    elif algorithm == 'foreca':
        data_test_projected = calc_projection_foreca(data_train=data_train, 
                                                     data_test=data_test,
                                                     output_dim=output_dim,
                                                     seed=seed,
                                                     repetition_index=repetition_index)
    elif algorithm == 'sfa':
        data_test_projected = calc_projection_sfa(data_train=data_train, 
                                                  data_test=data_test,
                                                  output_dim=output_dim)
    elif algorithm == 'pfa':
        data_test_projected = calc_projection_pfa(data_train=data_train, 
                                                  data_test=data_test,
                                                  output_dim=output_dim, 
                                                  p=p, 
                                                  K=K)
    elif algorithm == 'gpfa-1':
        if not neighborhood_graph:
            neighborhood_graph = False
        data_test_projected = calc_projection_gpfa(data_train=data_train, 
                                                   data_test=data_test, 
                                                   k=k, 
                                                   iterations=iterations, 
                                                   iteration_dim=iteration_dim, 
                                                   variance_graph=True, 
                                                   neighborhood_graph=neighborhood_graph,
                                                   weighted_edges=weighted_edges,
                                                   output_dim=output_dim)
    elif algorithm == 'gpfa-2':
        if not neighborhood_graph:
            neighborhood_graph = True
        data_test_projected = calc_projection_gpfa(data_train=data_train, 
                                                   data_test=data_test, 
                                                   k=k, 
                                                   iterations=iterations, 
                                                   iteration_dim=iteration_dim, 
                                                   variance_graph=False, 
                                                   neighborhood_graph=neighborhood_graph,
                                                   weighted_edges=weighted_edges,
                                                   output_dim=output_dim)
    elif algorithm == 'gpfa-1-kernelized':
        if not neighborhood_graph:
            neighborhood_graph = False
        data_test_projected = calc_projection_gpfa_kernelized(data_train=data_train, 
                                                              data_test=data_test, 
                                                              k=k,
                                                              kernel_poly_degree=kernel_poly_degree, 
                                                              iterations=iterations, 
                                                              iteration_dim=iteration_dim, 
                                                              variance_graph=True, 
                                                              weighted_edges=weighted_edges,
                                                              output_dim=output_dim)
    elif algorithm == 'gpfa-2-kernelized':
        if not neighborhood_graph:
            neighborhood_graph = True
        data_test_projected = calc_projection_gpfa_kernelized(data_train=data_train, 
                                                              data_test=data_test, 
                                                              k=k, 
                                                              kernel_poly_degree=kernel_poly_degree, 
                                                              iterations=iterations, 
                                                              iteration_dim=iteration_dim, 
                                                              variance_graph=False, 
                                                              weighted_edges=weighted_edges,
                                                              output_dim=output_dim)
    elif algorithm == 'lpp':
        data_test_projected = calc_projection_lpp(data_train=data_train, 
                                                  data_test=data_test, 
                                                  k=k,
                                                  weighted_edges=False,
                                                  output_dim=output_dim)
        
    # return prediction error
    return calc_error(data=data_test_projected, k=k, measure=measure)



def calc_baseline(N, k, data, measure, seed=None, repetition_index=None):
    
    # generate data
    if data == 'swiss_roll':
        data, _ = generate_training_data_swiss_roll(N=N, noisy_dims=0, seed=seed, repetition_index=repetition_index)
    elif data == 'oscillation':
        data, _ = generate_training_data_oscillation(N=N, noisy_dims=0, seed=seed, repetition_index=repetition_index)
    else:
        assert False
        
    return calc_error(data=data, k=k, measure=measure)



def main():
    
    # parameters
    algorithms = ['random', 'sfa', 'pfa', 'gpfa']#, 'foreca']
    p = 2
    K = 1
    k = 20#[20, 100] # [2, 3, 5, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500]
    N = 2000 #[1000, 2000, 3000, 4000, 5000] 1965
    noisy_dims = 20#[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100, 200, 300, 400]#, 500] #[0, 50, 100, 150, 200, 250, 300, 350, 400]
    #noisy_dims = [0, 50, 100]#, 150, 200, 250, 300, 350, 400]
    expansion = 2
    keep_variance = 1. #[.99, .98, .95, .90, .80]
    iterations = 50#[50, 100] #[1, 10, 20, 30, 40, 50, 100]
    iteration_dim = 2 # [2, 5, 10, 20, 50, 100, 200]
    neighborhood_graph=True
    event_prob = .1#[.1, .2, .3, .4, .5]
    num_states = 10
    max_steps = 4
    data = 'ladder' #'oscillation' # 'swiss_roll'
    measure = 'avg_det_of_cov' #'det_of_avg_cov'
    seed = 0
    
    # plotter arguments
    processes = None
    repetitions = 10
    plot_elapsed_time = False

    # plot results from algorithms
    result = eep.plot(prediction_error,
                      algorithm=algorithms,
                      k=k,
                      N=N,
                      p=p,
                      K=K,
                      expansion=expansion,
                      iterations=iterations,
                      noisy_dims=noisy_dims,
                      keep_variance=keep_variance,
                      iteration_dim=iteration_dim,
                      variance_graph=False,
                      neighborhood_graph=neighborhood_graph,
                      event_prob=event_prob,
                      num_states=num_states,
                      max_steps=max_steps,
                      data=data,
                      processes=processes,
                      repetitions=repetitions,
                      measure=measure,
                      save_plot=False,
                      show_plot=False,
                      seed=seed,
                      plot_elapsed_time=plot_elapsed_time)
  
    # plot a baseline
    #result_baseline = eep.evaluate(calc_baseline, N=N, k=k, data=data, measure=measure, repetitions=repetitions, processes=1)
    #baseline = np.mean(result_baseline.values, axis=-1)
    #plt.plot([1, noisy_dims[-1]], [baseline, baseline], '--', color='black')

    # show plot
    #plt.legend(algorithms, loc='best')
    #plt.legend(algorithms + ['baseline'], loc='best')
    #plt.gca().set_xscale('log')
    #plt.gca().set_yscale('log')
    #plt.savefig('plottr_results/%s.png' % result.result_prefix)
    plt.show()
    return



if __name__ == '__main__':
    main()
