import joblib
import numpy as np
import sys

from matplotlib import pyplot as plt

import mdp

sys.path.append('/home/weghebvc/workspace/git/GNUPFA/src/')
import PFANodeMDP

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

sys.path.append('/home/weghebvc/workspace/git/environments/src/')
from envs.env_dead_corners import EnvDeadCorners
from envs.env_event import EnvEvent
from envs.env_eeg import EnvEEG
from envs.env_face import EnvFace
from envs.env_kai import EnvKai
from envs.env_ladder import EnvLadder
from envs.env_mario_canned import EnvMarioCanned
from envs.env_meg import EnvMEG
from envs.env_oscillator import EnvOscillator
from envs.env_random import EnvRandom
from envs.env_ribbon import EnvRibbon
from envs.env_swiss_roll import EnvSwissRoll

import foreca.foreca_node as foreca_node
import gpfa
import NonlinearNoiseNode


# prepare joblib.Memory
cachedir = '/scratch/weghebvc'
#cachedir = None
mem = joblib.Memory(cachedir=cachedir, verbose=1)



def generate_training_data(N, noisy_dims=0, expansion=1, keep_variance=1., event_prob=.1, num_states=10, max_steps=4, corner_size=.2, data='swiss_roll', seed=None):
    
    assert data in ['random', 'oscillation', 'swiss_roll', 'face', 'event', 'ladder', 'ribbon', 'swiss_roll_squared_noise', 'kai', 'dead_corners', 'mario_window', 'eeg', 'meg']
    
    # generate data
    if data == 'random':
        data_train, data_test = generate_training_data_random(N=N, 
                                                              noisy_dims=noisy_dims, 
                                                              seed=seed)
    elif data == 'swiss_roll':
        data_train, data_test = generate_training_data_swiss_roll(N=N, 
                                                                  noisy_dims=noisy_dims, 
                                                                  seed=seed)
    elif data == 'oscillation':
        data_train, data_test = generate_training_data_oscillation(N=N, 
                                                                   noisy_dims=noisy_dims, 
                                                                   seed=seed)
    elif data == 'face':
        data_train, data_test = generate_training_data_face(N=N, 
                                                            noisy_dims=noisy_dims)
    elif data == 'event':
        data_train, data_test = generate_training_data_event(N=N,
                                                             noisy_dims=noisy_dims,
                                                             seed=seed,
                                                             prob=event_prob)
    elif data == 'ribbon':
        data_train, data_test = generate_training_data_ribbon(N=N,
                                                              noisy_dims=noisy_dims,
                                                              seed=seed)
    elif data == 'ladder':
        data_train, data_test = generate_training_data_ladder(N=N,
                                                              noisy_dims=noisy_dims,
                                                              num_states=num_states,
                                                              max_steps=max_steps,
                                                              seed=seed)
    elif data == 'swiss_roll_squared_noise':
        data_train, data_test = generate_training_data_swiss_roll(N=N, 
                                                                  noisy_dims=noisy_dims, 
                                                                  seed=seed)
        noise_node = NonlinearNoiseNode.NonlinearNoiseNode(dims_modified=2, seed=seed)
        data_train = noise_node.execute(data_train)
        data_test = noise_node.execute(data_test)
    elif data == 'kai':
        data_train, data_test = generate_training_data_kai(N=N, 
                                                           noisy_dims=noisy_dims, 
                                                           seed=seed)
    elif data == 'dead_corners':
        data_train, data_test = generate_training_data_dead_corners(N=N,
                                                                    corner_size=corner_size, 
                                                                    noisy_dims=noisy_dims, 
                                                                    seed=seed)
    elif data == 'mario_window':
        data_train, data_test = generate_training_data_mario(N=N,
                                                             window_only=True,
                                                             noisy_dims=noisy_dims, 
                                                             seed=seed)
    elif data == 'eeg':
        data_train, data_test = generate_training_data_eeg(N=N,
                                                           noisy_dims=noisy_dims, 
                                                           seed=seed)
    elif data == 'meg':
        data_train, data_test = generate_training_data_meg(N=[N,75],
                                                           noisy_dims=noisy_dims, 
                                                           seed=seed)
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
def generate_training_data_random(N, noisy_dims, seed=None):
    env = EnvRandom(ndim=2, seed=seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test



@mem.cache
def generate_training_data_swiss_roll(N, noisy_dims, seed=None):
    env = EnvSwissRoll(seed=seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test



@mem.cache
def generate_training_data_ribbon(N, noisy_dims, sigma_noise=.05, seed=None):
    env = EnvRibbon(seed=seed, step_size=.1, sigma_noise=sigma_noise)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test



@mem.cache
def generate_training_data_oscillation(N, noisy_dims, seed=None):
    env = EnvOscillator(transition_prob=.9, seed=seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test



@mem.cache
def generate_training_data_face(N, noisy_dims):
    env = EnvFace()
    data_train, data_test = env.generate_training_data(num_steps=[1500, 465], noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test



@mem.cache
def generate_training_data_event(N, noisy_dims, prob=.1, seed=None):
    env = EnvEvent(prob=prob, seed=seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test
    


@mem.cache
def generate_training_data_kai(N, noisy_dims, seed=None):
    env = EnvKai(seed=seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test
    


@mem.cache
def generate_training_data_dead_corners(N, noisy_dims, corner_size=.1, seed=None):
    # rev: 4
    env = EnvDeadCorners(corner_size=corner_size, seed=seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test
    


@mem.cache
def generate_training_data_ladder(N, noisy_dims, num_states=10, max_steps=4, seed=None):
    env = EnvLadder(num_states=num_states, max_steps=max_steps, seed=seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test
    


@mem.cache
def generate_training_data_mario(N, window_only, noisy_dims, seed=None):
    # rev: 2
    env = EnvMarioCanned(window_only=window_only, seed=seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test
    


@mem.cache
def generate_training_data_eeg(N, noisy_dims, seed=None):
    env = EnvEEG(seed=seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test
    


@mem.cache
def generate_training_data_meg(N, noisy_dims, seed=None):
    #rev: 2
    env = EnvMEG(seed=seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test
    


@mem.cache
def calc_projection_random(data_train, data_test, output_dim=1, seed=None):

    model = gpfa.RandomProjection(output_dim=output_dim, seed=seed)
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
def calc_projection_foreca(data_train, data_test, output_dim=1, seed=None):
    
    model = foreca_node.ForeCA(output_dim=output_dim, seed=seed)
    model.train(data_train)
    
    result = model.execute(data_test)
    return result



@mem.cache
def calc_projection_pfa(data_train, data_test, p, K, causal_features=False, output_dim=1):
    # rev 1
    
    if causal_features:
        data_train = data_train[::-1]
    
    model = PFANodeMDP.PFANode(p=p, k=K, affine=False, output_dim=output_dim)
    model.train(data_train)
    
    result = model.execute(data_test)
    return result



@mem.cache
def calc_projection_gpfa(data_train, data_test, k, iterations, variance_graph, neighborhood_graph, weighted_edges, causal_features, p=1, output_dim=1):
    # rev 14
    
    model = gpfa.gPFA(k=k, 
                      p=p,
                      output_dim=output_dim, 
                      iterations=iterations, 
                      #iteration_dim=iteration_dim, 
                      variance_graph=variance_graph,
                      neighborhood_graph=neighborhood_graph,
                      weighted_edges=weighted_edges,
                      causal_features=causal_features)
    model.train(data_train)
    
    result = model.execute(data_test)
    return result



@mem.cache
def calc_projection_gpfa_kernel(data_train, data_test, k, degree, iterations, variance_graph, output_dim=1):
    
    model = gpfa.gPFAkernel(k=k,
                            degree=degree, 
                            output_dim=output_dim, 
                            iterations=iterations, 
                            variance_graph=variance_graph)    
    model.train(data_train)
    
    result = model.execute(data_test)
    return result



@mem.cache
def calc_projection_gpfa_sr(data_train, data_test, k, iterations, variance_graph, output_dim=1):
    
    model = gpfa.gPFAsr(k=k,
                        output_dim=output_dim, 
                        iterations=iterations, 
                        variance_graph=variance_graph)    
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
def calc_error(data, k=10, p=1, measure='trace_of_avg_cov', reverse_error=False):
    # rev: 2
    
    if reverse_error:
        data = data[::-1,:]
    
    if measure == 'trace_of_avg_cov':
        return gpfa.calc_predictability_trace_of_avg_cov(data, k=k, p=p)
    else:
        assert False



def prediction_error(algorithm, N, k, p, P, K, iterations, noisy_dims, neighborhood_graph, 
                     kernel_poly_degree=3, expansion=1, weighted_edges=True, 
                     keep_variance=1., #iteration_dim=2, 
                     event_prob=.1, num_states=10, 
                     max_steps=4, corner_size=.2, data='swiss_roll', measure='trace_of_avg_cov', 
                     output_dim=1, reverse_error=False, seed=None):
    # rv: 19
    assert algorithm in ['random', 'foreca', 'sfa', 'pfa', 'cfa', 'gpfa-1', 'gpfa-2', 'gcfa-1', 'gcfa-2', 'gcfa-1-kernel', 'gcfa-2-kernel', 'gcfa-1-sr', 'gcfa-2-sr', 'lpp']
    
    # generate training data
    data_train, data_test = generate_training_data(N=N, 
                                                   noisy_dims=noisy_dims, 
                                                   expansion=expansion,
                                                   keep_variance=keep_variance,
                                                   event_prob=event_prob,
                                                   num_states=num_states,
                                                   max_steps=max_steps,
                                                   corner_size=corner_size, 
                                                   data=data, 
                                                   seed=seed)
    
    # train algorithm
    if algorithm == 'random':
        data_test_projected = calc_projection_random(data_train=data_train, 
                                                     data_test=data_test, 
                                                     output_dim=output_dim,
                                                     seed=seed)
    elif algorithm == 'foreca':
        data_test_projected = calc_projection_foreca(data_train=data_train, 
                                                     data_test=data_test,
                                                     output_dim=output_dim,
                                                     seed=seed)
    elif algorithm == 'sfa':
        data_test_projected = calc_projection_sfa(data_train=data_train, 
                                                  data_test=data_test,
                                                  output_dim=output_dim)
    elif algorithm == 'pfa':
        data_test_projected = calc_projection_pfa(data_train=data_train, 
                                                  data_test=data_test,
                                                  output_dim=output_dim, 
                                                  p=P, 
                                                  K=K,
                                                  causal_features=False)
    elif algorithm == 'cfa':
        data_test_projected = calc_projection_pfa(data_train=data_train, 
                                                  data_test=data_test,
                                                  output_dim=output_dim, 
                                                  p=p, 
                                                  K=K,
                                                  causal_features=True)
    elif algorithm == 'gpfa-1':
        data_test_projected = calc_projection_gpfa(data_train=data_train, 
                                                   data_test=data_test, 
                                                   k=k, 
                                                   p=p,
                                                   iterations=iterations, 
                                                   #iteration_dim=iteration_dim, 
                                                   variance_graph=True, 
                                                   neighborhood_graph=neighborhood_graph,
                                                   weighted_edges=weighted_edges,
                                                   causal_features=False,
                                                   output_dim=output_dim)
    elif algorithm == 'gpfa-2':
        data_test_projected = calc_projection_gpfa(data_train=data_train, 
                                                   data_test=data_test, 
                                                   k=k, 
                                                   p=p,
                                                   iterations=iterations, 
                                                   #iteration_dim=iteration_dim, 
                                                   variance_graph=False, 
                                                   neighborhood_graph=neighborhood_graph,
                                                   weighted_edges=weighted_edges,
                                                   causal_features=False,
                                                   output_dim=output_dim)
    elif algorithm == 'gcfa-1':
        data_test_projected = calc_projection_gpfa(data_train=data_train, 
                                                   data_test=data_test, 
                                                   k=k, 
                                                   p=p,
                                                   iterations=iterations, 
                                                   #iteration_dim=iteration_dim, 
                                                   variance_graph=True, 
                                                   neighborhood_graph=neighborhood_graph,
                                                   weighted_edges=weighted_edges,
                                                   causal_features=True,
                                                   output_dim=output_dim)
    elif algorithm == 'gcfa-2':
        data_test_projected = calc_projection_gpfa(data_train=data_train, 
                                                   data_test=data_test, 
                                                   k=k, 
                                                   p=p,
                                                   iterations=iterations, 
                                                   #iteration_dim=iteration_dim, 
                                                   variance_graph=False, 
                                                   neighborhood_graph=neighborhood_graph,
                                                   weighted_edges=weighted_edges,
                                                   causal_features=True,
                                                   output_dim=output_dim)
    elif algorithm == 'gcfa-1-kernel':
        data_test_projected = calc_projection_gpfa_kernel(data_train=data_train, 
                                                          data_test=data_test, 
                                                          k=k,
                                                          degree=kernel_poly_degree, 
                                                          iterations=iterations, 
                                                          variance_graph=True, 
                                                          output_dim=output_dim)
    elif algorithm == 'gcfa-2-kernel':
        data_test_projected = calc_projection_gpfa_kernel(data_train=data_train, 
                                                          data_test=data_test, 
                                                          k=k, 
                                                          degree=kernel_poly_degree, 
                                                          iterations=iterations, 
                                                          variance_graph=False, 
                                                          output_dim=output_dim)
    elif algorithm == 'gcfa-1-sr':
        data_test_projected = calc_projection_gpfa_sr(data_train=data_train, 
                                                      data_test=data_test, 
                                                      k=k,
                                                      iterations=iterations, 
                                                      variance_graph=True, 
                                                      output_dim=output_dim)
    elif algorithm == 'gcfa-2-sr':
        data_test_projected = calc_projection_gpfa_sr(data_train=data_train, 
                                                      data_test=data_test, 
                                                      k=k, 
                                                      iterations=iterations, 
                                                      variance_graph=False, 
                                                      output_dim=output_dim)
    elif algorithm == 'lpp':
        data_test_projected = calc_projection_lpp(data_train=data_train, 
                                                  data_test=data_test, 
                                                  k=k,
                                                  weighted_edges=False,
                                                  output_dim=output_dim)
        
    # return prediction error
    return calc_error(data=data_test_projected, k=k, p=p, measure=measure, reverse_error=reverse_error)



def calc_baseline(N, k, data, measure, seed=None):
    
    # generate data
    if data == 'swiss_roll':
        data, _ = generate_training_data_swiss_roll(N=N, noisy_dims=0, seed=seed)
    elif data == 'oscillation':
        data, _ = generate_training_data_oscillation(N=N, noisy_dims=0, seed=seed)
    else:
        assert False
        
    return calc_error(data=data, k=k, measure=measure)



if __name__ == '__main__':
    pass
