import joblib
import mdp
import numpy as np
import sys

from enum import Enum

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
from envs.env_ratlab import EnvRatlab
from envs.env_ribbon import EnvRibbon
from envs.env_swiss_roll import EnvSwissRoll



# prepare joblib.Memory
cache = joblib.Memory(cachedir='/scratch/weghebvc', verbose=1)


Envs = Enum('Envs', 'random oscillation swiss_roll face ladder ratlab ribbon kai dead_corners mario eeg meg')



def set_cachedir(cachedir=None):
    """
    Call this method to change the joblib caching of this module.
    """
    global cache
    cache = joblib.Memory(cachedir=cachedir, verbose=1)
    return



def generate_training_data(N, repetition_index, noisy_dims=0, expansion=1, keep_variance=1., num_states=10, max_steps=4, corner_size=.2, data='swiss_roll', seed=None):
    
    # generate data
    if data == Envs.random:
        data_train, data_test = generate_training_data_random(
                                    N=N, 
                                    noisy_dims=noisy_dims, 
                                    seed=seed,
                                    repetition_index=repetition_index)
    elif data == Envs.swiss_roll:
        data_train, data_test = generate_training_data_swiss_roll(
                                    N=N, 
                                    noisy_dims=noisy_dims, 
                                    seed=seed,
                                    repetition_index=repetition_index)
    elif data == Envs.oscillation:
        data_train, data_test = generate_training_data_oscillation(
                                    N=N, 
                                    noisy_dims=noisy_dims, 
                                    seed=seed,
                                    repetition_index=repetition_index)
    elif data == Envs.face:
        data_train, data_test = generate_training_data_face(
                                    N=N, 
                                    noisy_dims=noisy_dims,
                                    repetition_index=repetition_index)
    elif data == Envs.ribbon:
        data_train, data_test = generate_training_data_ribbon(
                                    N=N,
                                    noisy_dims=noisy_dims,
                                    seed=seed,
                                    repetition_index=repetition_index)
    elif data == Envs.ladder:
        data_train, data_test = generate_training_data_ladder(
                                    N=N,
                                    noisy_dims=noisy_dims,
                                    num_states=num_states,
                                    max_steps=max_steps,
                                    seed=seed,
                                    repetition_index=repetition_index)
    elif data == Envs.ratlab:
        data_train, data_test = generate_training_data_ratlab(
                                    N=N,
                                    noisy_dims=noisy_dims,
                                    seed=seed,
                                    repetition_index=repetition_index)
    elif data == Envs.kai:
        data_train, data_test = generate_training_data_kai(
                                    N=N, 
                                    noisy_dims=noisy_dims, 
                                    seed=seed,
                                    repetition_index=repetition_index)
    elif data == Envs.dead_corners:
        data_train, data_test = generate_training_data_dead_corners(
                                    N=N,
                                    corner_size=corner_size, 
                                    noisy_dims=noisy_dims, 
                                    seed=seed,
                                    repetition_index=repetition_index)
    elif data == Envs.mario:
        data_train, data_test = generate_training_data_mario(
                                    N=N,
                                    window_only=True,
                                    noisy_dims=noisy_dims, 
                                    seed=seed,
                                    repetition_index=repetition_index)
    elif data == Envs.eeg:
        data_train, data_test = generate_training_data_eeg(
                                    N=N,
                                    noisy_dims=noisy_dims, 
                                    seed=seed,
                                    repetition_index=repetition_index)
    elif data == Envs.meg:
        data_train, data_test = generate_training_data_meg(
                                    N=N,
                                    noisy_dims=noisy_dims, 
                                    seed=seed,
                                    repetition_index=repetition_index)
    else:
        assert False

    # PCA
    if keep_variance < 1.:
        pca = mdp.nodes.PCANode(output_dim=keep_variance)
        pca.train(data_train)
        data_train = pca.execute(data_train)
        data_test = pca.execute(data_test)

    # expansion        
    if expansion > 1:
        ex = mdp.nodes.PolynomialExpansionNode(degree=expansion)
        data_train = ex.execute(data_train)
        data_test = ex.execute(data_test)
        
    # whitening
    whitening = mdp.nodes.WhiteningNode(reduce=True)
    whitening.train(data_train)
    data_train = whitening.execute(data_train)
    data_test = whitening.execute(data_test)
    
    return data_train, data_test



#@mem.cache
def generate_training_data_random(N, noisy_dims, seed, repetition_index):
    seed = ep.calc_argument_seed()
    env = EnvRandom(ndim=2, seed=seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test



#@mem.cache
def generate_training_data_swiss_roll(N, noisy_dims, seed, repetition_index):
    seed = ep.calc_argument_seed()
    env = EnvSwissRoll(seed=seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test



#@mem.cache
def generate_training_data_ribbon(N, noisy_dims, seed, repetition_index, sigma_noise=.05):
    seed = ep.calc_argument_seed()
    env = EnvRibbon(seed=seed, step_size=.1, sigma_noise=sigma_noise)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test



#@mem.cache
def generate_training_data_oscillation(N, noisy_dims, seed, repetition_index):
    seed = ep.calc_argument_seed()
    env = EnvOscillator(transition_prob=.9, seed=seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test



#@mem.cache
def generate_training_data_face(N, noisy_dims):
    env = EnvFace()
    data_train, data_test = env.generate_training_data(num_steps=[1500, 465], noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test



#@mem.cache
def generate_training_data_event(N, noisy_dims, seed, repetition_index, prob=.1):
    seed = ep.calc_argument_seed()
    env = EnvEvent(prob=prob, seed=seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test
    


#@mem.cache
def generate_training_data_kai(N, noisy_dims, seed, repetition_index):
    seed = ep.calc_argument_seed()
    env = EnvKai(seed=seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test
    


#@mem.cache
def generate_training_data_dead_corners(N, noisy_dims, seed, repetition_index, corner_size=.1):
    seed = ep.calc_argument_seed()
    env = EnvDeadCorners(corner_size=corner_size, seed=seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test
    


#@mem.cache
def generate_training_data_ladder(N, noisy_dims, seed, repetition_index, num_states=10, max_steps=4):
    seed = ep.calc_argument_seed()
    env = EnvLadder(num_states=num_states, max_steps=max_steps, seed=seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test
    


#@mem.cache
def generate_training_data_ratlab(N, noisy_dims, seed, repetition_index):
    seed = ep.calc_argument_seed()
    env = EnvRatlab(seed=seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test
    


#@mem.cache
def generate_training_data_mario(N, window_only, seed, repetition_index, noisy_dims):
    seed = ep.calc_argument_seed()
    env = EnvMarioCanned(window_only=window_only, seed=seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test
    


#@mem.cache
def generate_training_data_eeg(N, noisy_dims, seed, repetition_index):
    seed = ep.calc_argument_seed()
    env = EnvEEG(seed=seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test
    


#@mem.cache
def generate_training_data_meg(N, noisy_dims, seed, repetition_index):
    seed = ep.calc_argument_seed()
    env = EnvMEG(seed=seed)
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=2)
    data_train = data_train[0]
    data_test = data_test[0]
    return data_train, data_test

