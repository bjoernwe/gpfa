import joblib
import mdp
import numpy as np
import sys

from enum import Enum

import foreca.foreca_node as foreca_node
import gpfa

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

sys.path.append('/home/weghebvc/workspace/git/GNUPFA/src/')
import PFANodeMDP

sys.path.append('/home/weghebvc/workspace/git/environments/src/')
from envs.environment import Noise
from envs.env_data import EnvData
from envs.env_data2d import EnvData2D
from envs.env_dead_corners import EnvDeadCorners
from envs.env_kai import EnvKai
from envs.env_ladder import EnvLadder
from envs.env_random import EnvRandom
from envs.env_swiss_roll import EnvSwissRoll



# prepare joblib.Memory
mem = joblib.Memory(cachedir='/scratch/weghebvc', verbose=1)


Datasets = Enum('Datasets', 'Random SwissRoll Face MarkovChain RatLab Kai Teleporter Mario Mario_window EEG MEG, Tumor')

Algorithms = Enum('Algorithms', 'None Random SFA ForeCA PFA GPFA1 GPFA2')

Measures = Enum('measures', 'delta delta_ndim gpfa gpfa_ndim')



def set_cachedir(cachedir=None):
    """
    Call this method to change the joblib caching of this module.
    """
    global mem
    mem = joblib.Memory(cachedir=cachedir, verbose=1)
    return



def update_seed_argument(**kwargs):
    """
    Helper function that replaces the the seed argument by a new seed that
    depends on all arguments. If repetition_index is given it will be removed.
    """
    new_seed = hash(frozenset(kwargs)) % np.iinfo(np.uint32).max
    if 'repetition_index' in kwargs:
        kwargs.pop('repetition_index')
    kwargs['seed'] = new_seed
    return kwargs



def generate_training_data(data, N, noisy_dims=0, chunks=2, repetition_index=None, seed=None, **kwargs):

    # generate data
    if data == Datasets.Random:
        fargs = update_seed_argument(ndim=noisy_dims, noise_dist=Noise.normal, repetition_index=repetition_index, seed=seed)
        env = EnvRandom(**fargs)
    elif data == Datasets.SwissRoll:
        fargs = update_seed_argument(sigma=kwargs.get('sigma', .5), repetition_index=repetition_index, seed=seed)
        env = EnvSwissRoll(**fargs)
    elif data == Datasets.Face:
        env = EnvData2D(dataset=EnvData2D.Datasets.Face, scaling=kwargs.get('scaling', 1.))
    elif data == Datasets.MarkovChain:
        fargs = update_seed_argument(num_states=kwargs.get('ladder_num_states', 10), 
                                     max_steps=kwargs.get('ladder_max_steps', 4), 
                                     allow_stay=kwargs.get('ladder_allow_stay', False),
                                     repetition_index=repetition_index, 
                                     seed=seed)
        env = EnvLadder(**fargs)
    elif data == Datasets.RatLab:
        env = EnvData2D(dataset=EnvData2D.Datasets.RatLab, scaling=kwargs.get('scaling', 1.))
    elif data == Datasets.Tumor:
        env = EnvData2D(dataset=EnvData2D.Datasets.Tumor, scaling=kwargs.get('scaling', 1.))
    elif data == Datasets.Kai:
        fargs = update_seed_argument(repetition_index=repetition_index, seed=seed)
        env = EnvKai(**fargs)
    elif data == Datasets.Teleporter:
        fargs = update_seed_argument(sigma=kwargs.get('dead_corners_sigma', .2), 
                                     corner_size=kwargs.get('dead_corners_corner_size', .1), 
                                     ndim=2, 
                                     repetition_index=repetition_index,
                                     seed=seed)
        env = EnvDeadCorners(**fargs)
    elif data == Datasets.Mario:
        env = EnvData2D(dataset=EnvData2D.Datasets.Mario, scaling=kwargs.get('scaling', 1.))
    elif data == Datasets.Mario_window:
        env = EnvData2D(dataset=EnvData2D.Datasets.Mario, window=((70,70),(90,90)), scaling=kwargs.get('scaling', 1.))
    elif data == Datasets.EEG:
        env = EnvData(dataset=EnvData.Datasets.EEG)
    elif data == Datasets.MEG:
        env = EnvData(dataset=EnvData.Datasets.MEG)
    else:
        assert False

    # generate data
    data_chunks = [chunk[0] for chunk in env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=False, chunks=chunks)]

    # PCA
    keep_variance = kwargs.get('keep_variance', 1.)
    if keep_variance < 1.:
        print "Dim. of %s before PCA: %d" % (data, data_chunks[0].shape[1])
        data_chunks = pca(data_chunks=data_chunks, keep_variance=keep_variance)
        print "Dim. of %s after PCA: %d" % (data, data_chunks[0].shape[1])
        
    # expansion
    expansion = kwargs.get('expansion', 1)        
    if expansion > 1:
        ex = mdp.nodes.PolynomialExpansionNode(degree=expansion)
        data_chunks = [ex.execute(chunk) for chunk in data_chunks]
        
    # whitening
    whitening = mdp.nodes.WhiteningNode(reduce=True)
    whitening.train(data_chunks[0])
    data_chunks = [whitening.execute(chunk) for chunk in data_chunks]
    
    return data_chunks



@mem.cache
def pca(data_chunks, keep_variance):
    pca = mdp.nodes.PCANode(output_dim=keep_variance, reduce=True)
    if data_chunks[0].shape[1] <= data_chunks[0].shape[0]:
        pca.train(data_chunks[0])
        data_chunks = [pca.execute(chunk) for chunk in data_chunks]
    else:
        pca.train(data_chunks[0].T)
        pca.stop_training()
        U = data_chunks[0].T.dot(pca.v)
        data_chunks = [chunk.dot(U) for chunk in data_chunks]
    return data_chunks



def train_model(algorithm, data_train, output_dim, seed, repetition_index, **kwargs):
    
    if algorithm == Algorithms.None:
        return None
    elif algorithm == Algorithms.Random:
        return train_random(data_train=data_train, 
                    output_dim=output_dim, 
                    seed=seed, 
                    repetition_index=repetition_index)
    elif algorithm == Algorithms.SFA:
        return train_sfa(data_train=data_train, output_dim=output_dim)
    elif algorithm == Algorithms.ForeCA:
        return train_foreca(data_train=data_train, 
                    output_dim=output_dim, 
                    seed=seed, 
                    repetition_index=repetition_index)
    elif algorithm == Algorithms.PFA:
        return train_pfa(data_train=data_train, 
                    output_dim=output_dim)
    elif algorithm == Algorithms.GPFA1:
        return train_gpfa(data_train=data_train,
                    k=kwargs['k'], 
                    p=kwargs.get('p', 1),
                    iterations=kwargs['iterations'], 
                    variance_graph=True,
                    neighborhood_graph=kwargs.get('neighborhood_graph', False), 
                    weighted_edges=kwargs.get('weighted_edges', True),
                    causal_features=kwargs.get('causal_features', True),
                    output_dim=output_dim)
    elif algorithm == Algorithms.GPFA2:
        return train_gpfa(data_train=data_train, 
                    k=kwargs['k'], 
                    p=kwargs.get('p', 1),
                    iterations=kwargs['iterations'], 
                    variance_graph=False,
                    neighborhood_graph=kwargs.get('neighborhood_graph', False), 
                    weighted_edges=kwargs.get('weighted_edges', True),
                    causal_features=kwargs.get('causal_features', True),
                    output_dim=output_dim)
    else:
        assert False



@mem.cache
def train_random(data_train, output_dim, seed, repetition_index):
    fargs = update_seed_argument(output_dim=output_dim, repetition_index=repetition_index, seed=seed)
    model = gpfa.RandomProjection(**fargs)
    model.train(data_train)
    return model



@mem.cache
def train_sfa(data_train, output_dim):
    model = mdp.nodes.SFANode(output_dim=output_dim)
    model.train(data_train)
    return model

 
 
@mem.cache
def train_foreca(data_train, output_dim, seed, repetition_index):
    fargs = update_seed_argument(output_dim=output_dim, repetition_index=repetition_index, seed=seed)
    model = foreca_node.ForeCA(**fargs)
    model.train(data_train)
    return model
 
 
 
@mem.cache
def train_pfa(data_train, p, K, output_dim):
    model = PFANodeMDP.PFANode(p=p, k=K, affine=False, output_dim=output_dim)
    model.train(data_train)
    return model
 
 
 
@mem.cache
def train_gpfa(data_train, k, iterations, variance_graph, neighborhood_graph=False, weighted_edges=True, causal_features=True, p=1, output_dim=1):
    model = gpfa.gPFA(k=k, 
                      p=p,
                      output_dim=output_dim, 
                      iterations=iterations, 
                      variance_graph=variance_graph,
                      neighborhood_graph=neighborhood_graph,
                      weighted_edges=weighted_edges,
                      causal_features=causal_features)
    model.train(data_train)
    return model



def calc_projected_data(data, algorithm, output_dim, N, repetition_index=None, noisy_dims=0, 
                        use_test_set=True, seed=None, **kwargs):

    chunks = 2 if use_test_set else 1
    data_chunks = generate_training_data(data=data, 
                                         N=N, 
                                         noisy_dims=noisy_dims,
                                         chunks=chunks,
                                         repetition_index=repetition_index, 
                                         seed=seed,
                                         **kwargs)
    
    model = train_model(algorithm=algorithm, 
                        data_train=data_chunks[0], 
                        output_dim=output_dim, 
                        seed=seed,
                        repetition_index=repetition_index,
                        **kwargs)
    
    if model is None:
        if use_test_set:
            projected_data = np.array(data_chunks[1], copy=True)
        else:
            projected_data = np.array(data_chunks[0], copy=True)
    else:
        if use_test_set:
            projected_data = model.execute(data_chunks[1])
        else:
            projected_data = model.execute(data_chunks[0])
        
    return projected_data



def prediction_error(measure, data, algorithm, output_dim, N, use_test_set=True, 
                     repetition_index=None, seed=None, **kwargs):
    
    projected_data = calc_projected_data(data=data, 
                                         algorithm=algorithm, 
                                         output_dim=output_dim, 
                                         N=N, 
                                         repetition_index=repetition_index, 
                                         use_test_set=use_test_set, 
                                         seed=seed, **kwargs)

    if measure == Measures.delta:
        return calc_delta(data=projected_data, ndim=False)
    elif measure == Measures.delta_ndim:
        return calc_delta(data=projected_data, ndim=True)
    elif measure == Measures.gpfa:
        return gpfa.calc_predictability_trace_of_avg_cov(x=projected_data, 
                                                         k=kwargs['k'], 
                                                         p=kwargs['p'],
                                                         ndim=False)
    elif measure == Measures.gpfa_ndim:
        return gpfa.calc_predictability_trace_of_avg_cov(x=projected_data, 
                                                         k=kwargs['k'], 
                                                         p=kwargs['p'],
                                                         ndim=True)
    else:
        assert False
    
    
    
def calc_delta(data, ndim=False):
    sfa = mdp.nodes.SFANode()
    sfa.train(data)
    sfa.stop_training()
    if ndim:
        return sfa.d
    return np.sum(sfa.d)



if __name__ == '__main__':
    #set_cachedir(cachedir=None)
    for measure in [Measures.delta, Measures.delta_ndim, Measures.gpfa, Measures.gpfa_ndim]:
        print prediction_error(measure=measure, 
                               data=Datasets.Mario_window, 
                               algorithm=Algorithms.GPFA2, 
                               output_dim=2, 
                               N=2000, 
                               k=10, 
                               p=1,
                               iterations=50,
                               seed=0)


