import joblib
import mdp
import numpy as np
import sys

from enum import Enum

import foreca.foreca_node as foreca_node
import gpfa

#sys.path.append('/home/weghebvc/workspace/git/explot/src/')
#import explot as ep

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

sys.path.append('/home/weghebvc/workspace/git/GNUPFA')
import PFACoreUtil



# prepare joblib.Memory
mem = joblib.Memory(cachedir='/scratch/weghebvc', verbose=1)


Datasets = Enum('Datasets', 'Random Crowd1 Crowd2 Crowd3 Dancing Mouth SwissRoll Face MarkovChain RatLab Kai Teleporter Mario Mario_window EEG MEG Traffic Tumor')

Algorithms = Enum('Algorithms', 'None Random SFA ForeCA PFA GPFA1 GPFA2 HiSFA HiForeCA HiPFA HiGPFA1 HiGPFA2')

Measures = Enum('Measures', 'delta delta_ndim omega omega_ndim pfa_ndim gpfa gpfa_ndim')



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
    new_seed = hash(frozenset(kwargs.items())) % np.iinfo(np.uint32).max
    if 'repetition_index' in kwargs:
        kwargs.pop('repetition_index')
    kwargs['seed'] = new_seed
    return kwargs



def generate_training_data(dataset, N, noisy_dims, n_chunks, repetition_index=None, seed=None, **kwargs):

    image_shape = None

    # generate dataset
    if dataset == Datasets.Random:
        fargs = update_seed_argument(ndim=noisy_dims, noise_dist=Noise.normal, repetition_index=repetition_index, seed=seed)
        env = EnvRandom(**fargs)
    elif dataset == Datasets.Crowd1:
        env = EnvData2D(dataset=EnvData2D.Datasets.Crowd1, scaling=kwargs.get('scaling', 1.), cachedir='/scratch/weghebvc', seed=0)
        image_shape = env.image_shape
    elif dataset == Datasets.Crowd2:
        env = EnvData2D(dataset=EnvData2D.Datasets.Crowd2, scaling=kwargs.get('scaling', 1.), cachedir='/scratch/weghebvc', seed=0)
        image_shape = env.image_shape
    elif dataset == Datasets.Crowd3:
        env = EnvData2D(dataset=EnvData2D.Datasets.Crowd3, scaling=kwargs.get('scaling', 1.), cachedir='/scratch/weghebvc', seed=0)
        image_shape = env.image_shape
    elif dataset == Datasets.Dancing:
        env = EnvData2D(dataset=EnvData2D.Datasets.Dancing, scaling=kwargs.get('scaling', 1.), cachedir='/scratch/weghebvc', seed=0)
        image_shape = env.image_shape
    elif dataset == Datasets.Mouth:
        env = EnvData2D(dataset=EnvData2D.Datasets.Mouth, scaling=kwargs.get('scaling', 1.), cachedir='/scratch/weghebvc', seed=0)
        image_shape = env.image_shape
    elif dataset == Datasets.SwissRoll:
        fargs = update_seed_argument(sigma=kwargs.get('sigma', .5), repetition_index=repetition_index, seed=seed)
        env = EnvSwissRoll(**fargs)
    elif dataset == Datasets.Face:
        env = EnvData2D(dataset=EnvData2D.Datasets.Face, scaling=kwargs.get('scaling', 1.), cachedir='/scratch/weghebvc', seed=0)
        image_shape = env.image_shape
    elif dataset == Datasets.MarkovChain:
        fargs = update_seed_argument(num_states=kwargs.get('ladder_num_states', 10), 
                                     max_steps=kwargs.get('ladder_max_steps', 4), 
                                     allow_stay=kwargs.get('ladder_allow_stay', False),
                                     repetition_index=repetition_index, 
                                     seed=seed)
        env = EnvLadder(**fargs)
    elif dataset == Datasets.RatLab:
        env = EnvData2D(dataset=EnvData2D.Datasets.RatLab, scaling=kwargs.get('scaling', 1.), cachedir='/scratch/weghebvc', seed=0)
        image_shape = env.image_shape
    elif dataset == Datasets.Tumor:
        env = EnvData2D(dataset=EnvData2D.Datasets.Tumor, scaling=kwargs.get('scaling', 1.), cachedir='/scratch/weghebvc', seed=0)
        image_shape = env.image_shape
    elif dataset == Datasets.Kai:
        fargs = update_seed_argument(repetition_index=repetition_index, seed=seed)
        env = EnvKai(**fargs)
    elif dataset == Datasets.Teleporter:
        fargs = update_seed_argument(sigma=kwargs.get('dead_corners_sigma', .2), 
                                     corner_size=kwargs.get('dead_corners_corner_size', .1), 
                                     ndim=2, 
                                     repetition_index=repetition_index,
                                     seed=seed)
        env = EnvDeadCorners(**fargs)
    elif dataset == Datasets.Mario:
        env = EnvData2D(dataset=EnvData2D.Datasets.Mario, scaling=kwargs.get('scaling', 1.), cachedir='/scratch/weghebvc', seed=0)
        image_shape = env.image_shape
    elif dataset == Datasets.Mario_window:
        env = EnvData2D(dataset=EnvData2D.Datasets.Mario, window=((70,70),(90,90)), scaling=kwargs.get('scaling', 1.), cachedir='/scratch/weghebvc', seed=0)
        image_shape = env.image_shape
    elif dataset == Datasets.EEG:
        env = EnvData(dataset=EnvData.Datasets.EEG)
    elif dataset == Datasets.MEG:
        env = EnvData(dataset=EnvData.Datasets.MEG)
    elif dataset == Datasets.Traffic:
        env = EnvData2D(dataset=EnvData2D.Datasets.Traffic, scaling=kwargs.get('scaling', 1.), cachedir='/scratch/weghebvc', seed=0)
        image_shape = env.image_shape
    else:
        assert False

    # generate dataset
    chunks = env.generate_training_data(num_steps=N, 
                                        noisy_dims=noisy_dims, 
                                        keep_variance=kwargs.get('keep_variance', 1.), 
                                        whitening=kwargs.get('whitening', False), 
                                        n_chunks=n_chunks)
    data_chunks = [chunk[0] for chunk in chunks]

#     # PCA
#     keep_variance = kwargs.get('keep_variance', 1.)
#     if keep_variance < 1.:
#         print "Dim. of %s before PCA: %d" % (dataset, data_chunks[0].shape[1])
#         data_chunks = pca(data_chunks=data_chunks, keep_variance=keep_variance)
#         print "Dim. of %s after PCA: %d" % (dataset, data_chunks[0].shape[1])
#         
#     # expansion
#     expansion = kwargs.get('expansion', 1)        
#     if expansion > 1:
#         ex = mdp.nodes.PolynomialExpansionNode(degree=expansion)
#         data_chunks = [ex.execute(chunk) for chunk in data_chunks]
#         
#     # whitening
#     whitening = mdp.nodes.WhiteningNode(reduce=True)
#     whitening.train(data_chunks[0])
#     data_chunks = [whitening.execute(chunk) for chunk in data_chunks]
    
    return data_chunks, image_shape



# @mem.cache
# def pca(data_chunks, keep_variance):
#     pca = mdp.nodes.PCANode(output_dim=keep_variance, reduce=True)
#     if data_chunks[0].shape[1] <= data_chunks[0].shape[0]:
#         pca.train(data_chunks[0])
#         data_chunks = [pca.execute(chunk) for chunk in data_chunks]
#     else:
#         pca.train(data_chunks[0].T)
#         pca.stop_training()
#         U = data_chunks[0].T.dot(pca.v)
#         data_chunks = [chunk.dot(U) for chunk in data_chunks]
#     return data_chunks



class PowerExpansion(mdp.Node):
    
    def __init__(self, input_dim, expansion=.8, dtype=None):
        super(PowerExpansion, self).__init__(input_dim=input_dim, output_dim=2*input_dim, dtype=dtype)
        self.expansion = expansion
        
    @staticmethod
    def is_trainable():
        return False
    
    def _execute(self, x):
        assert x.ndim == 2
        return np.hstack([x, np.abs(x)**.8])
    
    

def build_hierarchy_flow(image_x, image_y, output_dim, node_class, node_output_dim, 
                         expansion, channels_xy_1, spacing_xy_1, channels_xy_n, 
                         spacing_xy_n, node_kwargs):

    #channels_xy_1 = kwargs.pop('channels_xy_1', (12, 12))
    #spacing_xy_1  = kwargs.pop('spacing_xy_1',  ( 8,  8))
    #channels_xy_n = kwargs.pop('channels_xy_n', ( 3,  3))
    #spacing_xy_n  = kwargs.pop('spacing_xy_1',  ( 2,  2))
    
    #print 'channels_xy_1 = %s' % (channels_xy_1, )
    #print 'spacing_xy_1  = %s' % (spacing_xy_1, )
    #print 'channels_xy_n = %s' % (channels_xy_n, )
    #print 'spacing_xy_n  = %s' % (spacing_xy_n, )

    switchboards = []
    layers = []
    
    while len(layers) == 0 or layers[-1].output_dim > 20:

        if channels_xy_n == (2,1) or channels_xy_n == (1,2):
            channels_xy_n = (channels_xy_n[1], channels_xy_n[0])
            spacing_xy_n = (spacing_xy_n[1], spacing_xy_n[0])

        if len(layers) == 0:
            # first layer
            switchboards.append(mdp.hinet.Rectangular2dSwitchboard(in_channels_xy    = (image_x, image_y),
                                                                   field_channels_xy = channels_xy_1,
                                                                   field_spacing_xy  = spacing_xy_1,
                                                                   in_channel_dim    = 1,
                                                                   ignore_cover      = True))
        else:
            switchboards.append(mdp.hinet.Rectangular2dSwitchboard(in_channels_xy    = switchboards[-1].out_channels_xy,
                                                                   field_channels_xy = channels_xy_n,
                                                                   field_spacing_xy  = spacing_xy_n,
                                                                   in_channel_dim    = layers[-1][-1].output_dim,
                                                                   ignore_cover      = True))
    
        flow_nodes = []
        print 'creating layer with %s = %d nodes' % (switchboards[-1].out_channels_xy, switchboards[-1].output_channels)
        for _ in range(switchboards[-1].output_channels):
            nodes = []
            nodes.append(mdp.nodes.IdentityNode(input_dim=switchboards[-1].out_channel_dim))
            if len(layers) == 0:
                # first layer
                nodes.append(mdp.nodes.NoiseNode(noise_args=(0, .01), input_dim=nodes[-1].output_dim))
                nodes.append(mdp.nodes.PCANode(input_dim=nodes[-1].output_dim, output_dim=int(.95*nodes[-1].output_dim), reduce=False))
            if expansion:
                nodes.append(PowerExpansion(input_dim=nodes[-1].output_dim))
                nodes.append(mdp.nodes.NoiseNode(noise_args=(0, .01), input_dim=nodes[-1].output_dim, output_dim=nodes[-1].output_dim))
            nodes.append(node_class(input_dim=nodes[-1].output_dim, output_dim=node_output_dim, **node_kwargs))
            flow_node = mdp.hinet.FlowNode(mdp.Flow(nodes))
            flow_nodes.append(flow_node)
        print '%s: %d -> %d' % (node_class.__name__, nodes[-1].input_dim, nodes[-1].output_dim)
        layers.append(mdp.hinet.Layer(flow_nodes))
        
    hierarchy = []
    for switch, layer in zip(switchboards, layers):
        hierarchy.append(switch)
        hierarchy.append(layer)
        
    if expansion:
        hierarchy.append(PowerExpansion(input_dim=hierarchy[-1].output_dim))
        hierarchy.append(mdp.nodes.NoiseNode(noise_args=(0, .01), input_dim=hierarchy[-1].output_dim, output_dim=hierarchy[-1].output_dim))
    
    hierarchy.append(node_class(input_dim=hierarchy[-1].output_dim, output_dim=output_dim, **node_kwargs))
    flow = mdp.Flow(hierarchy)
    
    print ''
    for node in flow:
        print node.__class__.__name__, node.input_dim, ' -> ', node.output_dim
        
    return flow



def train_model(algorithm, data_train, output_dim, seed, repetition_index, image_shape=None, **kwargs):
    
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
                    output_dim=output_dim,
                    p=kwargs['p'],
                    K=kwargs['K'])
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
    elif algorithm == Algorithms.HiSFA:
        return train_hi_sfa(data_train=data_train,
                            image_shape=image_shape,
                            output_dim=output_dim,
                            expansion=kwargs['expansion'],
                            channels_xy_1=kwargs.get('channels_xy_1', (5,3)),
                            spacing_xy_1=kwargs.get('spacing_xy_1', (3,3)),
                            channels_xy_n=kwargs.get('channels_xy_n', (2,1)),
                            spacing_xy_n=kwargs.get('spacing_xy_n', (2,1)),
                            node_output_dim=kwargs.get('node_output_dim', 10))
    elif algorithm == Algorithms.HiForeCA:
        return train_hi_foreca(data_train=data_train,
                               image_shape=image_shape,
                               output_dim=output_dim,
                               expansion=kwargs['expansion'],
                               channels_xy_1=kwargs.get('channels_xy_1', (5,3)),
                               spacing_xy_1=kwargs.get('spacing_xy_1', (3,3)),
                               channels_xy_n=kwargs.get('channels_xy_n', (2,1)),
                               spacing_xy_n=kwargs.get('spacing_xy_n', (2,1)),
                               node_output_dim=kwargs.get('node_output_dim', 10))
    elif algorithm == Algorithms.HiPFA:
        return train_hi_pfa(data_train=data_train,
                            p=kwargs['p'],
                            K=kwargs['K'],
                            image_shape=image_shape,
                            output_dim=output_dim,
                            expansion=kwargs['expansion'],
                            channels_xy_1=kwargs.get('channels_xy_1', (5,3)),
                            spacing_xy_1=kwargs.get('spacing_xy_1', (5,3)),
                            channels_xy_n=kwargs.get('channels_xy_n', (2,1)),
                            spacing_xy_n=kwargs.get('spacing_xy_n', (2,1)),
                            node_output_dim=kwargs.get('node_output_dim', 10))
    elif algorithm == Algorithms.HiGPFA1:
        return train_hi_gpfa(data_train=data_train,
                             p=kwargs['p'],
                             k=kwargs['k'],
                             iterations=kwargs['iterations'],
                             variance_graph=True,
                             image_shape=image_shape,
                             output_dim=output_dim,
                             expansion=kwargs['expansion'],
                             channels_xy_1=kwargs.get('channels_xy_1', (5,3)),
                             spacing_xy_1=kwargs.get('spacing_xy_1', (5,3)),
                             channels_xy_n=kwargs.get('channels_xy_n', (2,1)),
                             spacing_xy_n=kwargs.get('spacing_xy_n', (2,1)),
                             node_output_dim=kwargs.get('node_output_dim', 10))
    elif algorithm == Algorithms.HiGPFA2:
        return train_hi_gpfa(data_train=data_train,
                             p=kwargs['p'],
                             k=kwargs['k'],
                             iterations=kwargs['iterations'],
                             variance_graph=False,
                             image_shape=image_shape,
                             output_dim=output_dim,
                             expansion=kwargs['expansion'],
                             channels_xy_1=kwargs.get('channels_xy_1', (5,3)),
                             spacing_xy_1=kwargs.get('spacing_xy_1', (5,3)),
                             channels_xy_n=kwargs.get('channels_xy_n', (2,1)),
                             spacing_xy_n=kwargs.get('spacing_xy_n', (2,1)),
                             node_output_dim=kwargs.get('node_output_dim', 10))
    else:
        assert False



@mem.cache
def train_random(data_train, output_dim, seed, repetition_index):
    # rev: 2
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
def train_hi_sfa(data_train, image_shape, output_dim, expansion, channels_xy_1, 
                 spacing_xy_1, channels_xy_n, spacing_xy_n, node_output_dim):
    # rev: 4
    flow = build_hierarchy_flow(image_x=image_shape[1], 
                                image_y=image_shape[0], 
                                output_dim=output_dim, 
                                node_class=mdp.nodes.SFANode, 
                                node_output_dim=node_output_dim,
                                expansion=expansion,
                                channels_xy_1=channels_xy_1,
                                spacing_xy_1=spacing_xy_1,
                                channels_xy_n=channels_xy_n,
                                spacing_xy_n=spacing_xy_n,
                                node_kwargs={})
    flow.train(data_train)
    return flow

 
 
@mem.cache
def train_foreca(data_train, output_dim, seed, repetition_index):
    fargs = update_seed_argument(output_dim=output_dim, seed=seed, repetition_index=repetition_index)
    model = foreca_node.ForeCA(**fargs)
    model.train(data_train)
    return model
 
 
 
@mem.cache
def train_hi_foreca(data_train, image_shape, output_dim, expansion, channels_xy_1, 
                    spacing_xy_1, channels_xy_n, spacing_xy_n, node_output_dim):
    # rev: 0
    flow = build_hierarchy_flow(image_x=image_shape[1], 
                                image_y=image_shape[0], 
                                output_dim=output_dim, 
                                node_class=foreca_node.ForeCA, 
                                node_output_dim=node_output_dim,
                                expansion=expansion,
                                channels_xy_1=channels_xy_1,
                                spacing_xy_1=spacing_xy_1,
                                channels_xy_n=channels_xy_n,
                                spacing_xy_n=spacing_xy_n,
                                node_kwargs={})
    flow.train(data_train)
    return flow

 
 
@mem.cache
def train_pfa(data_train, p, K, output_dim):
    model = PFANodeMDP.PFANode(p=p, k=K, affine=False, output_dim=output_dim)
    model.train(data_train)
    model.stop_training()
    return model
 
 
 
@mem.cache
def train_hi_pfa(data_train, p, K, image_shape, output_dim, expansion, channels_xy_1, 
                 spacing_xy_1, channels_xy_n, spacing_xy_n, node_output_dim):
    # rev: 2
    flow = build_hierarchy_flow(image_x=image_shape[1], 
                                image_y=image_shape[0], 
                                output_dim=output_dim, 
                                node_class=PFANodeMDP.PFANode, 
                                node_output_dim=node_output_dim,
                                expansion=expansion,
                                channels_xy_1=channels_xy_1,
                                spacing_xy_1=spacing_xy_1,
                                channels_xy_n=channels_xy_n,
                                spacing_xy_n=spacing_xy_n,
                                node_kwargs={'p': p, 'k': K})
    flow.train(data_train)
    return flow

 
 
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



@mem.cache
def train_hi_gpfa(data_train, p, k, iterations, variance_graph, image_shape, 
                  output_dim, expansion, channels_xy_1, spacing_xy_1, channels_xy_n, 
                  spacing_xy_n, node_output_dim, neighborhood_graph=False, 
                  weighted_edges=True, causal_features=True):
    # rev: 2
    flow = build_hierarchy_flow(image_x=image_shape[1], 
                                image_y=image_shape[0], 
                                output_dim=output_dim, 
                                node_class=gpfa.gPFA, 
                                node_output_dim=node_output_dim,
                                expansion=expansion,
                                channels_xy_1=channels_xy_1,
                                spacing_xy_1=spacing_xy_1,
                                channels_xy_n=channels_xy_n,
                                spacing_xy_n=spacing_xy_n,
                                node_kwargs={'p': p, 
                                             'k': k,
                                             'iterations': iterations,
                                             'variance_graph': variance_graph,
                                             'neighborhood_graph': neighborhood_graph,
                                             'weighted_edges': weighted_edges,
                                             'causal_features': causal_features})
    flow.train(data_train)
    return flow
 


def calc_projected_data(dataset, algorithm, output_dim, N, repetition_index=None, noisy_dims=0, 
                        use_test_set=True, seed=None, **kwargs):

    n_chunks = 2 if use_test_set else 1
    data_chunks, image_shape = generate_training_data(dataset=dataset, 
                                                      N=N, 
                                                      noisy_dims=noisy_dims,
                                                      n_chunks=n_chunks,
                                                      repetition_index=repetition_index, 
                                                      seed=seed,
                                                      **kwargs)
    
    model = train_model(algorithm=algorithm, 
                        data_train=data_chunks[0], 
                        output_dim=output_dim, 
                        image_shape=image_shape,
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
        
    return projected_data, model, data_chunks, image_shape



def prediction_error(measure, dataset, algorithm, output_dim, N, use_test_set, 
                     repetition_index=None, seed=None, **kwargs):
    
    projected_data, model, data_chunks, _ = calc_projected_data(dataset=dataset, 
                                                                algorithm=algorithm, 
                                                                output_dim=output_dim, 
                                                                N=N, 
                                                                use_test_set=use_test_set, 
                                                                repetition_index=repetition_index, 
                                                                seed=seed, **kwargs)
    
    return prediction_error_on_data(data=projected_data, measure=measure, model=model, data_chunks=data_chunks)
    
    

def prediction_error_on_data(data, measure, model=None, data_chunks=None, **kwargs):

    if measure == Measures.delta:
        return calc_delta(data=data, ndim=False)
    elif measure == Measures.delta_ndim:
        return calc_delta(data=data, ndim=True)
    elif measure == Measures.omega:
        return calc_omega(data=data)
    elif measure == Measures.omega_ndim:
        return calc_omega_ndim(data=data)
    elif measure == Measures.pfa_ndim:
        return calc_autoregressive_error(data=data, 
                                         p=kwargs['p'], 
                                         K=kwargs['K'],
                                         model=model,
                                         data_chunks=data_chunks)
    elif measure == Measures.gpfa:
        return gpfa.calc_predictability_trace_of_avg_cov(x=data, 
                                                         k=kwargs['k'], 
                                                         p=kwargs['p'],
                                                         ndim=False)
    elif measure == Measures.gpfa_ndim:
        return gpfa.calc_predictability_trace_of_avg_cov(x=data, 
                                                         k=kwargs['k'], 
                                                         p=kwargs['p'],
                                                         ndim=True)
    else:
        assert False
    
    
    
def _principal_angle(A, B):
    """A and B must be column-orthogonal.
    Golub: Matrix Computations, 1996
    """
    assert A.ndim == B.ndim == 2
    for i, col in enumerate(A.T):
        A[:,i] /= np.linalg.norm(col)
    for i, col in enumerate(B.T):
        B[:,i] /= np.linalg.norm(col)
    _, S, _ = np.linalg.svd(np.dot(A.T, B))
    return np.arccos(min(S.min(), 1.0))



def principle_angle_models(dataset, algorithm1, algorithm2, dim1, dim2, N, use_test_set, repetition_index=None, seed=None, **kwargs):
     
    if dim1 is None:
        dim1 = dim2
     
    _, model1, _, _ = calc_projected_data(dataset=dataset, 
                                       algorithm=algorithm1, 
                                       output_dim=dim1, 
                                       N=N, 
                                       use_test_set=use_test_set, 
                                       repetition_index=repetition_index, 
                                       seed=seed, **kwargs)
 
    _, model2, _, _ = calc_projected_data(dataset=dataset, 
                                       algorithm=algorithm2, 
                                       output_dim=dim2, 
                                       N=N, 
                                       use_test_set=use_test_set, 
                                       repetition_index=repetition_index, 
                                       seed=seed, **kwargs)
 
    A = None
    if algorithm1 == Algorithms.Random:
        A = model1.U
    elif algorithm1 == Algorithms.SFA:
        A = model1.sf
    elif algorithm1 == algorithm1.ForeCA:
        A = model1.U
    elif algorithm1 == algorithm1.PFA:
        A = model1.Ar
    elif algorithm1 == algorithm1.GPFA1:
        A = model1.U
    elif algorithm1 == algorithm1.GPFA2:
        A = model1.U
    else:
        assert False 
          
    B = None
    if algorithm2 == Algorithms.Random:
        B = model2.U
    elif algorithm2 == Algorithms.SFA:
        B = model2.sf
    elif algorithm2 == Algorithms.ForeCA:
        B = model2.U
    elif algorithm2 == Algorithms.PFA:
        B = model2.Ar
    elif algorithm2 == Algorithms.GPFA1:
        B = model2.U
    elif algorithm2 == Algorithms.GPFA2:
        B = model2.U
    else:
        assert False 
         
    return _principal_angle(A=A, B=B)
    
    
    
def principle_angle_signals(dataset, algorithm1, algorithm2, dim1, dim2, N, use_test_set, repetition_index=None, seed=None, **kwargs):
    
    if dim1 is None:
        dim1 = dim2
     
    signals1, _, _, _ = calc_projected_data(dataset=dataset, 
                                       algorithm=algorithm1, 
                                       output_dim=dim1, 
                                       N=N, 
                                       use_test_set=use_test_set, 
                                       repetition_index=repetition_index, 
                                       seed=seed, **kwargs)
 
    signals2, _, _, _ = calc_projected_data(dataset=dataset, 
                                       algorithm=algorithm2, 
                                       output_dim=dim2, 
                                       N=N, 
                                       use_test_set=use_test_set, 
                                       repetition_index=repetition_index, 
                                       seed=seed, **kwargs)
         
    return _principal_angle(A=signals1, B=signals2)
    
    
    
def calc_delta(data, ndim=False):
    sfa = mdp.nodes.SFANode()
    sfa.train(data)
    sfa.stop_training()
    if ndim:
        return sfa.d
    return np.sum(sfa.d)



def calc_autoregressive_error(data, p, K, model=None, data_chunks=None):
    #W = PFACoreUtil.calcRegressionCoeffRefImp(data=data, p=p)
    print 'extracted_data:', data.shape
    print 'training_data:', data_chunks[0].shape
    print 'pfa.W', model.W.shape
    print 'pfa.W0', model.W0.shape
    print 'pfa.S', model.S.shape
    print 'pfa.Ar', model.Ar.shape
    #print 'pfa.Ar0', model.Ar.shape
    return PFACoreUtil.empiricalRawErrorComponentsRefImp(data=data, W=model.W0.T.dot(model.Ar).T, k=K, srcData=data_chunks[0], W0=model.W0)



def calc_omega(data):
    from foreca.foreca_omega import omega
    return omega(data)



def calc_omega_ndim(data):
    from foreca.foreca_omega import omega
    return [omega(dat) for dat in data.T]



if __name__ == '__main__':
    #set_cachedir(cachedir=None)
    for measure in [Measures.delta, Measures.delta_ndim, Measures.gpfa, Measures.gpfa_ndim]:
        print prediction_error(measure=measure, 
                               dataset=Datasets.Mario_window, 
                               algorithm=Algorithms.GPFA2, 
                               output_dim=2, 
                               N=2000, 
                               k=10, 
                               p=1,
                               iterations=50,
                               seed=0)

