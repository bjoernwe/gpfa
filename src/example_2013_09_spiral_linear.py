import numpy as np

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

import mdp

import fpp

from studienprojekt.env_swiss_roll import EnvSwissRoll

if __name__ == '__main__':
    
    # parameters
    k = 5
    N = 5000
    expansion = 1
    noisy_dims = 50
    whitening = True
    #noisy_steps = 3
    
    # data
    env = EnvSwissRoll(sigma=0.5)
    data0, _, labels = env.do_random_steps(num_steps=N)

    # add noisy dim
    for i in range(noisy_dims):
        noise_complete = 1. * np.random.random(N)
        data0 = np.insert(data0, 2, axis=1, values=noise_complete)

    # expansion        
    expansion_node = mdp.nodes.PolynomialExpansionNode(degree=expansion)
    data = expansion_node.execute(data0)
    
    # whitening
    if whitening:
        whitening_node = mdp.nodes.WhiteningNode()
        whitening_node.train(data)
        data = whitening_node.execute(data)
    
    # algorithms
    models = []
    models.append(mdp.nodes.SFANode())
    models.append(mdp.nodes.LLENode(k=k))
    #models.append(mdp.nodes.HLLENode(k=55))
    #models.append(future_preserving_map.FuturePreservingMap(output_dim=2, 
    models.append(fpp.FPPLinear(output_dim=2, 
                                k=k, 
                                normalized_laplacian=False, 
                                neighbor_edges=False))
    
    # learn
    for j, model in enumerate(models):
        print "(%d)" % (j)
    
        model.train(data)
        model.stop_training()
    
        # plot
        data2 = model.execute(data)
        pyplot.subplot(1, 3, j+1)
        pyplot.scatter(x=data2[:,0], y=data2[:,1], c=labels, s=50, edgecolor='None')
        pyplot.title(model.__class__.__name__)
        
    pyplot.show()
    