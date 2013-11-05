"""
An example that applies different dimensionality reductions on a swiss role
hidden in noise.
"""

import numpy as np

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
Axes3D

import mdp

import fpp

from studienprojekt.env_swiss_roll import EnvSwissRoll

if __name__ == '__main__':

    # parameters
    k = 5
    N = 5000
    expansion = 1
    noisy_dims = 48
    whitening = True
    normalized_laplacian = True
    neighbor_edges = True
    chunks = 1

    # algorithms
    models = []
    models.append(mdp.nodes.SFANode())
    #models.append(mdp.nodes.LLENode(k=k))
    #models.append(mdp.nodes.HLLENode(k=55))
    #models.append(future_preserving_map.FuturePreservingMap(output_dim=2,
    models.append(fpp.GraphSFA(output_dim=2,
                                k=k,
                                normalized_laplacian=normalized_laplacian,
                                neighbor_edges=neighbor_edges))
    models.append(fpp.FPPLinear(output_dim=2,
                                k=k,
                                normalized_laplacian=normalized_laplacian,
                                neighbor_edges=neighbor_edges))

    # learn
    for j, model in enumerate(models):

        # current algorithm
        print "%s" % (model.__class__.__name__)

        # for every chunk
        for _ in range(chunks):

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

            # train
            model.train(data)

        # plot
        data2 = model.execute(data)
        pyplot.subplot(1, 3, j+1)
        pyplot.scatter(x=data2[:,0], y=data2[:,1], c=labels, s=50, edgecolor='None')
        pyplot.title(model.__class__.__name__)

    # show plot
    pyplot.show()
