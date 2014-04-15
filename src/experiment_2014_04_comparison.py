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

#import PFANodeMDP
#import PFANodeMDPRefImp

from envs.env_ribbon import EnvRibbon

from studienprojekt.env_swiss_roll import EnvSwissRoll

if __name__ == '__main__':

    # parameters
    k = 10
    N = 2000
    expansion = 1
    noisy_dims = 20-2
    whitening = True
    neighbor_graph = False
    chunks = 1

    # algorithms
    models = []
    #models.append(fpp.gPFA(output_dim=2, k=k, iterations=1, iteration_dim=5, minimize_variance=True))
    models.append(fpp.FPP(output_dim=2,
                          k=k,
                          iterations=1,
                          iteration_dim=5,
                          minimize_variance=True,
                          normalized_objective=False,
                          preserve_past=False,
                          neighbor_graph=False))
    models.append(fpp.FPP(output_dim=2,
                          k=k,
                          iterations=1,
                          iteration_dim=5,
                          minimize_variance=False,
                          normalized_objective=False,
                          preserve_past=False,
                          neighbor_graph=False))
    #models.append(fpp.gPFA(output_dim=2, k=k, iterations=1, iteration_dim=5, minimize_variance=False))
    models.append(fpp.FPP(output_dim=2,
                          k=k,
                          iterations=1,
                          iteration_dim=5,
                          minimize_variance=True,
                          normalized_objective=True,
                          preserve_past=False,
                          neighbor_graph=False))
    models.append(fpp.FPP(output_dim=2,
                          k=k,
                          iterations=1,
                          iteration_dim=5,
                          minimize_variance=False,
                          normalized_objective=True,
                          preserve_past=False,
                          neighbor_graph=False))

    # learn
    for j, model in enumerate(models):

        # current algorithm
        print "%s" % (model.__class__.__name__)

        # for every chunk
        for _ in range(chunks):

            # data
            #env = EnvSwissRoll(sigma=0.5, seed=None)
            env = EnvRibbon(step_size=2, seed=None)
            data0, _, labels = env.do_random_steps(num_steps=N)
            print data0.shape, labels.shape

            # add noisy dim
            R = np.random.RandomState(seed=0)
            for i in range(noisy_dims):
                noise_complete = 1. * R.rand(N)
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
        ax = pyplot.subplot(1, len(models), j+1)
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        pyplot.scatter(x=data2[:-1,0], y=data2[:-1,1], c=labels, s=50, edgecolor='None')
        pyplot.title("%s\n graph=%s\n normalized=%s" % (model.__class__.__name__, 'variance' if model.minimize_variance else 'star', 'True' if model.normalized_objective else 'False'))

    # show plot
    pyplot.show()
