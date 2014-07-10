import numpy as np

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
Axes3D

import mdp

import fpp

from envs.env_cube import EnvCube

if __name__ == '__main__':

    # parameters
    k = 5
    N = 5000
    expansion = 1
    noisy_dims = 50-2
    whitening = True
    chunks = 1
    minimize_variance = False
    normalized_objective = True
    seed = None

    # algorithms
    model = fpp.FPP(output_dim=2,
                    k=k,
                    iterations=5,
                    iteration_dim=10,
                    minimize_variance=minimize_variance,
                    normalized_objective=normalized_objective)

    # learn
    
    # current algorithm
    print "%s" % (model.__class__.__name__)

    # for every chunk
    for c in range(chunks):

        # data
        env = EnvCube(seed=seed)
        data0, _, _ = env.do_random_steps(num_steps=N)
        print data0.shape
        
        # colors
        labels = np.zeros(data0.shape[0], dtype=int)
        for i, dat in enumerate(data0):
            x = int(5 * dat[0] - 1e-6)
            y = int(5 * dat[1] - 1e-6)
            labels[i] = (x + y) % 2

        # add noisy dim
        R = np.random.RandomState(seed=seed)
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
    pyplot.scatter(x=data2[:,0], y=data2[:,1], c=labels, s=50, cmap=pyplot.cm.get_cmap('Blues'))#, edgecolor='None')
    #for t in range(N-1):
    #    pyplot.plot(data2[t:t+2,0], data2[t:t+2,1])
    pyplot.title(model.__class__.__name__)

    # show plot
    print 'finish'
    pyplot.show()
