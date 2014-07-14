import numpy as np

from matplotlib import pyplot

import mdp

import fpp

from envs.env_cube import EnvCube
from envs.env_oscillator import EnvOscillator

if __name__ == '__main__':

    # parameters
    k = 5
    N = 5000
    expansion = 1
    noisy_dims = 50-2
    whitening = True
    iterations = 5
    minimize_variance = False
    normalized_objective = True
    seed = None
    
    environments = [EnvCube(seed=seed), EnvOscillator(seed=seed)]
    
    for e, env in enumerate(environments):

        # data
        data0, _, _ = env.do_random_steps(num_steps=N)
        labels = np.zeros(data0.shape[0], dtype=int)
        for i, dat in enumerate(data0):
            x = int(4 * dat[0] - 1e-6)
            y = int(4 * dat[1] - 1e-6)
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

        # algorithms
        models = []
        models.append(fpp.FPP(output_dim=2,
                              k=k,
                              iterations=iterations,
                              iteration_dim=10,
                              minimize_variance=minimize_variance,
                              normalized_objective=normalized_objective))
        models.append(fpp.LPP(output_dim=2, k=k))

        # train & plot        
        for m, model in enumerate(models):
    
            # current algorithm
            print "%s" % (model.__class__.__name__)
        
            # train
            model.train(data)
            result = model.execute(data)
        
            # plot
            pyplot.subplot(1, 4, 2*e+m+1)
            pyplot.scatter(x=result[:,0], y=result[:,1], c=labels, s=50, cmap=pyplot.cm.get_cmap('Blues'))
            pyplot.title(model.__class__.__name__)

    # show plot
    print 'finish'
    pyplot.show()
