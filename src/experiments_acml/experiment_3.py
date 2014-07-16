import numpy as np

from matplotlib import pyplot

import mdp

import fpp

from envs.env_swiss_roll_3d import EnvSwissRoll3D

if __name__ == '__main__':

    # parameters
    k = 5
    N = 5000
    whitening = True
    expansion_degree = 2
    seed = None
    
    # data
    env = EnvSwissRoll3D()
    data_raw, _, labels = env.do_random_steps(num_steps=N)
    
    # expansion
    expansion = mdp.nodes.PolynomialExpansionNode(degree=expansion_degree)
    data = expansion.execute(data_raw)

    # whitening
    if whitening:
        whitening_node = mdp.nodes.WhiteningNode()
        whitening_node.train(data)
        data = whitening_node.execute(data)

    # algorithms
    model = fpp.FPP(output_dim=2,
                    k=k,
                    iterations=1,
                    iteration_dim=2,
                    minimize_variance=False,
                    normalized_objective=True)

    # train
    model.train(data)
    result = model.execute(data)

    # plot
    fig = pyplot.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    s = ax.scatter(data_raw[1:,0], data_raw[1:,2], data_raw[1:,1], c=labels, s=30, linewidth='0.2', cmap=pyplot.cm.get_cmap('Blues'))
    s.set_edgecolors = s.set_facecolors = lambda *args:None
    ax = fig.add_subplot(1, 2, 2)
    ax.scatter(x=result[1:,0], y=result[1:,1], c=labels, s=50, linewidth='0.5', cmap=pyplot.cm.get_cmap('Blues'))

    # show plot
    print 'finish'
    pyplot.show()
