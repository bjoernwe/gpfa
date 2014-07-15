import numpy as np

from matplotlib import pyplot

import mdp

import fpp

from envs import env_swiss_roll_3d

if __name__ == '__main__':

    # parameters
    k = 5
    N = 5000
    iterations = 1
    seed = None
    
    # data
    env = env_swiss_roll_3d.EnvSwissRoll3D()
    data, _, labels = env.do_random_steps(num_steps=N)

    # model    
    model = fpp.FPPnl(output_dim=2,
                      k=k,
                      iterations=iterations,
                      iteration_dim=10,
                      minimize_variance=False,
                      normalized_objective=True)

    # train & plot
            
    # current algorithm
    print "%s" % (model.__class__.__name__)

    # train
    model.train(data)
    model.stop_training()
    print model.U
    #result = model.execute(data)

    # plot
    #pyplot.subplot(1, 4, 2*e+m+1)
    pyplot.scatter(x=model.U[:-1,1], y=model.U[:-1,2], c=labels, s=50, cmap=pyplot.cm.get_cmap('Blues'))
    pyplot.title(model.__class__.__name__)

    # show plot
    print 'finish'
    pyplot.show()
