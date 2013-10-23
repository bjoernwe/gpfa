import numpy as np

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

import mdp

import fpp

from studienprojekt.env_swiss_roll import EnvSwissRoll


if __name__ == '__main__':
    
    N = 5000
    noisy_dims = 3
    env = EnvSwissRoll(sigma=2.5)
    data, _, _ = env.do_random_steps(num_steps=N)
    
    # add noisy dim
    for i in range(noisy_dims):
        noise_complete = 1. * np.random.random(N)
        data = np.insert(data, 2, axis=1, values=noise_complete)
    
    # whitening
    whitening = mdp.nodes.WhiteningNode(reduce=True)
    whitening.train(data)
    data = whitening.execute(data)
    
    # learn
    fpm = fpp.FPP(output_dim=4, k=10, normalized_laplacian=True, neighbor_edges=True)
    fpm.train(x=data)
    fpm.stop_training()
    
    # plot
    for i in range(4):
        ax = pyplot.subplot(2, 2, i+1, projection='3d')
        s = ax.scatter3D(xs=data[:,0], ys=data[:,1], zs=data[:,2], s=50, c=fpm.U[:,i], edgecolor='None')
        pyplot.title(fpm.E[i])
     
    pyplot.show()
    