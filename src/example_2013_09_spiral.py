import numpy as np

from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

import future_preserving_map

from studienprojekt.env_swiss_roll import EnvSwissRoll

if __name__ == '__main__':
    
    N = 5000
    env = EnvSwissRoll(sigma=0.5)
    data, _, _ = env.do_random_steps(num_steps=N)
    
    # add noisy dim
    data = np.insert(data, 2, axis=1, values=np.random.random(N))
    
    # learn
    fpm = future_preserving_map.FuturePreservingMap(output_dim=9, k=10, normalized_laplacian=True, neighbor_edges=True)
    fpm.train(x=data)
    fpm.stop_training()
    
    # plot
    for i in range(9):
        ax = pyplot.subplot(3, 3, i+1, projection='3d')
        s = ax.scatter3D(xs=data[:,0], ys=data[:,1], zs=data[:,2], s=50, c=fpm.U[:,i], edgecolor='None')
        pyplot.title(fpm.E[i])
        #pyplot.colorbar(s)
     
    pyplot.show()