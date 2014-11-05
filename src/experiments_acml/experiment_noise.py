import numpy as np

from matplotlib import pyplot

import mdp

import PFANodeMDP

import gpfa

from envs.env_cube import EnvCube
from envs.env_oscillator import EnvOscillator

if __name__ == '__main__':

    # parameters
    k = 30
    N = 1000
    noisy_dims = 2
    whitening = True
    iterations = 5
    constraint_optimization = True
    seed = None
    
    # prepare data and noise
    environments = [EnvCube(seed=seed)]#, EnvOscillator(seed=seed)]
    R = np.random.RandomState(seed=seed)
    noise = [R.rand(N, noisy_dims), R.randint(0, 2, size=(N, noisy_dims))]
    
    for e, env in enumerate(environments):

        # data
        data, _, _ = env.do_random_steps(num_steps=N)
        
        # color data (checkerboard)
        labels = np.zeros(N, dtype=int)
        for i, dat in enumerate(data):
            x = int(4 * dat[0] - 1e-6)
            y = int(4 * dat[1] - 1e-6)
            labels[i] = (x + y) % 2

        # plot data sets            
        pyplot.subplot(3, 2, e+1)
        pyplot.scatter(x=data[:,0], y=data[:,1], c=labels, s=50, linewidth='0.5', cmap=pyplot.cm.get_cmap('Blues'))
        pyplot.title(['random walk in unit square', 'corners of unit square'][e])

        # add noisy dims
        data = np.hstack([data, noise[e]])

        # whitening
        if whitening:
            whitening_node = mdp.nodes.WhiteningNode()
            whitening_node.train(data)
            data = whitening_node.execute(data)
            
        # save data as CSV
        np.savetxt('data.csv', data, delimiter=',', fmt='%10.8f')
        
        for i in range(4):
            pyplot.subplot(2,2,i+1)
            pyplot.plot(data[:,i])
            #pyplot.plot(np.fft.rfft(data[:,i]))
        pyplot.show()

        # algorithms
        models = []
        models.append(gpfa.gPFA(output_dim=2,
                                k=k,
                                iterations=iterations,
                                iteration_dim=2,
                                variance_graph=False,
                                neighborhood_graph=True,
                                constraint_optimization=constraint_optimization))
        models.append(gpfa.LPP(output_dim=2, k=k))
        #models.append(PFANodeMDP.PFANode(p=2, k=4, affine=False, output_dim=2))

        # train & plot
        for m, model in enumerate(models):
    
            # current algorithm
            print "%s" % (model.__class__.__name__)
        
            # train
            model.train(data)
            result = model.execute(data)
            print model.U
        
            # plot
            pyplot.subplot(3, 2, e+2*m+3)
            pyplot.scatter(x=result[:,0], y=result[:,1], c=labels, s=50, linewidth='0.5', cmap=pyplot.cm.get_cmap('Blues'))
            pyplot.title('%s, data set %d' % (model.__class__.__name__, e+1))

    # show plot
    print 'finish'
    pyplot.show()
