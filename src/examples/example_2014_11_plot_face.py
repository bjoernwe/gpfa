import numpy as np
import sys

from matplotlib import pyplot as plt

import mdp

sys.path.append('/home/weghebvc/workspace/PFAStefan/src/')
import PFANodeMDP

import gpfa

sys.path.append('/home/weghebvc/workspace/Worldmodel/src/')
from envs.env_face import EnvFace



def experiment(algorithm, N, k, p, K, iterations, noisy_dims, variance_graph, neighborhood_graph=False, iteration_dim=2, data='swiss_roll', measure='var'):
    
    assert algorithm in ['sfa', 'pfa', 'gpfa']
    assert data in ['swiss_roll', 'face']
    assert measure in ['var', 'star']
    
    # generate data
    env = EnvFace()
    data_train, data_test = env.generate_training_data(num_steps=[1500, 465], noisy_dims=noisy_dims, whitening=False, chunks=2)
    pca = mdp.nodes.PCANode(output_dim=.99)
    pca.train(data_train[0])
    data_train = pca.execute(data_train[0])
    data_test = pca.execute(data_test[0])
    print data_train.shape
    
    whitening = mdp.nodes.WhiteningNode()
    whitening.train(data_train)
    data_train = whitening.execute(data_train)
    data_test = whitening.execute(data_test)
    
    # train algorithm
    if algorithm == 'sfa':
        model = mdp.nodes.SFANode(output_dim=2)
    elif algorithm == 'pfa':
        model = PFANodeMDP.PFANode(p=p, k=K, affine=False, output_dim=2)
    elif algorithm == 'gpfa':
        model = gpfa.gPFA(k=k, 
                          output_dim=2, 
                          iterations=iterations, 
                          iteration_dim=iteration_dim, 
                          variance_graph=variance_graph, 
                          neighborhood_graph=False)
    else:
        return

    # train and evaluate test data    
    model.train(data_train)
    result = model.execute(data_test)
    return (result, data_test[1], data_test[2])



def main():
    
    # parameters
    algorithms = ['sfa', 'pfa', 'gpfa']
    p = 2
    K = 8
    k = 50 # [2, 3, 5, 10, 20, 30, 40, 50, 100]
    N = 2000 # [1000, 2000, 3000, 4000, 5000] 1965
    noisy_dims = 0 # [0, 100, 200, 300, 400, 500, 600]#[0, 50, 100, 150, 200, 250, 300, 350, 400]
    iterations = 60 # [1, 10, 20, 30]#, 40, 50, 100]
    iteration_dim = 2
    data = 'face'
    
    for a, algorithm in enumerate(algorithms):
        result = experiment(algorithm=algorithm,
                            k=k,
                            N=N,
                            p=p,
                            K=K,
                            iterations=iterations,
                            noisy_dims=noisy_dims,
                            iteration_dim=iteration_dim,
                            variance_graph=False,
                            neighborhood_graph=False,
                            data=data,
                            measure='var')
        plt.subplot(2, 2, a+1)
        plt.title('%s (%3.4f)' % (algorithm, gpfa.calc_predictability_avg_det_of_cov(result[0], k=k)))
        plt.scatter(x=result[0][:,0], y=result[0][:,1])
        
    plt.show()
    return



if __name__ == '__main__':
    main()
