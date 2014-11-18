import itertools
import numpy as np
import scipy.spatial.distance
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/PFAStefan/src/')
import PFANodeMDP

import gpfa
import plotter

sys.path.append('/home/weghebvc/workspace/Worldmodel/src/')
from envs.env_swiss_roll import EnvSwissRoll



def experiment_gpfa(N, k, iterations, noisy_dims, iteration_dim, variance_graph, measure='var'):
    
    assert measure in ['var', 'star']
    
    # generate data    
    env = EnvSwissRoll()
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, chunks=2)
    
    # train algorithm
    model = gpfa.gPFA(k=k, output_dim=2, iterations=iterations, iteration_dim=iteration_dim, variance_graph=variance_graph)
    model.train(data_train[0])
    
    # evaluate solution
    result = model.execute(data_test[0])
    if measure == 'var':
        return calc_predictability_var(result, k)
    else:
        return calc_predictability_star(result, k)
    return



def experiment_pfa(N, k, noisy_dims, p=2, K=8, measure='var'):
    
    assert measure in ['var', 'star']
    
    # generate data
    env = EnvSwissRoll()
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, chunks=2)
    
    # train algorithm
    model = PFANodeMDP.PFANode(p=p, k=K, affine=False, output_dim=2)
    model.train(data_train[0])
    
    # evaluate solution
    result = model.execute(data_test[0])
    if measure == 'var':
        return calc_predictability_var(result, k)
    else:
        return calc_predictability_star(result, k)
    return


def experiment_foreca(N, k, noisy_dims):
    
    # generate and save data
    env = EnvSwissRoll()
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, chunks=2)
    np.savetxt("example_2014_11_plotter_foreca_train.csv", data_train[0], delimiter=",")
    np.savetxt("example_2014_11_plotter_foreca_test.csv", data_test[0], delimiter=",")
    return



def main():
    
    # parameters
    k = 30 #[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    N = 2000 #[1000, 2000, 3000, 4000, 5000]
    iterations = 10 #[1, 10, 20, 30, 40, 50]
    processes = None
    iteration_dim = 50 #[2, 5, 10, 20, 50, 100, 200]
    noisy_dims = [0, 100, 200] #[0, 50, 100, 150, 200, 250, 300, 350, 400]
    repetitions = 10
    
    # PFA
#     plotter.plot(experiment_pfa,
#                  k=k,
#                  N=N,
#                  noisy_dims=noisy_dims,
#                  processes=processes,
#                  repetitions=repetitions,
#                  measure='star',
#                  save_plot=False,
#                  show_plot=False)
    
    # gPFA 1
    plotter.plot(experiment_gpfa,
                 k=k,
                 N=N,
                 iterations=iterations,
                 noisy_dims=noisy_dims,
                 iteration_dim=iteration_dim,
                 variance_graph=True,
                 processes=processes,
                 repetitions=repetitions,
                 measure='star',
                 save_plot=False,
                 show_plot=False)
    
    # gPFA 2
    plotter.plot(experiment_gpfa,
                 k=k,
                 N=N,
                 iterations=iterations,
                 noisy_dims=noisy_dims,
                 iteration_dim=iteration_dim,
                 variance_graph=False,
                 processes=processes,
                 repetitions=repetitions,
                 measure='star',
                 save_plot=True,
                 show_plot=False)
    
    #plt.legend(['PFA', 'gPFA'])
    #plt.show()
    return



if __name__ == '__main__':
    main()
