import mdp
import numpy as np

#from matplotlib import pyplot as plt

import easyexplot as eep

#from envs import env_swiss_roll

import gpfa

from foreca import foreca_node
from foreca import foreca_omega


def predictability(measure, model, output_dim, k=10, iterations=20):
    #env = env_swiss_roll.EnvSwissRoll()
    #X, _, _ = env.generate_training_data(num_steps=1400, noisy_dims=6, whitening=True)[0]
    X = np.array(np.loadtxt('equityFunds.csv', delimiter=';', dtype='str')[1:,1:], dtype=float)
    
    if True:
        whitening = mdp.nodes.WhiteningNode()
        whitening.train(X)
        X = whitening.execute(X)

    if model == 'SFA':
        model = mdp.nodes.SFANode(output_dim=output_dim)
    elif model == 'gPFA':
        model = gpfa.gPFA(output_dim=output_dim, k=k, iterations=20, iteration_dim=output_dim)
    elif model == 'ForeCA':
        model = foreca_node.ForeCA(output_dim=output_dim)
    
    model.train(X)
    Y = model.execute(X)
    
    if measure == 'omega':
        return np.mean(foreca_omega.omega(Y))
    else:
        return gpfa.calc_predictability_avg_det_of_cov(Y, k)



def main():
    
    eep.plot(predictability,
             measure='avg_det_cov', 
             model=['SFA', 'gPFA', 'ForeCA'], 
             output_dim=range(1,9), 
             k=20,
             iterations=30,
             cachedir='.',
             processes=None)



if __name__ == '__main__':
    main()
    