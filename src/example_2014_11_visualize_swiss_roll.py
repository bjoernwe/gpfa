import numpy as np
import sys

from matplotlib import pyplot as plt

import mdp

import gpfa

sys.path.append('/home/weghebvc/workspace/Worldmodel/src/')
from envs.env_oscillator import EnvOscillator
from envs.env_swiss_roll import EnvSwissRoll



def main():

    noisy_dims = [5, 5, 5, 5]
    
    for i, n in enumerate(noisy_dims):
    
        env = EnvOscillator()
        #env = EnvSwissRoll()
        X, _, labels = env.generate_training_data(num_steps=2000, 
                                                  noisy_dims=n, 
                                                  whitening=True, 
                                                  chunks=1)[0]
        model = gpfa.gPFA(output_dim=2, k=30, iterations=50)
        #model = mdp.nodes.SFANode(output_dim=2)
        model.train(X)
        Y = model.execute(X)
    
        plt.subplot(2, 2, i+1)
        plt.title(gpfa.calc_predictability_det_var(data=Y, k=20))
        plt.scatter(Y[1:,0], Y[1:,1], s=70, c=labels, linewidths=0.1)
        plt.gray()
        
    plt.show()                     
    return



if __name__ == '__main__':
    main()
