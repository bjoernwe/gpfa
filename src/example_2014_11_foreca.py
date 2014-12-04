import numpy as np
import sys

from subprocess import call

sys.path.append('/home/weghebvc/workspace/Worldmodel/src/')
from envs.env_sine import EnvSine
from envs.env_swiss_roll import EnvSwissRoll

import gpfa
import plotter



def experiment_foreca(N, k, noisy_dims, measure='var'):
    
    # generate and save data
    #env = EnvSine()
    env = EnvSwissRoll()
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, chunks=2)
    data_train_mean = np.mean(data_train[0], axis=0) 
    data_train = (data_train[0] - data_train_mean, None, None)
    data_test  = (data_test[0] - data_train_mean, None, None)
    run_id = str(np.random.randint(100000, 1000000))
    np.savetxt("example_2014_11_foreca_train_%s.csv" % run_id, data_train[0], delimiter=",")
    
    # run R script and load result (extraction matrix)
    call(['Rscript', 'example_2014_11_foreca.r', run_id])
    W = np.loadtxt('example_2014_11_foreca_result_%s.csv' % run_id)
    
    # clean files
    call(['rm', 'example_2014_11_foreca_train_%s.csv' % run_id])
    call(['rm', 'example_2014_11_foreca_result_%s.csv' % run_id])
    
    # evaluate result
    result = data_test[0].dot(W)
    #result = np.array(result, ndmin=2).T
    if measure == 'var':
        return gpfa.calc_predictability_var(result, k)
    else:
        return gpfa.calc_predictability_star(result, k)
    return



def main():
    #print experiment_foreca(N=2000, k=30, noisy_dims=2, measure='var')
    plotter.plot(experiment_foreca, 
                 N=2000, #[100, 200, 500, 1000, 2000, 3000, 4000, 5000], 
                 k=30, 
                 noisy_dims=[1, 2, 5, 10, 20, 50, 100],#, 200, 300, 400, 500],
                 processes=None, 
                 repetitions=10)



if __name__ == '__main__':
    main()
