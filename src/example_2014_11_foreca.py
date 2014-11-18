import numpy as np

from subprocess import call

from envs.env_swiss_roll import EnvSwissRoll

import gpfa
import plotter



def experiment_foreca(N, k, noisy_dims, measure='var'):
    
    # generate and save data
    env = EnvSwissRoll()
    data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, chunks=2)
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
    if measure == 'var':
        return gpfa.calc_predictability_var(result, k)
    else:
        return gpfa.calc_predictability_star(result, k)
    return



def main():
    plotter.plot(experiment_foreca, 
                 N=[100, 1000, 2000, 3000, 4000], 
                 k=30, 
                 noisy_dims=2, #[0, 1, 2, 5, 10],#, 20, 50],
                 processes=None, 
                 repetitions=10)



if __name__ == '__main__':
    main()
