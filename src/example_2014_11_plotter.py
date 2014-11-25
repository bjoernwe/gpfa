import sys

import mdp

sys.path.append('/home/weghebvc/workspace/PFAStefan/src/')
import PFANodeMDP

import gpfa
import plotter

sys.path.append('/home/weghebvc/workspace/Worldmodel/src/')
from envs.env_face import EnvFace
from envs.env_swiss_roll import EnvSwissRoll



def experiment(algorithm, N, k, p, K, iterations, noisy_dims, variance_graph, iteration_dim=2, data='swiss_roll', measure='var'):
    
    assert algorithm in ['sfa', 'pfa', 'gpfa']
    assert data in ['swiss_roll', 'face']
    assert measure in ['var', 'star']
    
    # generate data
    if data == 'swiss_roll':
        env = EnvSwissRoll()
        data_train, data_test = env.generate_training_data(num_steps=N, noisy_dims=noisy_dims, whitening=True, chunks=2)
    elif data == 'face':
        env = EnvFace()
        data_train, data_test = env.generate_training_data(num_steps=[1500, 465], noisy_dims=noisy_dims, whitening=True, chunks=2)
    else:
        return
    
    # train algorithm
    if algorithm == 'sfa':
        model = mdp.nodes.SFANode(output_dim=2)
    elif algorithm == 'pfa':
        model = PFANodeMDP.PFANode(p=p, k=K, affine=False, output_dim=2)
    elif algorithm == 'gpfa':
        model = gpfa.gPFA(k=k, output_dim=2, iterations=iterations, iteration_dim=iteration_dim, variance_graph=variance_graph)
    else:
        return
    
    model.train(data_train[0])
    
    # evaluate solution
    result = model.execute(data_test[0])
    if measure == 'var':
        return gpfa.calc_predictability_var(result, k)
    else:
        return gpfa.calc_predictability_star(result, k)
    return



def main():
    
    # parameters
    algorithms = ['sfa', 'pfa', 'gpfa']
    p = 2
    K = 8
    k = 20 # [2, 3, 5, 10, 20, 30, 40, 50, 100]
    N = 2000 #[1000, 2000, 3000, 4000, 5000] 1965
    noisy_dims = [0, 100, 200, 300, 400, 500, 600]#[0, 50, 100, 150, 200, 250, 300, 350, 400]
    iterations = 40 # [1, 10, 20, 30]#, 40, 50, 100]
    iteration_dim = 2
    data = 'swiss_roll'
    
    # plotter arguments
    processes = None
    repetitions = 50
    save_results = False

#     print experiment(algorithm='pfa',
#                      k=k,
#                      N=N,
#                      p=p,
#                      K=K,
#                      iterations=iterations,
#                      noisy_dims=200,
#                      iteration_dim=iteration_dim,
#                      variance_graph=False,
#                      data=data,
#                      measure='var')
                     
    for i, a in enumerate(algorithms):
        is_last_iteration = (i==len(algorithms)-1)    
        plotter.plot(experiment,
                     algorithm=a,
                     k=k,
                     N=N,
                     p=p,
                     K=K,
                     iterations=iterations,
                     noisy_dims=noisy_dims,
                     iteration_dim=iteration_dim,
                     variance_graph=False,
                     data=data,
                     processes=processes,
                     repetitions=repetitions,
                     measure='var',
                     legend=algorithms,
                     save_result=save_results,
                     save_plot=is_last_iteration,
                     show_plot=is_last_iteration)

#     plotter.plot(experiment,
#                  algorithm=algorithms,
#                  k=k,
#                  N=N,
#                  p=p,
#                  K=K,
#                  iterations=iterations,
#                  noisy_dims=noisy_dims,
#                  iteration_dim=iteration_dim,
#                  variance_graph=False,
#                  data=data,
#                  processes=processes,
#                  repetitions=repetitions,
#                  measure='var',
#                  legend=algorithms,
#                  save_result=save_results,
#                  save_plot=True,
#                  show_plot=True)
    
    return



if __name__ == '__main__':
    main()
