import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

import experiments.experiment_base as eb

import plot



def plot_experiment(N=10000, k=2, p=3, K=0, keep_variance=.99, iterations=50, output_dim=2, 
                    include_random=False, include_sfa=True, include_foreca=False, 
                    include_gcfa=True, x_offset=0, y_label=True, legend=False):
    plot.plot_experiment(dataset=eb.Datasets.EEG2_stft_128, 
                         N=N, 
                         k=k, 
                         p=p, 
                         K=K, 
                         noisy_dims=0,
                         keep_variance=keep_variance, 
                         iterations=iterations, 
                         output_dim=output_dim,
                         repetitions=1, 
                         include_random=include_random,
                         include_sfa=include_sfa, 
                         include_foreca=include_foreca, 
                         include_gcfa=include_gcfa,
                         x_offset=x_offset, 
                         y_label=y_label, 
                         legend=legend,
                         seed=0)
    
    
    
def main():
    plt.figure()
    plot_experiment(keep_variance=list(np.arange(.90, .995, .01)))
    plt.figure()
    plot_experiment(N=[2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
    plt.figure()
    plot_experiment(iterations=[20, 40, 60, 80, 100, 150])
    plt.figure()
    plot_experiment(k=range(2,5) + range(5,21,5))
    plt.figure()
    plot_experiment(p=[1,2,3,4])
    plt.show()


if __name__ == '__main__':
    main()
