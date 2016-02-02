import numpy as np
import sys

from matplotlib import pyplot as plt

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

import experiments.experiment_base as eb

import plot



def plot_experiment(N=1500, k=2, p=2, K=0, keep_variance=.98, iterations=50, output_dim=2, include_random=False, include_foreca=False, include_gcfa=True, x_offset=0, y_label=True, legend=False):
    plot.plot_experiment(data='face', 
                         N=N, 
                         k=k, 
                         p=p, 
                         P=p,
                         K=K, 
                         noisy_dims=0,
                         keep_variance=keep_variance, 
                         iterations=iterations, 
                         output_dim=output_dim,
                         repetitions=1, 
                         include_random=include_random, 
                         include_foreca=include_foreca, 
                         include_gcfa=include_gcfa, 
                         x_offset=x_offset, 
                         y_label=y_label, 
                         legend=legend,
                         seed=0)
    
    
    
def main():
    #plt.figure()
    plt.subplot(2, 2, 1)
    plot_experiment(keep_variance=list(np.arange(.80, 1.01, .02)))
    plt.subplot(2, 2, 2)
    #plot_experiment(N=[1500, 1750, 2000, 2250, 2500])
    #plt.subplot(2, 2, 3)
    #plot_experiment(iterations=[20, 40, 60, 80, 100, 150])
    #plt.subplot(1, 2, 2)
    plot_experiment(k=range(2,5) + range(5,21,5))
    plt.show()


def main_plot():
    plt.figure()
    plot_experiment(keep_variance=list(np.arange(.80, 1.01, .02)), output_dim=3)
    plt.figure()
    plot_experiment(keep_variance=list(np.arange(.80, 1.01, .02)), output_dim=6)
    plt.figure()
    plot_experiment(keep_variance=list(np.arange(.80, 1.01, .02)), output_dim=9)
    plt.figure()
    plot_experiment(keep_variance=list(np.arange(.80, 1.01, .02)), output_dim=12, legend=True)
    plt.show()
    
    
if __name__ == '__main__':
    main()
    #main_plot()
