import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

import experiment_base as eb



def n_dimensions(**kwargs):
    return eb.generate_training_data(**kwargs)[0].shape[1] 



def main():
    
    datasets = [(eb.Datasets.EEG, 2000, 1.),
                (eb.Datasets.Face, 1965, 1.),
                (eb.Datasets.Mario_window, 2000, 1.),
                (eb.Datasets.MEG, 375, 1.),
                (eb.Datasets.RatLab, 2000, .5),
                (eb.Datasets.Tumor, 500, .25)]
    
    for i, (dataset, N, scaling) in enumerate(datasets):
        plt.subplot(3, 2, i+1)
        ep.plot(n_dimensions, data=dataset, 
                              N=N,
                              scaling=scaling, 
                              chunks=1,
                              keep_variance=np.arange(.7, 1., .01),
                              show_plot=False,
                              seed=0)
    plt.show()



if __name__ == '__main__':
    main()
        