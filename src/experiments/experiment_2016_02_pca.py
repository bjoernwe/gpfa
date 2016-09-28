import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

import experiment_base as eb



def n_dimensions(**kwargs):
    return eb.generate_training_data(**kwargs)[0][0].shape[1]



def main():
    
    datasets = [#(eb.Datasets.Crowd1, 2000, 1.),
                #(eb.Datasets.Crowd2, 2000, .5),
                (eb.Datasets.EEG, 2000, 1.),
                (eb.Datasets.Face, 1965/2, 1.),
                #(eb.Datasets.Mario, 2000, 1.),
                #(eb.Datasets.Mario_window, 2000, 1.),
                (eb.Datasets.MEG, 375/2, 1.),
                (eb.Datasets.Mouth, 2000, 1.),
                #(eb.Datasets.RatLab, 2000, .5),
                #(eb.Datasets.Traffic, 2000, 1.),
                #(eb.Datasets.Tumor, 500/2, .25),
                (eb.Datasets.WAV, 2000, 1.),
                ]
    
    plt.figure(figsize=(22., 12.))
    
    for i, (dataset, N, scaling) in enumerate(datasets):
        plt.subplot(2, 3, i+1)
        ep.plot(n_dimensions, dataset=dataset, 
                              N=N,
                              noisy_dims=0,
                              scaling=scaling, 
                              n_chunks=1,
                              keep_variance=np.arange(.7, 1., .01),
                              show_plot=False,
                              seed=0)
    plt.show()
    plt.savefig('experiment_2016_02_pca.pdf')



if __name__ == '__main__':
    main()
        