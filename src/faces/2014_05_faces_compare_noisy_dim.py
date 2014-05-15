#import numpy as np

#from matplotlib import pyplot
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

#import mdp

from fpp import *


if __name__ == '__main__':
    
    # parameters
    k = 5
    iterations = 15
    reduce_variance = False
    whitening = False
    additive_noise = 0
    additional_noise_dim = 560
    additional_noise_std = 50
    spacing = [[0.015, 0.02], [0.015, 0.02]]
    
    # load data file
    faces_raw = np.load('faces.npy')
    faces = np.array(faces_raw, copy=True)
    print faces_raw.shape

    # PCA
    if reduce_variance:
        pca = mdp.nodes.PCANode(output_dim=0.99)
        pca.train(faces)
        faces = pca.execute(faces)
        print 'dim after pca:', faces.shape

    # plot variance
    #pyplot.hist(np.std(faces_raw, axis=0))
    #pyplot.show()

    # additive noise
    if additive_noise > 0:
        faces += additive_noise * np.random.randn(faces.shape[0], faces.shape[1])
        
    # additional dimensions
    if additional_noise_dim > 0:
        noise = additional_noise_std * np.random.randn(faces.shape[0], additional_noise_dim)
        faces = np.hstack([faces, noise])
        
    # whiten data
    if whitening:
        whitening = mdp.nodes.WhiteningNode(reduce=True)
        whitening.train(faces)
        faces = whitening.execute(faces)
        print 'dim after whitening:', faces.shape


    #for algorithm in ['SFA', 'LPP', 'gPFA']:
    fig, ax = pyplot.subplots(1, 2)
    for a, algorithm in enumerate(['LPP', 'gPFA']):

        # model
        if algorithm == 'SFA':
            node = mdp.nodes.SFANode(output_dim=2)
        elif algorithm == 'LPP':
            node = LPP(output_dim=2,
                       k=k,
                       normalized_objective=True)
        elif algorithm == 'gPFA':
            node = FPP(output_dim=2,
                       k=k,
                       iterations=iterations,
                       iteration_dim=2,
                       minimize_variance=False,
                       normalized_objective=True)
        else:
            print 'unexpected algorithm', algorithm
            assert False
            
    
        # training
        node.train(faces)
        result = node.execute(faces)
        
        # normalize result
        #result /= np.std(result, axis=0)
        result /= np.max(result, axis=0) - np.min(result, axis=0)
        
        # plot
        ax[a].scatter(result[:,0], result[:,1])
        
        plotted_faces = np.empty((0, 2))
    
        for i in np.random.permutation(faces.shape[0]):
            
            # already another picture close by?
            if np.any((np.abs(plotted_faces[:,0] - result[i,0]) < 0.05) & \
                      (np.abs(plotted_faces[:,1] - result[i,1]) < 0.07)):
                continue
            else:
                plotted_faces = np.insert(plotted_faces, plotted_faces.shape[0], values=result[i,:], axis=0)
    
            xy = result[i]
        
            arr = faces_raw[i].reshape((28, 20))
            im = OffsetImage(arr, zoom=1, cmap=pyplot.get_cmap('gray'))
        
            ab = AnnotationBbox(im, xy,
                                xybox=(0., 0.),
                                xycoords='data',
                                boxcoords="offset points",
                                pad=0,
                                arrowprops=None)
        
            ax[a].add_artist(ab)
            ax[a].set_title(algorithm)
        
        pyplot.draw()
        
    pyplot.show()
    