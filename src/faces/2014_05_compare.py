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
    velocity = False
    
    # load data file
    faces_raw = np.load('faces.npy')
    
    # concatenate time steps
    if velocity:
        faces = np.hstack([faces_raw[:-1], faces_raw[1:]])
    else:
        faces = np.array(faces_raw, copy=True)
    print faces.shape
    
    # PCA
    if reduce_variance:
        pca = mdp.nodes.PCANode(output_dim=0.99)
        pca.train(faces)
        faces = pca.execute(faces)
        print 'dim after pca:', faces.shape
        
    # whiten data
    if whitening:
        whitening = mdp.nodes.WhiteningNode(reduce=True)
        whitening.train(faces)
        faces = whitening.execute(faces)
        print 'dim after whitening:', faces.shape


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
                       iteration_dim=4,
                       minimize_variance=False,
                       normalized_objective=True)
        else:
            print 'unexpected algorithm', algorithm
            assert False
            
    
        # training
        node.train(faces)
        results = node.execute(faces)
        
        # normalize results
        #results /= np.std(results, axis=0)
        results /= np.max(results, axis=0) - np.min(results, axis=0)
        
        # plot
        ax[a].scatter(results[:,0], results[:,1])
        
        plotted_faces = np.empty((0, 2))
    
        for i in np.random.permutation(faces.shape[0]):
            
            if np.any((np.abs(plotted_faces[:,0] - results[i,0]) < 0.05) & \
                      (np.abs(plotted_faces[:,1] - results[i,1]) < 0.07)):
                continue
            else:
                plotted_faces = np.insert(plotted_faces, plotted_faces.shape[0], values=results[i,:], axis=0)
    
            xy = results[i]
        
            arr = faces_raw[i].reshape((28, 20))
            im = OffsetImage(arr, zoom=1, cmap=pyplot.get_cmap('gray'))
        
            ab = AnnotationBbox(im, xy,
                                xybox=(0., 0.),
                                xycoords='data',
                                boxcoords="offset points",
                                pad=0,
                                arrowprops=None)
        
            ax[a].add_artist(ab)
            if reduce_variance:
                ax[a].set_title('PCA + %s' % algorithm)
            else:
                ax[a].set_title(algorithm)
        
        pyplot.draw()
        
    pyplot.show()
    