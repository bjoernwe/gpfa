import numpy as np

from matplotlib import pyplot
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import mdp

import fpp

if __name__ == '__main__':

    # parameters
    k = 5
    noisy_dims = 200
    whitening = False
    iterations = 1
    seed = None

    # load data file
    faces_raw = np.load('faces.npy')
    faces = np.array(faces_raw, copy=True)
    N, D = faces.shape

    # PCA
    pca = mdp.nodes.PCANode(output_dim=0.99)
    pca.train(faces)
    faces = pca.execute(faces)
    print 'dim after pca:', faces.shape

    # add noisy dims
    R = np.random.RandomState(seed=seed)
    sigmas = np.std(faces_raw, axis=0)
    perc = np.percentile(sigmas, 100)
    print perc
    faces_noisy = np.hstack([faces, perc * R.randn(N, noisy_dims)])

    # prepare plot
    fig, ax = pyplot.subplots(1, 1)
    
    # run experiments with and without noise
    #for d, data in enumerate([faces, faces_noisy]):
    for d, data in enumerate([faces]):

        # algorithms
        models = []
        models.append(fpp.FPP(output_dim=2,
                              k=k,
                              iterations=iterations,
                              iteration_dim=10,
                              minimize_variance=False,
                              normalized_objective=True))
        #models.append(fpp.LPP(output_dim=2, k=k))

        # train & plot        
        for m, model in enumerate(models):
    
            # current algorithm
            print "%s" % (model.__class__.__name__)
        
            # train
            model.train(data)
            result = model.execute(data)
        
            # plot
            #ax[d][m].scatter(result[:,0], result[:,1])
            #ax[d][m].set_title(model.__class__.__name__)
            ax.scatter(result[:,0], result[:,1])
            ax.set_title(model.__class__.__name__)
            
            plotted_faces = np.empty((0, 2))
        
            for i in np.random.permutation(faces.shape[0]):
                
                if np.any((np.abs(plotted_faces[:,0] - result[i,0]) < 15) & \
                          (np.abs(plotted_faces[:,1] - result[i,1]) < 70)):
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
            
                ax.add_artist(ab)
            
            pyplot.draw()

    # show plot
    print 'finish'
    pyplot.show()
