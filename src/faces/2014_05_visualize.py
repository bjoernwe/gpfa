#import numpy as np

#from matplotlib import pyplot
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

#import mdp

from fpp import *


if __name__ == '__main__':
    
    # parameters
    algorithm = 'lpp'
    k = 5
    iterations = 5
    reduce_variance = True
    whitening = False
    normalized_objective = True
    additive_noise = 0
    additional_noise_dim = 0
    additional_noise_std = 200
    
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
        
    #std = np.std(faces, axis=0)
    #pyplot.hist(std)
        
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

    # model
    if algorithm == 'sfa':
        node = mdp.nodes.SFANode(output_dim=2)
    elif algorithm == 'lpp':
        node = LPP(output_dim=2,
                   k=k,
                   normalized_objective=normalized_objective)
    elif algorithm == 'fpp':
        node = FPP(output_dim=2,
                   k=k,
                   iterations=iterations,
                   iteration_dim=4,
                   minimize_variance=False,
                   normalized_objective=normalized_objective)
    else:
        print 'unexpected algorithm', algorithm
        assert False
        

    # training
    node.train(faces)
    node.stop_training()
    result = node.execute(faces)
    #print node.U.shape
    #result = node.U
    #result /= np.max(result, axis=0) - np.min(result, axis=0)

    # normalize to variance one in sum
    cov = np.cov(result.T)
    E, U = np.linalg.eigh(cov)
    W = U.dot(np.diag(1./np.sqrt((np.sum(E)*np.ones(2)))).dot(U.T))
    result = result.dot(W)
    
    # plot
    fig, ax = pyplot.subplots()
    ax.scatter(result[:,0], result[:,1])
    
    #for j in range(faces.shape[0]-1):
    #    ax.plot(result[j:j+2,0], result[j:j+2,1], '-')
    
    plotted_faces = np.empty((0, 2))

    for i in np.random.permutation(faces.shape[0]):
        
        if np.any((np.abs(plotted_faces[:,0] - result[i,0]) < 0.008) & \
                  (np.abs(plotted_faces[:,1] - result[i,1]) < 0.014)):
            continue
        else:
            plotted_faces = np.insert(plotted_faces, plotted_faces.shape[0], values=result[i,:], axis=0)

        xy = result[i]
    
        arr = faces_raw[i].reshape((28, 20))
        #arr = faces[i].reshape((2*28, 20))
        im = OffsetImage(arr, zoom=1, cmap=pyplot.get_cmap('gray'))
    
        ab = AnnotationBbox(im, xy,
                            xybox=(0., 0.),
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0,
                            arrowprops=None)
    
        ax.add_artist(ab)
    
    if reduce_variance:
        ax.set_title('PCA + %s' % algorithm.upper())
    else:
        ax.set_title(algorithm.upper())
    
    pyplot.draw()
    pyplot.show()
    