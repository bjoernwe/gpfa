"""
This example uses video images as input signal. The images are expected to be
in the video subdirectory and are named like 00000001.jpg, 00000002.jpg, ...
This is the format produced by mplayer:

  mplayer -vo jpeg video.mp4

"""

import numpy as np

from matplotlib import pyplot
from scipy import ndimage

import mdp

import fpp


if __name__ == "__main__":

    first_image = 1#22
    last_image = 3022
    N = last_image - first_image + 1
    zoom = 0.1

    # load first image
    image = ndimage.imread('video/{:08}.jpg'.format(first_image), flatten=True)
    small_image = ndimage.zoom(image, zoom)
    H, W = small_image.shape
    vector = small_image.ravel('F')

    # signal dim
    D = small_image.size
    print 'image size: (%d, %d)' % (W, H)
    print 'signal dim.:', D
    print 'number of points:', N

    # store data
    #data = np.zeros((N, D))
    data = 0.01 * np.random.randn(N, D)
    data[0] += vector

    # load all images
    for i in range(1, N):
        t = first_image + i
        image = ndimage.imread('video/{:08}.jpg'.format(t), flatten=True)
        small_image = ndimage.zoom(image, zoom)
        vector = small_image.ravel('F')
        data[i] += vector
        if i % 1000 == 0:
            print i
            
    # whitening
    print data.shape
    whitening_node = mdp.nodes.WhiteningNode(svd=True)
    whitening_node.train(data)
    data = whitening_node.execute(data)
    assert data.shape[1] == D
        
    k = 3
    iterations = 3
    fpp_node = fpp.FPP(output_dim=10,
                       k=k,
                       iterations=iterations,
                       iteration_dim=10,
                       preserve_past=False,
                       neighbor_graph=False)
    #fpp_node = mdp.nodes.SFANode(output_dim=10)
    
    fpp_node.train(data)
    y = fpp_node.execute(data)
    #fpp_node.stop_training()
    
    for i in range(fpp_node.output_dim):
        feature = np.reshape(fpp_node.U[:,i], (H, W), 'F')
        #feature = np.reshape(fpp_node.sf[:,i], (H, W), 'F')
        pyplot.subplot(2, fpp_node.output_dim, i+1)
        pyplot.imshow(feature)
        pyplot.subplot(2, fpp_node.output_dim, i+fpp_node.output_dim+1)
        pyplot.plot(y[:,i])
        
    pyplot.show()
    