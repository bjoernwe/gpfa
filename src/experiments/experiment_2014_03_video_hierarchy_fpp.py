#! /usr/bin/python

import glob
import numpy as np

from scipy import ndimage
from pylab import *

import mdp
import mdp.nodes
import mdp.parallel

import fpp


"""
This example uses video images as input signal. The images are expected to be
in the video subdirectory and are named like 00000001.jpg, 00000002.jpg, ...
This is the format produced by mplayer:

  mplayer -vo jpeg video.mp4

"""

if __name__ == "__main__":

    #------------------------------------------------------[ Parameters ]
    
    zoom_factor = 0.15
    filename = 'result_fpp__FoT_18fJ2k.npy'

    #------------------------------------------------------[ Load Data ]

    # list of input files
    input_files = glob.glob("video/*.jpg")
    input_files.sort()
    N = len(input_files)

    # load first image
    for infile in input_files:
        image = ndimage.imread(infile, flatten=True)
        image = ndimage.zoom(image, zoom_factor)
        break

    # data array
    image_height, image_width = image.shape
    image_dim = image.size
    data = np.zeros((N, image_dim))

    # load all images
    print 'loading images ...'
    for i in range(0, N):
        image = ndimage.imread(input_files[i], flatten=True)
        image = ndimage.zoom(image, zoom_factor)
        data[i,:] = image.ravel()
        print input_files[i]
        #break

    #data = np.load('data_ape.npy').T
    #image_width, image_height = 120, 120
    N, D = data.shape
    print 'data dimensions:', data.shape

    #------------------------------------------------------[ Add noise ]

    noise_node  = mdp.nodes.NoiseNode( input_dim=D,
                                       noise_args = (0, 0.01) )
    data = noise_node.execute(data)
    #np.save('data.npy', data)

    #------------------------------------------------------[ SFA Hierarchy ]

    # raw data switchboard
    switchboard_1 = mdp.hinet.Rectangular2dSwitchboard( in_channels_xy    = (image_width, image_height),
                                                        field_channels_xy = (10, 10),
                                                        field_spacing_xy  = (5, 5),
                                                        in_channel_dim    = 1,
                                                        ignore_cover      = True)

    # sfa in layer 1
    whitening_1a = mdp.nodes.WhiteningNode(input_dim=switchboard_1.out_channel_dim)
    fpp_node_1a = fpp.FPP(input_dim = switchboard_1.out_channel_dim,
                          output_dim = 32,
                          k = 5,
                          iterations = 5,
                          iteration_dim = 32,
                          preserve_past = False,
                          neighbor_graph = False)
    expansion_1 = mdp.nodes.QuadraticExpansionNode( input_dim = fpp_node_1a.output_dim )
    whitening_1b = mdp.nodes.WhiteningNode(input_dim=expansion_1.output_dim)
    fpp_node_1b = fpp.FPP(input_dim = expansion_1.output_dim,
                          output_dim = 32,
                          k = 5,
                          iterations = 5,
                          iteration_dim = 32,
                          preserve_past = False,
                          neighbor_graph = False)
    sfa_node_1 = mdp.hinet.FlowNode( mdp.Flow([ whitening_1a, fpp_node_1a, expansion_1, whitening_1b, fpp_node_1b ]) )

    # repetition of sfa node in layer 1
    sfa_layer_1 = mdp.hinet.CloneLayer( sfa_node_1, n_nodes = switchboard_1.output_channels )


    # switchboard for layer 2
#     switchboard_2 = mdp.hinet.Rectangular2dSwitchboard( in_channels_xy    = switchboard_1.out_channels_xy,
#                                                         field_channels_xy = (4, 4),
#                                                         field_spacing_xy  = (2, 2),
#                                                         in_channel_dim    = sfa_node_1.output_dim,
#                                                         ignore_cover      = True)
# 
#     # sfa in layer 2
#     whitening_2a = mdp.nodes.WhiteningNode(input_dim=switchboard_1.out_channel_dim)
#     fpp_node_2a = fpp.FPP(input_dim = switchboard_2.out_channel_dim,
#                           output_dim = 32,
#                           k = 5,
#                           iterations = 5,
#                           iteration_dim = 32,
#                           preserve_past = False,
#                           neighbor_graph = False)
#     expansion_2 = mdp.nodes.QuadraticExpansionNode( input_dim = fpp_node_2a.output_dim )
#     whitening_2b = mdp.nodes.WhiteningNode(input_dim=switchboard_1.out_channel_dim)
#     fpp_node_2b = fpp.FPP(input_dim = switchboard_2.out_channel_dim,
#                           output_dim = 32,
#                           k = 5,
#                           iterations = 5,
#                           iteration_dim = 32,
#                           preserve_past = False,
#                           neighbor_graph = False)
#     sfa_node_2 = mdp.hinet.FlowNode( mdp.Flow([ whitening_2a, fpp_node_2a, expansion_2, whitening_2b, fpp_node_2b ]) )
# 
# 
#     # repetition of sfa node in layer 2
#     sfa_layer_2 = mdp.hinet.CloneLayer( sfa_node_2, n_nodes = switchboard_2.output_channels )
# 
# 
#     # switchboard for layer 3
#     switchboard_3 = mdp.hinet.Rectangular2dSwitchboard( in_channels_xy    = switchboard_2.out_channels_xy,
#                                                         field_channels_xy = (4, 4),
#                                                         field_spacing_xy  = (2, 2),
#                                                         in_channel_dim    = sfa_node_2.output_dim,
#                                                         ignore_cover      = True)
# 
#     # sfa in layer 3
#     sfa_node_3a = mdp.nodes.SFANode( input_dim = switchboard_3.out_channel_dim,
#                                      output_dim = 32 )
#     expansion_3 = mdp.nodes.QuadraticExpansionNode( input_dim = sfa_node_3a.output_dim )
#     sfa_node_3b = mdp.nodes.SFANode( input_dim = expansion_3.output_dim,
#                                      output_dim = 32 )
#     sfa_node_3 = mdp.hinet.FlowNode( mdp.Flow([ sfa_node_3a, expansion_3, sfa_node_3b ]) )
# 
# 
#     # repetition of sfa node in layer 3
#     sfa_layer_3 = mdp.hinet.CloneLayer( sfa_node_3, n_nodes = switchboard_3.output_channels )


    # final sfa step
    sfa_node_Xa = mdp.nodes.SFANode( input_dim = sfa_layer_1.output_dim,
                                     output_dim = 32 )
    expansion_X = mdp.nodes.QuadraticExpansionNode( input_dim = sfa_node_Xa.output_dim )
    sfa_node_Xb = mdp.nodes.SFANode( input_dim = expansion_X.output_dim,
                                     output_dim = 32 )
    sfa_node_X = mdp.hinet.FlowNode( mdp.Flow([ sfa_node_Xa, expansion_X, sfa_node_Xb ]) )

    # final flow
    #flow = mdp.Flow([switchboard_1, sfa_layer_1, switchboard_2, sfa_layer_2, switchboard_3, sfa_layer_3, sfa_node_X])
    flow = mdp.Flow([switchboard_1, sfa_layer_1, sfa_node_X])

    # show the flow
    #mdp.hinet.show_flow(flow, filename="test.html", show_size=True)

    #------------------------------------------------------[ Training ]

    flow.train(data)
    flow.save('flow.dat')
    print ':)'

    #------------------------------------------------------[ Evaluation ]

    result = flow.execute(data)
    np.save(filename, result)
    print 'result saved to', filename
    plot(result[:,0:5])
    show()
    