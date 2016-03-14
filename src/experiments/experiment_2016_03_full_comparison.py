import collections
import matplotlib.pyplot as plt
import mdp
import numpy as np
import sys

sys.path.append('/home/weghebvc/workspace/git/explot/src/')
import explot as ep

import experiment_base as eb



def main():

    output_dim = 10
    expansion = True
    
                # dataset                   N       scale   hi-net
    datasets = [#(eb.Datasets.Crowd1,        [2000,1067],   .25, {'expansion': expansion, 'channels_xy_1': (5,3), 'spacing_xy_1': (5,3), 'channels_xy_n': (2,1), 'spacing_xy_n': (2,1), 'node_output_dim': 10}),
                #(eb.Datasets.Crowd3,        [2000,547],   .25,  {'expansion': expansion, 'channels_xy_1': (4,3), 'spacing_xy_1': (4,3), 'channels_xy_n': (2,1), 'spacing_xy_n': (2,1), 'node_output_dim': 10}),
                
                #(eb.Datasets.Crowd2,        [2000,350],   .25,  {'expansion': expansion, 'channels_xy_1': (5,3), 'spacing_xy_1': (5,3), 'channels_xy_n': (2,1), 'spacing_xy_n': (2,1), 'node_output_dim': 10}),
                #(eb.Datasets.Dancing,       [2000,1736],   .25, {'expansion': expansion, 'channels_xy_1': (5,3), 'spacing_xy_1': (5,3), 'channels_xy_n': (2,1), 'spacing_xy_n': (2,1), 'node_output_dim': 10}),
                #(eb.Datasets.Face,          [1500,465],   1.,   {'expansion': expansion, 'channels_xy_1': (3,5), 'spacing_xy_1': (3,5), 'channels_xy_n': (2,1), 'spacing_xy_n': (2,1), 'node_output_dim': 10}),
                #(eb.Datasets.Mario,         2000,   .5,         {'expansion': expansion, 'channels_xy_1': (5,4), 'spacing_xy_1': (5,4), 'channels_xy_n': (2,1), 'spacing_xy_n': (2,1), 'node_output_dim': 10}),
                #(eb.Datasets.Mouth,         2000,   1.,         {'expansion': expansion, 'channels_xy_1': (5,3), 'spacing_xy_1': (5,3), 'channels_xy_n': (2,1), 'spacing_xy_n': (2,1), 'node_output_dim': 10}),
                #(eb.Datasets.RatLab,        2000,   .25,        {'expansion': expansion, 'channels_xy_1': (5,3), 'spacing_xy_1': (5,1), 'channels_xy_n': (2,1), 'spacing_xy_n': (2,1), 'node_output_dim': 10}),
                (eb.Datasets.Traffic,       2000,   .25,        {'expansion': expansion, 'channels_xy_1': (5,3), 'spacing_xy_1': (5,3), 'channels_xy_n': (2,1), 'spacing_xy_n': (2,1), 'node_output_dim': 10}),
                ]
    
    k = 5
    K = 1
    p = 1
    iterations = 50

    # TODO: remove measure
    algorithms = [(eb.Algorithms.HiSFA, eb.Measures.delta_ndim, {}), 
                  (eb.Algorithms.HiPFA, None, {'p': p, 'K': K}),
                  (eb.Algorithms.HiGPFA2, eb.Measures.gpfa_ndim, {'p': p, 'k': k, 'iterations': iterations}),
                  #(eb.Algorithms.HiForeCA, None, {}),
                 ]
    
    NA = len(algorithms)
    
    #measures = [(eb.Measures.delta_ndim, {}), 
    #            (eb.Measures.gpfa_ndim, {'k': k, 'p': p})]
    
    for _, (dataset, N, scaling, kwargs_dat) in enumerate(datasets):
            
        #
        # plot extracted signals
        #
        
        plt.figure(figsize=(30., 16.))
        suptitle = '%s (x%0.2f): %s' % (dataset, scaling, kwargs_dat)
        plt.suptitle(suptitle)
        print suptitle, '\n'
        
        chunks = {}
        models = {}
        for a, (algorithm, _, kwargs_alg) in enumerate(algorithms):
            
            #plt.subplot(NA+3,1,a+1)
            plt.subplot2grid((NA+2,2*NA+1), (a,0), colspan=2*NA)
            alg_str = '%s : %s' % (algorithm, kwargs_alg)
            plt.title(alg_str)
            print alg_str, '\n'
            
            kwargs = dict(kwargs_dat)
            kwargs.update(kwargs_alg)
            
            chunks[algorithm] = []
            for use_test_set in [False, True]:
                chunk, model, original_data, image_shape = eb.calc_projected_data(algorithm=algorithm, 
                                                                                  # data 
                                                                                  dataset=dataset, 
                                                                                  output_dim=output_dim,
                                                                                  N=N,
                                                                                  scaling=scaling,
                                                                                  keep_variance=1., 
                                                                                  use_test_set=use_test_set,
                                                                                  # misc 
                                                                                  seed=0,
                                                                                  **kwargs) 
                chunks[algorithm].append(chunk)
                models[algorithm] = model
            
            colors = ['b', 'r']
            offset = 0#output_dim-2
            for d in range(output_dim-1, -1+offset, -1):
                if isinstance(N, collections.Iterable):
                    plt.plot(range(N[0]), chunks[algorithm][0][:,d], color=_color(c=colors[0], d=d-offset))
                    plt.plot(range(N[0], N[0]+N[1]), chunks[algorithm][1][:,d], color=_color(c=colors[1], d=d-offset))
                else:
                    for i, color in enumerate(colors):
                        plt.plot(range(i*N, (i+1)*N), chunks[algorithm][i][:,d], color=_color(c=color, d=d-offset))
                        
            # print hierarchy
            hierarchy_strs = []
            for node in model:#s[eb.Algorithms.HiSFA]:
                if isinstance(node, mdp.hinet.Rectangular2dSwitchboard):
                    hierarchy_strs.append("%s: %d -> %d\n%dx%d = %d:" % (node, node.input_dim, node.output_dim, node.out_channels_xy[0], node.out_channels_xy[1], node.output_channels))
                elif isinstance(node, mdp.hinet.Layer):
                    for subnode in node[0]:
                        hierarchy_strs.append('  %s: %d -> %d' % (subnode, subnode.input_dim, subnode.output_dim))
                else:
                    hierarchy_strs.append('%s: %d -> %d' % (node, node.input_dim, node.output_dim))
            hierarchy_str = '\n'.join(hierarchy_strs)
            print hierarchy_str, '\n'
    
        #
        # compare signals with different measures
        #    
        #for m, (measure, kwargs_mea) in enumerate(measures):
        for m, (_, measure, kwargs_mea) in enumerate(algorithms):
            
            if measure is None:
                continue
            
            #plt.subplot(NA+3,2,len(algorithms)*2+1+m)
            plt.subplot2grid((NA+2,2*NA+1), (NA,2*m), colspan=2)
            plt.ylabel(measure)
                
            for a, (algorithm, _, kwargs_alg) in enumerate(algorithms):
                
                kwargs = dict(kwargs_dat)
                kwargs.update(kwargs_mea)
                kwargs.update(kwargs_alg)
                errors = eb.prediction_error(measure=measure, 
                                             dataset=dataset, 
                                             algorithm=algorithm, 
                                             output_dim=output_dim, 
                                             N=N, 
                                             scaling=scaling,
                                             keep_variance=1., 
                                             use_test_set=False,
                                             **kwargs)
                colors = ['b', 'r', 'g', 'y']
                w = .8 / len(algorithms)
                plt.bar(np.arange(output_dim)+.6+a*w, height=errors, width=w, color=colors[a])
    
        #
        # principle angles between signals
        #
        colors = ['r', 'g', 'y']
        for a, (algorithm, _, kwargs_alg) in enumerate(algorithms[1:]):
            #plt.subplot(NA+3, NA-1, NA*(NA-1)+NA+a)
            plt.subplot2grid((NA+2,2*NA+1), (NA+1,2*a+2), colspan=2)
            angles = [eb._principal_angle(chunks[eb.Algorithms.HiSFA][0], chunks[algorithm][0][:,:i+1]) for i in range(output_dim)]
            plt.plot(range(1, len(angles)+1), angles, c=colors[a], linewidth=1.3)
            plt.gca().set_ylim([0, np.pi/2])
            
        #
        # example image and hierarchy configuration
        #
        #plt.subplot(NA+3, 2, 2*(NA+2)+1)
        plt.subplot2grid((NA+2,2*NA+1), (0,2*NA))
        plt.imshow(original_data[0][0].reshape(image_shape), cmap='gist_gray', interpolation='none')
        #plt.subplot(NA+3, 2, 2*(NA+2)+2)
        plt.subplot2grid((NA+2,2*NA+1), (1,2*NA), rowspan=NA+1)
        plt.text(.02, .5, hierarchy_str, fontsize=8, verticalalignment='center')

    plt.show()



def _color(c, d):
    tb = np.array([0,0,1], dtype=np.float)
    tr = np.array([1,0,0], dtype=np.float)
    tw = np.array([1,1,1], dtype=np.float)
    f = 1. / (d+1)**1.5
    if c == 'r':
        c = tr
    elif c == 'b':
        c = tb
    else:
        assert False
    return f*c + (1-f)*tw



if __name__ == '__main__':
    main()
        