import collections
import datetime
import functools
import inspect
import multiprocessing
import numpy as np
import os
import pickle
import time



def evaluate(f, repetitions=1, processes=None, save_result=True, **kwargs):
    """
    Evaluates the real-valued function f using the given keyword arguments. One 
    of the arguments must be an iterable, which is used for parallel evaluation.
    """
    
    # look for iterable arguments
    iterable_arguments = [k for (k, v) in kwargs.items() 
                          if isinstance(v, collections.Iterable)]

    if len(iterable_arguments) == 0:

        print 'Warning: No iterable argument found for plotting.'
        return

    elif len(iterable_arguments) >= 2:

        print 'Warning: More than one iterable argument found for plotting.'
        return

    else:
        
        # get default arguments of function f and update them with given ones
        #
        # this is not strictly necessary but otherwise the argument lists lacks
        # the default ones which should be included in the plot
        fargspecs = inspect.getargspec(f)
        fkwargs = {}
        if fargspecs.defaults is not None:
            fkwargs = dict(zip(fargspecs.args[-len(fargspecs.defaults):], fargspecs.defaults))
        fkwargs.update(kwargs)

        # make sure, all arguments are defined for function f
        if len(fargspecs.args) > len(fkwargs):
            undefined_args = set(fargspecs.args) - set(fkwargs.keys())
            print 'Error: Undefined arguments:', str.join(', ', undefined_args)
            return

        # the iterable argument and its name
        iter_arg_name = iterable_arguments[0]
        iter_arg      = fkwargs.pop(iter_arg_name)

        # wrap function f
        f_partial = functools.partial(_f_wrapper, iter_arg_name=iter_arg_name, f=f,
                                      **fkwargs)

        # prepare argument list for repetitions
        f_iter_arg = iter_arg
        if repetitions > 1:
            f_iter_arg = np.repeat(iter_arg, repetitions)

        # start a pool of processes
        if processes is None:
            processes = multiprocessing.cpu_count()
        time_start = time.localtime()
        pool = multiprocessing.Pool(processes=processes)
        values = pool.map(f_partial, f_iter_arg, chunksize=1)
        pool.close()
        pool.join()
        time_stop = time.localtime()

        # re-arrange repetitions in 2d array
        if repetitions > 1:
            values = np.reshape(values, (len(iter_arg), repetitions))
            
        # calculate a prefix for result files
        timestamp = time.strftime('%Y%m%d%H%M%S', time_start)
        number_of_results = len([f for f in os.listdir('plotter_results/') if f.startswith(timestamp)])
        result_prefix = '%s%02d' % (timestamp, number_of_results)
        
        # prepare result
        result = {}    
        result['values'] = values
        result['time_start'] = time_start
        result['time_stop'] = time_stop
        result['iter_arg_name'] = iter_arg_name
        result['iter_arg'] = iter_arg
        result['kwargs'] = fkwargs
        result['script'] = ([s for s in inspect.stack() if os.path.basename(s[1]) != 'plotter.py'] + [None])[0]
        result['repetitions'] = repetitions
        result['result_prefix'] = result_prefix
        
        if save_result:
            with open('plotter_results/%s.pkl' % result_prefix, 'wb') as f:
                pickle.dump(result, f)
        
        return result
    
    return



def plot(f, repetitions=1, processes=None, show_plot=True, save_plot=True, **kwargs):
    """
    Plots the real-valued function f using the given keyword arguments. One of 
    the arguments must be an iterable, which is used for the x-axis.
    """

    # makes evaluate() independent from matplotlib
    from matplotlib import pyplot as plt

    # run the experiment
    result = evaluate(f, repetitions=repetitions, **kwargs)
    if result is None:
        return
    
    # calculate running time
    time_start = result['time_start']
    time_stop = result['time_stop']
    time_diff = time.mktime(time_stop) - time.mktime(time_start)
    time_delta = datetime.timedelta(seconds=time_diff)
    time_start_str = time.strftime('%Y-%m-%d %H:%M:%S', time_start)
    if time_start.tm_yday == time_stop.tm_yday:
        time_stop_str = time.strftime('%H:%M:%S', time_start)
    else:
        time_stop_str = time.strftime('%Y-%m-%d %H:%M:%S', time_start)

    # either errorbar plot or regular plot
    x_values = result['iter_arg']
    y_values = result['values']
    if repetitions > 1:
        plt.errorbar(x_values, np.mean(y_values, axis=1),
                     yerr=np.std(y_values, axis=1))
    else:
        plt.plot(x_values, y_values)

    # describe plot
    plt.xlabel(result['iter_arg_name'])
    plt.suptitle(result['script'])
    plotted_args = result['kwargs'].copy()
    if repetitions > 1:
        plotted_args['repetitions'] = repetitions
    plt.title('Time: %s - %s (%s)\n' % (time_start_str, time_stop_str, time_delta) + 
              'Parameters: %s' % str.join(', ', ['%s=%s' % (k,v) for k,v in plotted_args.items()]),
              fontsize=12)
    plt.subplots_adjust(top=0.85)

    # save plot in file
    if save_plot:
        if not os.path.exists('plotter_results'):
            os.makedirs('plotter_results')
        plt.savefig('plotter_results/%s.png' % result['result_prefix'])

    # show plot
    if show_plot:
        plt.show()

    return



def _f_wrapper(arg, iter_arg_name, f, **kwargs):
    """
    A simple wrapper for function f that allows having a specific argument
    ('arg_name') as the first argument. This is the method that is actually
    managed and called by the multiprocessing pool. Therefore the argument 
    'niceness' is removed from **kwargs and used to increment the niceness of 
    the current process (default: 10). Also the NumPy's random number generator 
    is initialized with a new seed.
    """
    os.nice(kwargs.pop('niceness', 20))
    np.random.seed()
    kwargs[iter_arg_name] = arg
    return f(**kwargs)



def _example_func(x, y='ignore me!', z=False):
    """
    A simple example function with three arguments x, y and z.
    """
    fx = x**2 / 10
    fy = np.sin(y)
    return fx + fy + .5 * np.random.randn()



def main():
    from matplotlib import pyplot as plt
    plt.subplot(1, 2, 1)
    plot(_example_func, x=0, y=range(10), repetitions=10, show_plot=False)
    plt.subplot(1, 2, 2)
    plot(_example_func, x=range(10), y=0, repetitions=10)



if __name__ == '__main__':
    main()
