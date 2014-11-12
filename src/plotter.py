import collections
import datetime
import functools
import inspect
import multiprocessing
import numpy as np
import os
import time

from matplotlib import pyplot as plt



def plot(f, **kwargs):
    """
    Plots the real-valued function f using its given arguments. One of the 
    arguments to be an iterable, which is used for the x-axis.
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
            default_args = dict(zip(fargspecs.args[-len(fargspecs.defaults):], fargspecs.defaults))
            fkwargs = default_args.copy()
        fkwargs.update(kwargs)

        # extract arguments for plotter itself
        iter_arg_name = iterable_arguments[0]
        iter_arg      = fkwargs.pop(iter_arg_name)
        show_plot     = fkwargs.pop('show_plot', True)
        repetitions   = fkwargs.pop('repetitions', 1)
        processes     = fkwargs.pop('processes', None)
        if processes is None:
            processes = multiprocessing.cpu_count()

        # make sure, all arguments are defined for function f
        undefined_args = set(fargspecs.args)
        undefined_args.discard(iter_arg_name)  # remove iterable argument
        undefined_args.difference_update(fkwargs.keys())  # remove other known arguments
        if len(undefined_args) > 0:
            print 'Error: Undefined arguments:', str.join(', ', undefined_args)
            return

        # wrap function f
        f_partial = functools.partial(_f_wrapper, iter_arg_name=iter_arg_name, f=f,
                                      **fkwargs)

        # prepare argument list for repetitions
        if repetitions > 1:
            old_arg = iter_arg
            iter_arg = np.array(iter_arg)
            iter_arg = np.repeat(iter_arg, repetitions)

        # start a pool of processes
        time_start = time.localtime()
        pool = multiprocessing.Pool(processes=processes)
        result = pool.map(f_partial, iter_arg, chunksize=1)
        pool.close()
        pool.join()
        time_stop = time.localtime()

        # calculate running time
        time_diff = time.mktime(time_stop) - time.mktime(time_start)
        time_delta = datetime.timedelta(seconds=time_diff)
        time_start_str = time.strftime('%Y-%m-%d %H:%M:%S', time_start)
        if time_start.tm_yday == time_stop.tm_yday:
            time_stop_str = time.strftime('%H:%M:%S', time_start)
        else:
            time_stop_str = time.strftime('%Y-%m-%d %H:%M:%S', time_start)

        # either errorbar plot or regular plot
        if repetitions > 1:
            iter_arg = old_arg
            result = np.reshape(result, (len(iter_arg), repetitions))
            plt.errorbar(iter_arg, np.mean(result, axis=1),
                         yerr=np.std(result, axis=1))
        else:
            plt.plot(iter_arg, result)

        # describe plot
        plt.xlabel(iter_arg_name)
        plt.suptitle(inspect.stack()[1][1])
        plt.title('Time: %s - %s (%s)\n' % (time_start_str, time_stop_str, time_delta) + 
                  'Parameters: %s' % str.join(', ', ['%s=%s' % (k,v) for k,v in fkwargs.items()]),
                  fontsize=12)
        plt.subplots_adjust(top=0.85)

        if show_plot:
            if not os.path.exists('plotter_results'):
                os.makedirs('plotter_results')
            timestamp = time.strftime('%Y%m%d%H%M%S', time_start)
            plt.savefig('plotter_results/%s%02d.png' % (timestamp, len([f for f in os.listdir('plotter_results/') if f.startswith(timestamp)])))
            plt.show()

        return

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
    plt.subplot(1, 2, 1)
    plot(_example_func, x=0, y=range(10), repetitions=10, show_plot=False)
    plt.subplot(1, 2, 2)
    plot(_example_func, x=range(10), y=0, repetitions=10)



if __name__ == '__main__':
    main()
