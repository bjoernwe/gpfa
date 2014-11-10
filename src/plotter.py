import collections
import datetime
import functools
import inspect
import multiprocessing
import numpy as np
import os
import time

from matplotlib import pyplot as plt


"""
A simple wrapper for function f that allows having a specific argument
('arg_name') as the first argument. The argument 'niceness' is removed from
kwargs and used to increment the niceness of the current process (default: 10).
Also the NumPy's random number generator is initialized with a new seed.
"""
def _f_wrapper(arg, arg_name, f, **kwargs):
    os.nice(kwargs.pop('niceness', 10))
    np.random.seed()
    kwargs[arg_name] = arg
    return f(**kwargs)



"""
Plots the real-valued function f using its given arguments. One of the argument
is expected to be an iterable, which is used for the x-axis.
"""
def plot(f, **kwargs):
    
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

        # extract arguments for plotter
        arg_name    = iterable_arguments[0]
        arg         = kwargs.pop(arg_name)
        processes   = kwargs.pop('processes', multiprocessing.cpu_count())
        show_plot   = kwargs.pop('show_plot', True)
        repetitions = kwargs.pop('repetitions', 1)

        # wrap function f
        f_partial = functools.partial(_f_wrapper, arg_name=arg_name, f=f,
                                      **kwargs)

        # prepare argument list for repetitions
        if repetitions > 1:
            old_arg = arg
            arg = np.array(arg)
            arg = np.repeat(arg, repetitions)

        # start a pool of processes
        time_start = time.localtime()
        pool = multiprocessing.Pool(processes=processes)
        result = pool.map(f_partial, arg, chunksize=1)
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
            arg = old_arg
            result = np.reshape(result, (len(arg), repetitions))
            plt.errorbar(arg, np.mean(result, axis=1),
                         yerr=np.std(result, axis=1))
        else:
            plt.plot(arg, result)

        # describe plot
        plt.xlabel(arg_name)
        plt.title(inspect.stack()[1][1], y=1)
        plt.suptitle('Time: %s - %s (%s)\n' % (time_start_str, time_stop_str, time_delta) + 
                     'Parameters: %s' % str.join(', ', ['%s=%s' % (k,v) for k,v in kwargs.items()]))
        plt.subplots_adjust(top=0.85)

        # show plot
        if show_plot:
            plt.show()

        return result

    return


"""
A simple example function with two arguments x and y.
"""
def _example_func(x, y):
    fx = x**2 / 10
    fy = np.sin(y)
    return fx + fy + .5 * np.random.randn()


def main():
    plot(_example_func, x=0, y=range(10), repetitions=10)
    #plot(_example_func, x=range(10), y=0, repetitions=100, show_plot=False)
    #plot(_example_func, x=0, y=range(10), repetitions=100, show_plot=False)
    #plt.show()


if __name__ == '__main__':
    main()
