import collections
import numpy as np
import multiprocessing


def my_func(a, b, c=False):
    print 'a:', a
    print 'b:', b
    print 'c:', c
    print '\n'


"""
A simple wrapper for function f that allows having a specific argument
('arg_name') as the first argument.
"""
def f_wrapper(arg, arg_name, f, **kwargs):
    kwargs[arg_name] = arg
    return f(**kwargs)


"""
Plots the real-valued function f using its given arguments. One of the argument
is expected to be an iterable, which is used for the x-axis.
"""
def plot(f, **kwargs):
    iterable_arguments = [k for (k, v) in kwargs.items()
                          if isinstance(v, collections.Iterable)]
    if len(iterable_arguments) == 0:
        print 'Warning: No iterable argument found for plotting.'
        return
    elif len(iterable_arguments) >= 2:
        print 'Warning: More than one iterable argument found for plotting.'
        return
    else:
        arg_name = iterable_arguments[0]
        arg = kwargs.pop(arg_name)
        for x in arg:
            f_wrapper(x, arg_name, f, **kwargs)
    return


def main():
    plot(my_func, a=-1, b=[1,2,3])


if __name__ == '__main__':
    main()
