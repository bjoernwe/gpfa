import numpy as np
import sys



def principal_angles(A, B):
    """A and B must be column-orthogonal.
    Returns min and max principle angle.
    Golub: Matrix Computations, 1996
    [http://www.disi.unige.it/person/BassoC/teaching/python_class02.pdf]
    """
    if A.ndim == 1:
        A = np.array(A, ndmin=2).T
    if B.ndim == 1:
        B = np.array(B, ndmin=2).T
    assert A.ndim == B.ndim == 2
    A = np.linalg.qr(A)[0]
    B = np.linalg.qr(B)[0]
    _, S, _ = np.linalg.svd(np.dot(A.T, B))
    angles = np.arccos(np.clip(S, -1, 1))
    return np.min(angles), np.max(angles)



def format_arg_value(arg_val):
    """ Return a string representing a (name, value) pair.
    
    >>> format_arg_value(('x', (1, 2, 3)))
    'x=(1, 2, 3)'
    """
    arg, val = arg_val
    if isinstance(val, np.ndarray):
        return '%s=numpy.ndarray<%s>' % (arg, val.shape)
    return "%s=%r" % (arg, val)
    
    
    
def echo(fn, write=sys.stdout.write):
    """ Echo calls to a function.
    
    Returns a decorated version of the input function which "echoes" calls
    made to it by writing out the function's name and the arguments it was
    called with.
    """
    import functools
    # Unpack function's arg count, arg names, arg defaults
    code = fn.func_code
    argcount = code.co_argcount
    argnames = code.co_varnames[:argcount]
    fn_defaults = fn.func_defaults or list()
    argdefs = dict(zip(argnames[-len(fn_defaults):], fn_defaults))
    
    @functools.wraps(fn)
    def wrapped(*v, **k):
        # Collect function arguments by chaining together positional,
        # defaulted, extra positional and keyword arguments.
        positional = map(format_arg_value, zip(argnames, v))
        defaulted = [format_arg_value((a, argdefs[a]))
                     for a in argnames[len(v):] if a not in k]
        nameless = map(repr, v[argcount:])
        keyword = map(format_arg_value, k.items())
        args = positional + defaulted + nameless + keyword
        write("%s(%s)\n\n" % (fn.__name__, ", ".join(args)))
        return fn(*v, **k)
    return wrapped



def echo_on_exception(fn, write=sys.stdout.write):
    """ Echo calls to a function.
    
    Returns a decorated version of the input function which "echoes" calls
    made to it by writing out the function's name and the arguments it was
    called with.
    """
    import functools
    # Unpack function's arg count, arg names, arg defaults
    code = fn.func_code
    argcount = code.co_argcount
    argnames = code.co_varnames[:argcount]
    fn_defaults = fn.func_defaults or list()
    argdefs = dict(zip(argnames[-len(fn_defaults):], fn_defaults))
    
    @functools.wraps(fn)
    def wrapped(*v, **k):
        try:
            return fn(*v, **k)
        except Exception:
            # Collect function arguments by chaining together positional,
            # defaulted, extra positional and keyword arguments.
            positional = map(format_arg_value, zip(argnames, v))
            defaulted = [format_arg_value((a, argdefs[a]))
                         for a in argnames[len(v):] if a not in k]
            nameless = map(repr, v[argcount:])
            keyword = map(format_arg_value, k.items())
            args = positional + defaulted + nameless + keyword
            write("Threw exception: %s(%s)\n\n" % (fn.__name__, ", ".join(args)))
            raise
    return wrapped



@echo_on_exception
def test(a, b = 4, c = 'blah-blah', *args, **kwargs):
    assert c



def f_identity(x):
    return x



def f_exp08(x):
    return np.abs(x)**.8



def test_principle_angles():

    A = np.array([[1,0], 
                  [0,1], 
                  [0,0], 
                  [0,0]]) 
    B = np.array([[0,0], 
                  [0,1], 
                  [1,0], 
                  [0,0]])
    C = np.array([[0,0], 
                  [0,0], 
                  [1,0], 
                  [0,1]])
    
    assert principal_angles(A, A) == (0, 0)
    assert principal_angles(A, B) == (0, np.pi/2) 
    assert principal_angles(A, C) == (np.pi/2, np.pi/2)
    
    print 'okay' 



def main():
    test(1, c=False)
    #test_principle_angles()



if __name__ == '__main__':
    main()
    