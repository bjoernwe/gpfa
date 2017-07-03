__docformat__ = "restructuredtext en"

import numpy as np
import scipy

import mdp
from mdp import numx, Node, NodeException, TrainingException
from mdp.utils import (mult, pinv, CovarianceMatrix, QuadraticForm,
                       symeig, SymeigException)

class SFA(Node):

    def __init__(self, include_fast_signals, input_dim=None, output_dim=None, dtype=None,
                 include_last_sample=True):
        """
        This is a copy of the standard SFA node with the additional boolean parameter ``include_fast_signals''. When set
        to True, delta values are not sorted by increasing size but by decreasing difference from 2, i.e., signals with
        the most extreme delta values are extracted fist, no matter if slow or fast.

        For the ``include_last_sample`` switch have a look at the
        SFANode class docstring.
         """
        super(SFA, self).__init__(input_dim, output_dim, dtype)
        self._include_last_sample = include_last_sample
        self._include_fast_signals = include_fast_signals

        # init two covariance matrices
        # one for the input data
        self._cov_mtx = CovarianceMatrix(dtype)
        # one for the derivatives
        self._dcov_mtx = CovarianceMatrix(dtype)

        # set routine for eigenproblem
        self._symeig = symeig

        # SFA eigenvalues and eigenvectors, will be set after training
        self.d = None
        self.sf = None  # second index for outputs
        self.avg = None
        self._bias = None  # avg multiplied with sf
        self.tlen = None

    def time_derivative(self, x):
        """Compute the linear approximation of the time derivative."""
        # this is faster than a linear_filter or a weave-inline solution
        return x[1:, :]-x[:-1, :]

#     def _set_range(self):
#         if self.output_dim is not None and self.output_dim <= self.input_dim:
#             # (eigenvalues sorted in ascending order)
#             rng = (1, self.output_dim)
#         else:
#             # otherwise, keep all output components
#             rng = None
#             self.output_dim = self.input_dim
#         return rng

    def _check_train_args(self, x, *args, **kwargs):
        # check that we have at least 2 time samples to
        # compute the update for the derivative covariance matrix
        s = x.shape[0]
        if  s < 2:
            raise TrainingException('Need at least 2 time samples to '
                                    'compute time derivative (%d given)'%s)
        
    def _train(self, x, include_last_sample=None):
        """
        For the ``include_last_sample`` switch have a look at the
        SFANode class docstring.
        """
        if include_last_sample is None:
            include_last_sample = self._include_last_sample
        # works because x[:None] == x[:]
        last_sample_index = None if include_last_sample else -1

        # update the covariance matrices
        self._cov_mtx.update(x[:last_sample_index, :])
        self._dcov_mtx.update(self.time_derivative(x))

    def _stop_training(self, debug=False):
        ##### request the covariance matrices and clean up
        self.cov_mtx, self.avg, self.tlen = self._cov_mtx.fix()
        del self._cov_mtx
        # do not center around the mean:
        # we want the second moment matrix (centered about 0) and
        # not the second central moment matrix (centered about the mean), i.e.
        # the covariance matrix
        self.dcov_mtx, self.davg, self.dtlen = self._dcov_mtx.fix(center=False)
        del self._dcov_mtx

        #rng = self._set_range()

        #### solve the generalized eigenvalue problem
        # the eigenvalues are already ordered in ascending order
        try:
            self.d, self.sf = scipy.linalg.eigh(self.dcov_mtx, self.cov_mtx)
            if self._include_fast_signals:
                idc = np.argsort(np.abs(2-self.d))[::-1][:min(self.output_dim, self.input_dim)]
            else:
                idc = np.argsort(self.d)[::-1][:min(self.output_dim, self.input_dim)]
            self.d = self.d[idc]
            self.sf = self.sf[:,idc]
            
            d = self.d
            # check that we get only *positive* eigenvalues
            if d.min() < 0:
                err_msg = ("Got negative eigenvalues: %s."
                           " You may either set output_dim to be smaller,"
                           " or prepend the SFANode with a PCANode(reduce=True)"
                           " or PCANode(svd=True)"% str(d))
                raise NodeException(err_msg)
        except SymeigException, exception:
            errstr = str(exception)+"\n Covariance matrices may be singular."
            raise NodeException(errstr)

        if not debug:
            # delete covariance matrix if no exception occurred
            del self.cov_mtx
            del self.dcov_mtx

        # store bias
        self._bias = mult(self.avg, self.sf)

    def _execute(self, x, n=None):
        """Compute the output of the slowest functions.
        If 'n' is an integer, then use the first 'n' slowest components."""
        if n:
            sf = self.sf[:, :n]
            bias = self._bias[:n]
        else:
            sf = self.sf
            bias = self._bias
        return mult(x, sf) - bias

    def _inverse(self, y):
        return mult(y, pinv(self.sf)) + self.avg

    def get_eta_values(self, t=1):
        """Return the eta values of the slow components learned during
        the training phase. If the training phase has not been completed
        yet, call `stop_training`.

        The delta value of a signal is a measure of its temporal
        variation, and is defined as the mean of the derivative squared,
        i.e. delta(x) = mean(dx/dt(t)^2).  delta(x) is zero if
        x is a constant signal, and increases if the temporal variation
        of the signal is bigger.

        The eta value is a more intuitive measure of temporal variation,
        defined as
        eta(x) = t/(2*pi) * sqrt(delta(x))
        If x is a signal of length 't' which consists of a sine function
        that accomplishes exactly N oscillations, then eta(x)=N.

        :Parameters:
           t
             Sampling frequency in Hz.

             The original definition in (Wiskott and Sejnowski, 2002)
             is obtained for t = number of training data points, while
             for t=1 (default), this corresponds to the beta-value defined in
             (Berkes and Wiskott, 2005).
        """
        if self.is_training():
            self.stop_training()
        return self._refcast(t / (2 * numx.pi) * numx.sqrt(self.d))


