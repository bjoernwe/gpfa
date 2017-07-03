import numpy as np
import scipy.linalg
import scipy.spatial.distance

import mdp


class RandomProjection(mdp.Node):

    def __init__(self, output_dim, input_dim=None, dtype=None, seed=None):
        super(RandomProjection, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.rnd = np.random.RandomState(seed=seed)
        self.whitening = None
        return

    def _train(self, x, is_whitened=False):
        # whiten data
        if not is_whitened:
            self.whitening = mdp.nodes.WhiteningNode(reduce=True)
            self.whitening.train(x)
            self.whitening.stop_training()
        return

    def _stop_training(self):
        if self.whitening:
            D = self.whitening.output_dim
        else:
            D = self.input_dim
        A = self.rnd.rand(D, D)
        A = A + A.T
        _, self.U = scipy.linalg.eigh(A, eigvals=(0, self.output_dim-1))
        assert np.allclose(self.U.T.dot(self.U), np.eye(self.output_dim, self.output_dim))
        return

    def _execute(self, x):
        if self.whitening:
            x = self.whitening.execute(x)
        return x.dot(self.U)
