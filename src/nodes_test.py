import matplotlib.pyplot as plt
import numpy as np
import unittest

import sfa_node


class NodesTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Generate some test signals
        :return:
        """

        # arguments
        n_samples = 500
        n_noisy_dims = 5

        # generate sine wave and alternating signal
        cls.signal1 = 10 * np.sin(np.linspace(0, 4*np.pi, num=n_samples))
        cls.signal2 = np.ones(n_samples)
        cls.signal2[::2] = -1
        signals = np.stack([cls.signal1, cls.signal2], axis=1)

        # generate noise
        noise = np.random.randn(n_samples, n_noisy_dims)

        # concatenate
        data = np.concatenate([signals[:,[0]], noise, signals[:,[1]]], axis=1)
        # mix signals via rotation matrix (45 degrees)
        R = np.array([[.7071, -.7071],[.7071, .7071]])
        M = np.eye(n_noisy_dims + 2)
        M[:2,:2] = R
        M[-2:,-2:] = R

        cls.data = np.dot(data, M)
        return

    def test_sfa(self):
        # sine wave & noise
        sfa = sfa_node.SFA(output_dim=2, include_fast_signals=False)
        sfa.train(self.data)
        signals = sfa.execute(self.data)
        assert self._is_similar(signals[:,0], self.signal1)
        assert not self._is_similar(signals[:,1], self.signal2)

        # fast & sine wave
        sfa = sfa_node.SFA(output_dim=2, include_fast_signals=True)
        sfa.train(self.data)
        signals = sfa.execute(self.data)
        assert self._is_similar(signals[:,0], self.signal2)
        assert self._is_similar(signals[:,1], self.signal1)

    def _is_similar(self, signal1, signal2):
        """
        Compares signals after normalization and under both signs
        :param signal1:
        :param signal2:
        :return:
        """
        return np.allclose(signal1 / np.std(signal1), signal2 / np.std(signal2), rtol=0, atol=1e-1) or \
               np.allclose(signal1 / np.std(signal1), -signal2 / np.std(signal2), rtol=0, atol=1e-1)


if __name__ == '__main__':
    unittest.main()
