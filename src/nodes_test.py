import matplotlib.pyplot as plt
import mdp
import numpy as np
import unittest

import PFANodeMDP

import foreca.foreca_node
import gpfa_node
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
        cls.signal1 = np.sqrt(2) * np.sin(np.linspace(0, 4*np.pi, num=n_samples))
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
        sfa = sfa_node.SFA(output_dim=3, include_fast_signals=False)
        sfa.train(self.data)
        signals = sfa.execute(self.data)
        assert self._is_similar(signals[:,0], self.signal1)
        assert not self._is_similar(signals[:,1], self.signal2)
        assert not self._is_similar(signals[:,2], self.signal1)
        assert not self._is_similar(signals[:,2], self.signal2)

    def test_sfa_fast(self):
        # fast & sine wave
        sfa = sfa_node.SFA(output_dim=3, include_fast_signals=True)
        sfa.train(self.data)
        signals = sfa.execute(self.data)
        assert self._is_similar(signals[:,0], self.signal2)
        assert self._is_similar(signals[:,1], self.signal1)
        assert not self._is_similar(signals[:,2], self.signal1)
        assert not self._is_similar(signals[:,2], self.signal2)

    def test_foreca(self):
        # without whitening
        foreca_node = foreca.foreca_node.ForeCA(output_dim=3, whitening=False)
        foreca_node.train(self.data)
        signals = foreca_node.execute(self.data)
        assert self._is_similar(signals[:,0], self.signal2)
        assert self._is_similar(signals[:,1], self.signal1)
        assert not self._is_similar(signals[:,2], self.signal1)
        assert not self._is_similar(signals[:,2], self.signal2)

    def test_foreca_whitening(self):
        # with whitening
        foreca_node = foreca.foreca_node.ForeCA(output_dim=3, whitening=True)
        foreca_node.train(self.data)
        signals = foreca_node.execute(self.data)
        assert self._is_similar(signals[:,0], self.signal2)
        assert self._is_similar(signals[:,1], self.signal1)
        assert not self._is_similar(signals[:,2], self.signal1)
        assert not self._is_similar(signals[:,2], self.signal2)

    def test_pfa(self):
        pfa = PFANodeMDP.PFANode(output_dim=3, p=1, k=1)
        pfa.train(self.data)
        signals = pfa.execute(self.data)
        assert self._is_similar(signals[:,0], self.signal2)
        assert self._is_similar(signals[:,1], self.signal1)
        assert not self._is_similar(signals[:,2], self.signal1)
        assert not self._is_similar(signals[:,2], self.signal2)

    def test_gpfa(self):
        # non-whitened data
        gpfa = gpfa_node.GPFA(output_dim=3, p=2, k=5, iterations=10)
        gpfa.train(self.data)
        signals = gpfa.execute(self.data)
        assert self._is_similar(signals[:,0], self.signal2)
        assert self._is_similar(signals[:,1], self.signal1)
        assert not self._is_similar(signals[:,2], self.signal1)
        assert not self._is_similar(signals[:,2], self.signal2)

    def test_gpfa_white(self):
        # whitened data
        whitening = mdp.nodes.WhiteningNode()
        whitening.train(self.data)
        data_white = whitening.execute(self.data)
        gpfa = gpfa_node.GPFA(output_dim=3, p=2, k=5, iterations=10)
        gpfa.train(data_white, is_whitened=True)
        signals = gpfa.execute(data_white)
        assert self._is_similar(signals[:,0], self.signal2)
        assert self._is_similar(signals[:,1], self.signal1)
        assert not self._is_similar(signals[:,2], self.signal1)
        assert not self._is_similar(signals[:,2], self.signal2)

    def test_gpfa_white_exception(self):
        # detection of non-white data
        gpfa = gpfa_node.GPFA(output_dim=3, p=2, k=5, iterations=10)
        self.assertRaises(ValueError, gpfa.train, x=self.data, is_whitened=True)

    def _is_similar(self, signal1, signal2, rtol=0, atol=1e-1):
        """
        Compares signals after normalization and under both signs

        :param signal1:
        :param signal2:
        :param rtol:
        :param atol:
        :return:
        """
        return np.allclose(signal1 / np.std(signal1), signal2 / np.std(signal2), rtol=rtol, atol=atol) or \
               np.allclose(signal1 / np.std(signal1), -signal2 / np.std(signal2), rtol=rtol, atol=atol)


if __name__ == '__main__':
    unittest.main()
