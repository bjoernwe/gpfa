#import joblib
import numpy as np
import os

from subprocess import call

import mdp


#mem = joblib.Memory(cachedir='/scratch/weghebvc', verbose=1)


#@mem.cache
def train(x, output_dim, whitening):
    
        # whitening
        m = None
        W = None
        if whitening:
            whitening_node = mdp.nodes.WhiteningNode(reduce=True)
            whitening_node.train(x)
            x = whitening_node.execute(x)
            m = whitening_node.avg
            W = whitening_node.v
    
        # save training data in CSV
        cwd = os.getcwd()
        fdir = os.path.dirname(os.path.abspath(__file__))
        rnd = np.random.RandomState()
        run_id = str(rnd.randint(100000, 1000000))
        np.savetxt("%s/foreca_node_train_%s.csv" % (cwd, run_id), x, delimiter=",")

        # run R script and load result (extraction matrix)
        call(['Rscript', '%s/foreca_node.r' % fdir, run_id, str(output_dim), cwd])
        U = np.loadtxt('%s/foreca_node_result_%s.csv' % (cwd, run_id))
        if output_dim < 2:
            U = np.array(U, ndmin=2).T

        # clean files
        call(['rm', '%s/foreca_node_train_%s.csv' % (cwd, run_id)])
        call(['rm', '%s/foreca_node_result_%s.csv' % (cwd, run_id)])
        assert U.shape == (x.shape[1], output_dim)
        #assert W.shape == (x.shape[1], x.shape[1])
        return m, W, U



class ForeCA(mdp.Node):
    '''
    A wrapper node for the ForeCA implementation in R.
    '''

    def __init__(self, output_dim, input_dim=None, whitening=True, dtype=None):
        super(ForeCA, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.whitening = whitening
        self.m = None
        self.W = None
        self.U = None
        #self.seed = seed
        #self.rnd = np.random.RandomState(seed)
        return
    
        
        
    def _train(self, x):
        self.m, self.W, self.U = train(x=x, output_dim=self.output_dim, whitening=self.whitening)
        return
    
    
    
    def _stop_training(self):
        pass
    
    
    
    def _execute(self, x):
        if self.m is not None:
            x = x - self.m
            x = x.dot(self.W)
        result = x.dot(self.U)
        assert result.shape == (x.shape[0], self.output_dim)
        return result



if __name__ == '__main__':
    x = np.random.randn(100,10)
    x = np.hstack([x,x])
    foreca = ForeCA(output_dim=1)
    foreca.train(x)
    y = foreca.execute(x)
    print np.cov(y.T)
    