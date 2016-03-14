import joblib
import numpy as np
import os

from subprocess import call

import mdp


mem = joblib.Memory(cachedir='/scratch/weghebvc', verbose=1)


@mem.cache
def train(x, output_dim, seed):
    
        # save training data in CSV
        cwd = os.getcwd()
        fdir = os.path.dirname(os.path.abspath(__file__))
        rnd = np.random.RandomState(seed)
        run_id = str(rnd.randint(100000, 1000000))
        np.savetxt("%s/foreca_node_train_%s.csv" % (cwd, run_id), x, delimiter=",")

        # run R script and load result (extraction matrix)
        output_dim = max(2, output_dim)
        call(['Rscript', '%s/foreca_node.r' % fdir, run_id, str(output_dim), cwd])
        W = np.loadtxt('%s/foreca_node_result_%s.csv' % (cwd, run_id))
        if output_dim < 2:
            W = W[:,0:1]

        # clean files
        call(['rm', '%s/foreca_node_train_%s.csv' % (cwd, run_id)])
        call(['rm', '%s/foreca_node_result_%s.csv' % (cwd, run_id)])
        return W



class ForeCA(mdp.Node):
    '''
    A wrapper node for the ForeCA implementation in R.
    '''

    def __init__(self, output_dim, input_dim=None, dtype=None, seed=None):
        super(ForeCA, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.W = None
        self.seed = seed
        #self.rnd = np.random.RandomState(seed)
        return
    
        
        
    def _train(self, x):
        self.W = train(x=x, output_dim=self.output_dim, seed=self.seed)
        return
    
    
    
    def _stop_training(self):
        pass
    
    
    
    def _execute(self, x):
        return x.dot(self.W)

