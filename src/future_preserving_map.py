import mdp


class FuturePreservingMap(mdp.Node):

    def __init__(self, input_dim=None, dtype=None):
        super(FuturePreservingMap, self).__init__(input_dim=input_dim, dtype=dtype)
        
    
    def _train(self, x):
        pass
    
    
    def _stop_training(self):
        pass