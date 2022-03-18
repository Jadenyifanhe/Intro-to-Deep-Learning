import numpy as np

class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8

        self.N = None # the batch_size

        self.Z = None
        self.NZ = None
        self.BZ = None

        # The following attributes will be tested
        self.V = np.ones((1, num_features))
        self.M = np.zeros((1, num_features))

        self.BW = np.ones((1, num_features))
        self.dLdBW = np.zeros((1, num_features))

        self.Bb = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # inference parameters
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def __call__(self, Z, eval=False):
        return self.forward(Z, eval)

    def forward(self, Z, eval=False):
        """
        Argument:
            x (np.array): (batch_size, num_features)
            eval (bool): inference status

        Return:
            out (np.array): (batch_size, num_features)

        NOTE: The eval parameter is to indicate whether we are in the 
        training phase of the problem or are we in the inference phase.
        So see what values you need to recompute when eval is True.
        """
        self.Z = Z
        self.N = batch_size = Z.shape[0]
        
        raise NotImplementedError("Batchnorm Forward Not Implemented")

    def backward(self, dLdBZ):
        """
        Argument:
            dLdBZ (np.array): (batch size, num_features)
        Return:
            out (np.array): (batch size, num_features)
        """

        raise NotImplementedError("Batchnorm Backward Not Implemented")
