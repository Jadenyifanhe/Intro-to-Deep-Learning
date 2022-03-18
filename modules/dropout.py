# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np

class Dropout(object):
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x, train=True):

        if train:
            self.mask = np.random.binomial(1, self.p, x.shape)
            out = x * self.mask / self.p

            return out

        else:

            return x

    def backward(self, delta):
        # TODO: Multiply mask with delta and return

        out = delta * self.mask
        return out