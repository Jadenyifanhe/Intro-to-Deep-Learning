import numpy as np


class Identity:
    
    def forward(self, Z):
    
        self.A = Z
        
        return self.A
    
    def backward(self):
    
        dAdZ = np.ones(self.A.shape, dtype="f")
        
        return dAdZ


class Sigmoid:

    def forward(self, Z):
        self.A = Z
        self.A = 1 / (1 + np.exp(-self.A))

        return self.A

    def backward(self):
        dAdZ = self.A - self.A ** 2

        return dAdZ


class Tanh:

    def forward(self, Z):
        self.A = Z
        self.A = (1 / 2 * (np.exp(self.A) - np.exp(-self.A))) / (1 / 2 * (np.exp(self.A) + np.exp(- self.A)))

        return self.A

    def backward(self):
        dAdZ = 1 - self.A ** 2

        return dAdZ


class ReLU:

    def forward(self, Z):
        self.A = Z
        self.A = np.maximum(self.A, 0)

        return self.A

    def backward(self):
        dAdZ = np.where(self.A > 0, 1.0, 0.0)

        return dAdZ
        
        
