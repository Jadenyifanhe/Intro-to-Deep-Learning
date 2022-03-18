import numpy as np
from resampling import *

class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        batch_size, in_channels, input_width, input_height = A.shape
        output_width = input_width - self.kernel + 1
        output_height = input_height - self.kernel + 1

        Z = np.zeros((batch_size, in_channels, output_width, output_height))
        self.pidx = np.zeros((batch_size, in_channels, output_width, output_height, 2))

        for i in range(batch_size):
            for j in range(in_channels):
                for w in range(output_width):
                    for h in range(output_height):
                        window = A[i, j, w:w + self.kernel, h:h + self.kernel]
                        x, y = np.unravel_index(np.argmax(window, axis=None), window.shape)
                        self.pidx[i, j, w, h, 0] = x + w
                        self.pidx[i, j, w, h, 1] = y + h
                        Z[i, j, w, h] = window[x, y]

        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        batch_size, out_channels, output_width, output_height = dLdZ.shape
        input_width = output_width + self.kernel - 1
        input_height = output_height + self.kernel - 1

        dLdA = np.zeros((batch_size, out_channels, input_width, input_height))

        for i in range(batch_size):
            for j in range(out_channels):
                for w in range(output_width):
                    for h in range(output_height):
                        x = int(self.pidx[i, j, w, h, 0])
                        y = int(self.pidx[i, j, w, h, 1])
                        dLdA[i][j][x][y] += dLdZ[i][j][w][h]

        return dLdA

class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        batch_size, in_channels, input_width, input_height = A.shape
        output_width = input_width - self.kernel + 1
        output_height = input_height - self.kernel + 1

        Z = np.zeros((batch_size, in_channels, output_width, output_height))

        for i in range(batch_size):
            for j in range(in_channels):
                for w in range(output_width):
                    for h in range(output_height):
                        Z[i, j, w, h] = np.mean(A[i, j, w:w + self.kernel, h:h + self.kernel])

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        batch_size, out_channels, output_width, output_height = dLdZ.shape
        input_width = output_width + self.kernel - 1
        input_height = output_height + self.kernel - 1

        dLdA = np.zeros((batch_size, out_channels, input_width, input_height))

        for i in range(batch_size):
            for j in range(out_channels):
                for w in range(output_width):
                    for h in range(output_height):
                        for x in range(self.kernel):
                            for y in range(self.kernel):
                                dLdA[i, j, w + x, h + y] += 1 / (self.kernel ** 2) * dLdZ[i, j, w, h]

        return dLdA

class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride
        
        #Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(self.kernel)
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        
        Z_pool = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z_pool)

        return Z
    
    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA_ds = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdA_ds)

        return dLdA

class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        #Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(self.kernel)
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        Z_pool = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z_pool)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        dLdA_ds = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdA_ds)

        return dLdA
