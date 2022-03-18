# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *

class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        self.A = A
        batch_size, self.in_channels, input_size = self.A.shape
        self.output_size = input_size - self.kernel_size + 1

        Z = np.zeros((batch_size, self.out_channels, self.output_size))

        for i in range(batch_size):
            for j in range(self.out_channels):
                for k in range(self.output_size):
                    window = self.A[i, :, k:k + self.kernel_size]
                    Z[i, j, k] = np.sum(window * self.W[j])
                Z[i, j] = Z[i, j] + self.b[j]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """

        batch_size, out_channels, output_size = dLdZ.shape
        input_size = output_size + self.kernel_size - 1

        dLdA = np.zeros((batch_size, self.in_channels, input_size))

        for i in range(batch_size):
            for j in range(self.in_channels):
                for n in range(output_size):
                    for k in range(n, n + self.kernel_size):
                        dLdA[i, j, k] += sum(self.W[l, j, k-n] * dLdZ[i, l, n] for l in range(self.out_channels))

        for i in range(batch_size):
            for j in range(self.in_channels):
                for k in range(self.kernel_size):
                    for l in range(self.out_channels):
                        self.dLdW[l, j, k] += sum([self.A[i, j, n + k] * dLdZ[i, l, n] for n in range(output_size)])

        self.dLdb = np.sum(dLdZ, axis=(0, 2))

        return dLdA

class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
    
        self.stride = stride

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size)
        self.downsample1d = Downsample1d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Call Conv1d_stride1
        Z = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdZ = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ)

        return dLdA


class Conv2d_stride1():
    def __init__(self, in_channels, out_channels,
                 kernel_size, weight_init_fn=None, bias_init_fn=None):

        # Do not modify this method

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, self.in_channels, input_width, input_height = self.A.shape
        output_width = input_width - self.kernel_size + 1
        output_height = input_height - self.kernel_size + 1

        Z = np.zeros((batch_size, self.out_channels, output_width, output_height))

        for i in range(batch_size):
            for j in range(self.out_channels):
                for w in range(output_width):
                    for h in range(output_height):
                        window = self.A[i, :, w:w + self.kernel_size, h:h + self.kernel_size]
                        Z[i, j, w, h] = np.sum(window * self.W[j])
                Z[i, j] = Z[i, j] + self.b[j]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        batch_size, self.out_channels, output_width, output_height = dLdZ.shape
        input_width = output_width + self.kernel_size - 1
        input_height = output_height + self.kernel_size - 1

        dLdA = np.zeros((batch_size, self.in_channels, input_width, input_height))

        for i in range(batch_size):
            for j in range(self.in_channels):
                for w in range(output_width):
                    for h in range(output_height):
                        for x in range(w, w + self.kernel_size):
                            for y in range(h, h + self.kernel_size):
                                dLdA[i, j, x, y] += sum([self.W[l, j, x - w, y - h] * dLdZ[i, l, w, h] for l in range(self.out_channels)])

        for i in range(batch_size):
            for j in range(self.in_channels):
                for w in range(output_width):
                    for h in range(output_height):
                        for x in range(self.kernel_size):
                            for y in range(self.kernel_size):
                                for l in range(self.out_channels):
                                    self.dLdW[l, j, x, y] += self.A[i, j, w + x, h + y] * dLdZ[i, l, w, h]

        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))

        return dLdA

class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names
        self.stride = stride

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size)
        self.downsample2d = Downsample2d(self.stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        # Call Conv2d_stride1
        Z = self.conv2d_stride1.forward(A)

        # downsample
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        # Call downsample1d backward
        dLdA = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdA)

        return dLdA

class ConvTranspose1d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv1d stride 1 and upsample1d isntance
        #TODO
        self.upsample1d = Upsample1d(self.upsampling_factor)
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        #TODO
        # upsample
        A_upsampled = self.upsample1d.forward(A)

        # Call Conv1d_stride1()
        Z = self.conv1d_stride1.forward(A_upsampled)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #TODO
        #Call backward in the correct order
        delta_out = self.conv1d_stride1.backward(dLdZ)

        dLdA =  self.upsample1d.backward(delta_out)

        return dLdA

class ConvTranspose2d():
    def __init__(self, in_channels, out_channels, kernel_size, upsampling_factor,
                weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.upsampling_factor = upsampling_factor

        # Initialize Conv2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn)
        self.upsample2d = Upsample2d(self.upsampling_factor)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        # upsample
        A_upsampled = self.upsample2d.forward(A)

        # Call Conv2d_stride1()
        Z = self.conv2d_stride1.forward(A_upsampled)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        #Call backward in correct order
        delta_out = self.conv2d_stride1.backward(dLdZ)

        dLdA = self.upsample2d.backward(delta_out)

        return dLdA

class Flatten():

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """

        self.batch_size, self.in_channels, self.in_width = A.shape

        Z = np.reshape(A, (self.batch_size, self.in_channels * self.in_width))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        # Cannot convert back
        # batch_size = dLdZ.shape[0]
        # in_channels * in_width = dLdZ.shape[1] * dLdZ.shape[2]

        dLdA = np.reshape(dLdZ, (self.batch_size, self.in_channels, self.in_width))

        return dLdA

