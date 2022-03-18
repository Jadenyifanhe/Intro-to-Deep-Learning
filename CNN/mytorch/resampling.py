import numpy as np

class Upsample1d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """

        batch_size, in_channels, input_width = A.shape
        output_width = input_width * self.upsampling_factor - (self.upsampling_factor - 1)

        Z = np.zeros((batch_size, in_channels, output_width))

        for i in range(batch_size):
            for j in range(in_channels):
                for k in range(input_width):
                    Z[i][j][k * self.upsampling_factor] = A[i][j][k]

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        batch_size, in_channels, output_width = dLdZ.shape
        input_width = (output_width + (self.upsampling_factor - 1)) // self.upsampling_factor

        dLdA = np.zeros((batch_size, in_channels, input_width))


        for i in range(batch_size):
            for j in range(in_channels):
                for k in range(input_width):
                    dLdA[i][j][k] = dLdZ[i][j][k * self.upsampling_factor]

        return dLdA

class Downsample1d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        self.width_A = A.shape[2]

        batch_size, in_channels, input_width = A.shape
        output_width = (input_width + (self.downsampling_factor - 1)) // self.downsampling_factor

        Z = np.zeros((batch_size, in_channels, output_width))

        for i in range(batch_size):
            for j in range(in_channels):
                for k in range(output_width):
                    Z[i][j][k] = A[i][j][k * self.downsampling_factor]

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """

        batch_size, in_channels, output_width = dLdZ.shape
        # Cannot convert back
        # input_width = output_width * self.downsampling_factor - (self.downsampling_factor - 1)
        dLdA = np.zeros((batch_size, in_channels, self.width_A))

        for i in range(batch_size):
            for j in range(in_channels):
                for k in range(output_width):
                    dLdA[i][j][k * self.downsampling_factor] = dLdZ[i][j][k]

        return dLdA

class Upsample2d():

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """

        batch_size, in_channels, input_width, input_height = A.shape
        output_width = input_width * self.upsampling_factor - (self.upsampling_factor - 1)
        output_height = input_height * self.upsampling_factor - (self.upsampling_factor - 1)

        Z = np.zeros((batch_size, in_channels, output_width, output_height))

        for i in range(batch_size):
            for j in range(in_channels):
                for w in range(input_width):
                    for h in range(input_height):
                        Z[i][j][w * self.upsampling_factor][h * self.upsampling_factor] = A[i][j][w][h]

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        batch_size, in_channels, output_width, output_height = dLdZ.shape
        input_width = (output_width + (self.upsampling_factor - 1)) // self.upsampling_factor
        input_height = (output_height + (self.upsampling_factor - 1)) // self.upsampling_factor

        dLdA = np.zeros((batch_size, in_channels, input_width, input_height))

        for i in range(batch_size):
            for j in range(in_channels):
                for w in range(input_width):
                    for h in range(input_height):
                        dLdA[i][j][w][h] = dLdZ[i][j][w * self.upsampling_factor][h * self.upsampling_factor]

        return dLdA

class Downsample2d():

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):

        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, in_channels, output_width, output_height)
        """

        self.width_A, self.height_A = A.shape[2], A.shape[3]

        batch_size, in_channels, input_width, input_height = A.shape
        output_width = (input_width + (self.downsampling_factor - 1)) // self.downsampling_factor
        output_height = (input_height + (self.downsampling_factor - 1)) // self.downsampling_factor

        Z = np.zeros((batch_size, in_channels, output_width, output_height))

        for i in range(batch_size):
            for j in range(in_channels):
                for w in range(output_width):
                    for h in range(output_height):
                        Z[i][j][w][h] = A[i][j][w * self.downsampling_factor][h * self.downsampling_factor]

        return Z

    def backward(self, dLdZ):

        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        batch_size, in_channels, output_width, output_height = dLdZ.shape
        # Cannot convert back
        # input_width = output_width * self.downsampling_factor - (self.downsampling_factor - 1)
        # input_height = output_height * self.downsampling_factor - (self.downsampling_factor - 1)

        dLdA = np.zeros((dLdZ.shape[0], dLdZ.shape[1], self.width_A, self.height_A))

        for i in range(batch_size):
            for j in range(in_channels):
                for w in range(output_width):
                    for h in range(output_height):
                        dLdA[i][j][w * self.downsampling_factor][h * self.downsampling_factor] = dLdZ[i][j][w][h]

        return dLdA