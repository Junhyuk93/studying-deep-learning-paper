import numpy as np


def conv_2d(x, kernel, bias):
    kernel_shape = kernel.shape[0]

    # Assuming Padding = 0, stride = 1
    output_shape = x.shape[0] - kernel_shape + 1
    result = np.zeros((output_shape, output_shape))

    for row in range(x.shape[0] - 1):
        for col in range(x.shape[1] -1):
            window = x[row: row + kernel_shape, col: col + kernel_shape]
            result[row, col] = np.sum(np.multiply(kernel,window))

    return result + bias