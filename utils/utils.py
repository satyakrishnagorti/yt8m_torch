import numpy as np


def dequantize(feat_matrix, max_quantized_value=2, min_quantized_value=-2):
    """
        De-quantize the feature from the byte format to the float format.

        Args:
        feat_matrix: the input ndarray.
        max_quantized_value: the maximum of the quantized value.
        min_quantized_value: the minimum of the quantized value.

        Returns:
        A floating tensor which has the same shape as feat_matrix.
    """
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_matrix * scalar + bias


def pad_if_necessary(tensor, new_size):
    """
    Pads 0 arrays along 0th dimension if tensor.shape[0] < new_size
    :param tensor: ndarray of shape (x, feature_size)
    :param new_size: new size along 0th dim
    :return: ndarray of shape (new_size, feature_size)
    """

    if tensor.shape[0] > new_size:
        return tensor[:new_size, :]

    remaining = new_size - tensor.shape[0]
    zero_arr = np.zeros(tensor.shape[1]).reshape(-1, tensor.shape[1])
    return np.concatenate((tensor, np.repeat(zero_arr, remaining, axis=0)), axis=0)
