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


def resize_axis(tensor, axis, new_size, fill_value=0):
    """
        Truncates or pads a tensor to new_size on on a given axis.

        Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
        size increases, the padding will be performed at the end, using fill_value.

        Args:
            tensor: The tensor to be resized.
            axis: An integer representing the dimension to be sliced.
            new_size: An integer or 0d tensor representing the new value for
              tensor.shape[axis].
            fill_value: Value to use to fill any new entries in the tensor. Will be cast
              to the type of tensor.

        Returns:
            The resized tensor.
      """

    raise NotImplementedError