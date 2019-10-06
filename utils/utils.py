import os
import torch
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


def find_class_by_name(name, modules):
    """
        Searches the provided modules for the named class and returns it.
    """
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)


def get_feature_similarity_matrix(batch_video_matrix, batch_num_frames, device, normalize=True):

    batch_size = batch_video_matrix.shape[0]

    batch_video_norm = torch.norm(batch_video_matrix, dim=2, keepdim=True)
    video_normed_matrix = batch_video_matrix / batch_video_norm
    video_normed_matrix[video_normed_matrix != video_normed_matrix] = 0
    A = torch.bmm(video_normed_matrix, video_normed_matrix.permute(0, 2, 1))

    if normalize:
        sqrt_num_frames = torch.sqrt(batch_num_frames.double().to(device))
        Is = torch.eye(300).repeat(batch_size, 1, 1).double().to(device)
        Ds = 1/sqrt_num_frames.squeeze(1)[:, None, None]
        Ds = Is * Ds
        A_norm = torch.bmm(Ds, torch.bmm(A, Ds))
        return A_norm

    return A


def save_model(model, save_dir, save_file='model', step_count=0):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    path = os.path.join(save_dir, save_file + "_" + str(step_count) + ".torch")
    torch.save(model.state_dict(), path)
