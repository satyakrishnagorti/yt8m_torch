import torch
import torch.nn as nn


class FrameLevelLogistic(nn.Module):

    def __init__(self, vocab_size, num_features=1152):
        super(FrameLevelLogistic, self).__init__()
        self.vocab_size = vocab_size
        self.fc = nn.Linear(num_features, vocab_size, bias=True)

    def forward(self, model_input, num_frames):
        """
        :param model_input: dim (B, 300, feature_size)
        :param num_frames: dim (B, 1)
        :return: output: dim (B, 300, vocab_size)
        """
        feature_size = model_input.shape[2]
        num_frames = num_frames.view(-1, 1)
        avg_pooled = torch.sum(model_input, dim=1) / num_frames
        return self.fc(avg_pooled)