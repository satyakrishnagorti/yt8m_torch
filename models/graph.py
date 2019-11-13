import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphModel(nn.Module):

    def __init__(self, vocab_size, num_features=1152, out_features=1152, g_bias=True):
        super(GraphModel, self).__init__()
        self.vocab_size = vocab_size
        self.start_conv1d = nn.Conv1d(num_features, num_features, kernel_size=5, stride=5)
        self.fc = nn.Linear(num_features, vocab_size, bias=True)
        self.graph_weight = Parameter(torch.FloatTensor(num_features, out_features))
        self.g_bias = g_bias
        if g_bias:
            self.graph_bias = Parameter(torch.FloatTensor(out_features))
        self.initialize()

    def initialize(self):
        torch.nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        torch.nn.init.uniform_(self.fc.bias, a=0.0, b=0.01)
        torch.nn.init.normal_(self.graph_weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.start_conv1d.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.start_conv1d.bias, mean=0.0, std=0.01)
        torch.nn.init.uniform_(self.graph_bias, a=0.0, b=0.01)

    def forward(self, model_input, num_frames, A_norm):
        """
        :param model_input: dim (B, 300, feature_size)
        :param A: dim(B, 300, 300) D A D' normalized similarity matrix
        :param num_frames: dim (B, 1)
        :return output: dim (B, 300, vocab_size)
        """
        feature_size = model_input.shape[2]
        batch_size = A_norm.shape[0]
        num_frames = num_frames.view(-1, 1)

        aggregation = torch.bmm(A_norm, model_input).reshape(-1, feature_size)
        graph_out = torch.mm(aggregation, self.graph_weight).reshape(batch_size, 300, feature_size)
        if self.g_bias:
            graph_out = self.graph_bias + graph_out
        # graph_out dims: (B, 300, 1024)

        # model_input: (B, 1024, 300)
        graph_out = graph_out.permute(0, 2, 1)

        # conv1d_out: (B, 60, 1024)
        conv1d_out = F.relu(self.start_conv1d(graph_out).permute(0, 2, 1))
        fc_out = F.sigmoid(self.fc(conv1d_out))
        return fc_out


class GraphModel_ConvFC(nn.Module):

    def __init__(self, vocab_size, num_features=1152, out_features=1152, g_bias=True):
        super(GraphModel_ConvFC, self).__init__()
        self.vocab_size = vocab_size
        self.start_conv1d = nn.Conv1d(num_features, num_features, kernel_size=1, stride=1)
        self.fc = nn.Linear(num_features, vocab_size, bias=True)
        self.graph_weight = Parameter(torch.FloatTensor(num_features, out_features))
        self.g_bias = g_bias
        if g_bias:
            self.graph_bias = Parameter(torch.FloatTensor(out_features))
        self.initialize()

    def initialize(self):
        torch.nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
        torch.nn.init.uniform_(self.fc.bias, a=0.0, b=0.01)
        torch.nn.init.normal_(self.graph_weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.start_conv1d.weight, mean=0.0, std=0.01)
        torch.nn.init.normal_(self.start_conv1d.bias, mean=0.0, std=0.01)
        torch.nn.init.uniform_(self.graph_bias, a=0.0, b=0.01)

    def forward(self, model_input, num_frames, A_norm):
        """
        :param model_input: dim (B, 300, feature_size)
        :param A: dim(B, 300, 300) D A D' normalized similarity matrix
        :param num_frames: dim (B, 1)
        :return output: dim (B, 300, vocab_size)
        """
        feature_size = model_input.shape[2]
        batch_size = A_norm.shape[0]
        num_frames = num_frames.view(-1, 1)

        aggregation = torch.bmm(A_norm, model_input).reshape(-1, feature_size)
        graph_out = torch.mm(aggregation, self.graph_weight).reshape(batch_size, 300, feature_size)
        if self.g_bias:
            graph_out = self.graph_bias + graph_out
        # graph_out dims: (B, 300, 1024)

        # model_input: (B, 1024, 300)
        graph_out = graph_out.permute(0, 2, 1)

        # conv1d_out: (B, 60, 1024)
        conv1d_out = F.relu(self.start_conv1d(graph_out).permute(0, 2, 1))
        fc_out = F.sigmoid(self.fc(conv1d_out))
        return fc_out


class GraphVideoModel(nn.Module):
    def __init__(self, vocab_size, num_features=1152, out_features=1152, g_bias=True):
        super(GraphVideoModel, self).__init__()
        self.vocab_size = vocab_size




