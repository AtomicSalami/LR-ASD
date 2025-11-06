import torch
import torch.nn as nn


class Fusion(nn.Module):
    def __init__(self, channel):
        super(Fusion, self).__init__()
        self.channel = channel
        self.attention = nn.Sequential(
            nn.Linear(channel * 2, channel),
            nn.ReLU(),
            nn.Linear(channel, channel),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        g = self.attention(x)
        return g * x1 + (1 - g) * x2


class Detector(nn.Module):
    def __init__(self, channel, causal: bool = False):
        super(Detector, self).__init__()
        self.gru_forward = nn.GRU(input_size=channel,
                                  hidden_size=channel // 4,
                                  num_layers=1,
                                  bidirectional=False,
                                  bias=True,
                                  batch_first=True)
        self.gru_backward = nn.GRU(input_size=channel,
                                   hidden_size=channel // 4,
                                   num_layers=1,
                                   bidirectional=False,
                                   bias=True,
                                   batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.attention = Fusion(channel // 2)
        self.causal = causal
        self.__init_weight()

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1, _ = self.gru_forward(self.drop(x))
        if self.causal:
            return x1
        x_rev = torch.flip(x, dims=[1])
        x2, _ = self.gru_backward(self.drop(x_rev))
        x2 = torch.flip(x2, dims=[1])
        return self.attention(x1, x2)
