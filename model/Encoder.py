import torch
import torch.nn as nn

# choose a safe group count (4 when possible, else 1)
def _g(in_ch, out_ch):
    return 4 if (in_ch % 4 == 0 and out_ch % 4 == 0) else 1


class Audio_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_1, kernel_2):
        super(Audio_Block, self).__init__()
        self.relu = nn.ReLU()
        self.padding_1 = int((kernel_1 - 1) / 2)
        self.padding_2 = int((kernel_2 - 1) / 2)

        self.m_1 = nn.Conv2d(in_channels, out_channels // 2,
                             kernel_size=(kernel_1, 1),
                             padding=(self.padding_1, 0), bias=False,
                             groups=_g(in_channels, out_channels // 2))
        self.m_norm_1 = nn.BatchNorm2d(out_channels // 2, momentum=0.01, eps=0.001)

        self.m_2 = nn.Conv2d(out_channels // 2, out_channels,
                             kernel_size=(kernel_2, 1),
                             padding=(self.padding_2, 0), bias=False,
                             groups=_g(out_channels // 2, out_channels))
        self.m_norm_2 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

        self.t_1 = nn.Conv2d(out_channels, out_channels,
                             kernel_size=(1, kernel_1),
                             padding=(0, self.padding_1), bias=False,
                             groups=_g(out_channels, out_channels))
        self.t_norm_1 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

        self.t_2 = nn.Conv2d(out_channels, out_channels,
                             kernel_size=(1, kernel_2),
                             padding=(0, self.padding_2), bias=False,
                             groups=_g(out_channels, out_channels))
        self.t_norm_2 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

    def forward(self, x):
        x = self.relu(self.m_norm_1(self.m_1(x)))
        x = self.relu(self.m_norm_2(self.m_2(x)))
        x = self.relu(self.t_norm_1(self.t_1(x)))
        x = self.relu(self.t_norm_2(self.t_2(x)))
        return x


class Visual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_1, kernel_2, is_down=False):
        super(Visual_Block, self).__init__()
        self.relu = nn.ReLU()
        self.padding_1 = int((kernel_1 - 1) / 2)
        self.padding_2 = int((kernel_2 - 1) / 2)

        if is_down:
            self.s_1 = nn.Conv3d(in_channels, out_channels // 2,
                                 kernel_size=(1, kernel_1, kernel_1),
                                 stride=(1, 2, 2),
                                 padding=(0, self.padding_1, self.padding_1),
                                 bias=False,
                                 groups=_g(in_channels, out_channels // 2))
        else:
            self.s_1 = nn.Conv3d(in_channels, out_channels // 2,
                                 kernel_size=(1, kernel_1, kernel_1),
                                 padding=(0, self.padding_1, self.padding_1),
                                 bias=False,
                                 groups=_g(in_channels, out_channels // 2))
        self.s_norm_1 = nn.BatchNorm3d(out_channels // 2, momentum=0.01, eps=0.001)

        self.s_2 = nn.Conv3d(out_channels // 2, out_channels,
                             kernel_size=(1, kernel_2, kernel_2),
                             padding=(0, self.padding_2, self.padding_2),
                             bias=False,
                             groups=_g(out_channels // 2, out_channels))
        self.s_norm_2 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

        self.t_1 = nn.Conv3d(out_channels, out_channels,
                             kernel_size=(kernel_1, 1, 1),
                             padding=(self.padding_1, 0, 0),
                             bias=False,
                             groups=_g(out_channels, out_channels))
        self.t_norm_1 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

        self.t_2 = nn.Conv3d(out_channels, out_channels,
                             kernel_size=(kernel_2, 1, 1),
                             padding=(self.padding_2, 0, 0),
                             bias=False,
                             groups=_g(out_channels, out_channels))
        self.t_norm_2 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

    def forward(self, x):
        x = self.relu(self.s_norm_1(self.s_1(x)))
        x = self.relu(self.s_norm_2(self.s_2(x)))
        x = self.relu(self.t_norm_1(self.t_1(x)))
        x = self.relu(self.t_norm_2(self.t_2(x)))
        return x
