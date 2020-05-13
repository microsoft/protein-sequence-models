import torch.nn as nn


class MaskedConv1d(nn.Conv1d):
    """ A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.

         Shape:
            Input: (N, in_channels, L)
            input_mask: (N, 1), optional
            Output: (N, out_channels, L)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int=1, dilation: int=1, groups: int=1,
                 bias: bool=True):
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                           groups=groups, bias=bias, padding=padding)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x)


class MaskedCausalConv1d(nn.Conv1d):

    """ A masked causal 1-D convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically
    to shift the inputs.

         Shape:
            Input: (N, in_channels, L)
            input_mask: (N, 1), optional
            Output: (N, out_channels, L)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int=1, dilation: int=1, groups: int=1,
                 bias: bool=True):
        super().__init__(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=0, dilation=dilation,
                                                 groups=groups, bias=bias)
        self.left_padding = dilation * (kernel_size - 1)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        x = nn.functional.pad(x.unsqueeze(2), [self.left_padding, 0, 0, 0]).squeeze(2)
        return super().forward(x)