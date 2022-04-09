import torch.nn as nn
import torch


class TestConv1dGroups(nn.Module):
    def __init__(self):
        super(TestConv1dGroups, self).__init__()
        self.conv = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=1, groups=2, bias=False)
        self.init_weights()

    def forward(self, x):
        return self.conv(x)

    def init_weights(self):
        print(f"conv.weight.shape: {self.conv.weight.shape}")
        self.conv.weight.data[0, 0, 0] = 2.
        self.conv.weight.data[1, 0, 0] = 4.
        self.conv.weight.data[2, 0, 0] = 6.
        self.conv.weight.data[3, 0, 0] = 8.


def test_conv1d_groups():
    in_x = torch.tensor([[[1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60]]]).float()
    print("in_x.shape", in_x.shape)
    print(in_x)

    net = TestConv1dGroups()
    out_y = net(in_x)

    print("out_y.shape", out_y.shape)
    print(out_y)


class TestConv1dDilation(nn.Module):
    def __init__(self):
        super(TestConv1dDilation, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, dilation=2, bias=False)
        self.init_weights()

    def forward(self, x):
        return self.conv(x)

    def init_weights(self):
        new_weights = torch.ones(self.conv.weight.shape) * 2.  # new_weights = [[[2, 2, 2]]]
        self.conv.weight = torch.nn.Parameter(new_weights, requires_grad=False)


def test_conv1d_dilation():
    in_x = torch.tensor([[[1, 2, 3, 4, 5, 6]]]).float()
    print("in_x.shape", in_x.shape)
    print(in_x)

    net = TestConv1dDilation()
    out_y = net(in_x)

    print("out_y.shape", out_y.shape)
    print(out_y)
    # tensor([[[18 = 1*2 + 3*2 + 5*2, 24 = 2*2 + 2*4 + 2*6]]])


if __name__ == '__main__':
    test_conv1d_groups()
    # test_conv1d_dilation()

