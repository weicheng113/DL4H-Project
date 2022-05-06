import torch.nn as nn
import torch

# https://jinglescode.github.io/2020/11/01/how-convolutional-layers-work-deep-learning-neural-networks/


class TestConv1d(nn.Module):
    def __init__(self):
        super(TestConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=1, bias=False)
        self.init_weights()

    def forward(self, x):
        return self.conv(x)

    def init_weights(self):
        print(f"conv.weight.shape: {self.conv.weight.shape}")
        self.conv.weight.data[0, 0, 0] = 2.
        self.conv.weight.data[0, 1, 0] = 2.
        self.conv.weight.data[1, 0, 0] = 4.
        self.conv.weight.data[1, 1, 0] = 4.
        self.conv.weight.data[2, 0, 0] = 6.
        self.conv.weight.data[2, 1, 0] = 6.
        self.conv.weight.data[3, 0, 0] = 8.
        self.conv.weight.data[3, 1, 0] = 8.


def test_conv1d():
    in_x = torch.tensor([[[1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60]]]).float()
    print("in_x.shape", in_x.shape)
    print(in_x)

    net = TestConv1d()
    out_y = net(in_x)

    print("out_y.shape", out_y.shape)
    print(out_y)


class TestConv1dGroups(nn.Module):
    def __init__(self):
        super(TestConv1dGroups, self).__init__()
        # 6 in_channels are divided into 2 groups, each group will have 3 parameters for dot product.
        # 4 out_channels will go through in feature maps 4 times. Each time it goes one group.
        # It takes 2 times to go through whole in_channels. Therefore, in total it will go through 2 rounds.
        self.conv = nn.Conv1d(in_channels=6, out_channels=4, kernel_size=1, groups=2, bias=False)
        self.init_weights()

    def forward(self, x):
        return self.conv(x)

    def init_weights(self):
        print(f"conv.weight.shape: {self.conv.weight.shape}")
        self.conv.weight.data[0, 0, 0] = 1.
        self.conv.weight.data[0, 1, 0] = 1.
        self.conv.weight.data[0, 2, 0] = 1.
        self.conv.weight.data[1, 0, 0] = 2.
        self.conv.weight.data[1, 1, 0] = 2.
        self.conv.weight.data[1, 2, 0] = 2.
        self.conv.weight.data[2, 0, 0] = 3.
        self.conv.weight.data[2, 1, 0] = 3.
        self.conv.weight.data[2, 2, 0] = 3.
        self.conv.weight.data[3, 0, 0] = 4.
        self.conv.weight.data[3, 1, 0] = 4.
        self.conv.weight.data[3, 2, 0] = 4.


def test_conv1d_groups():
    torch.set_printoptions(precision=0, sci_mode=False)
    in_x = torch.tensor([[[1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 50, 60],
                          [100, 200, 300, 400, 500, 600], [1000, 2000, 3000, 4000, 5000, 6000],
                          [10000, 20000, 30000, 40000, 50000, 60000], [100000, 200000, 300000, 400000, 500000, 600000]]]).float()
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


def test_repeat_interleave():
    y = torch.tensor([[1, 2], [3, 4]])
    print(f"y.shape: {y.shape}, y: {y}")

    repeated_dim0 = y.repeat_interleave(2, dim=0)
    print(f"repeated_dim0.shape: {repeated_dim0.shape}, repeated_dim0: {repeated_dim0}")

    repeated_dim1 = torch.repeat_interleave(y, 3, dim=1)
    print(f"repeated_dim1.shape: {repeated_dim1.shape}, repeated_dim1: {repeated_dim1}")


def test_permute():
    batch_size = 3
    samples_per_patient = 2
    feature_size = 4
    x = torch.arange(start=1, end=25).reshape(shape=(batch_size, samples_per_patient, feature_size))
    print(f"x.shape: {x.shape}")
    print(x)
    x_permuted = x.permute(0, 2, 1)
    print(f"x_permuted.shape: {x_permuted.shape}")
    print(x_permuted)
    x_split = torch.split(x_permuted, split_size_or_sections=feature_size//2, dim=1)
    print(f"x_split: ")
    print(f"{x_split}")
    print(f"torch.stack(x_split, dim=2): {torch.stack(x_split, dim=2).shape}")
    print(torch.stack(x_split, dim=2))
    x_stack = torch.stack(x_split, dim=2).reshape(shape=(batch_size, feature_size, samples_per_patient))
    print("x_stack:")
    print(x_stack)


if __name__ == '__main__':
    # test_conv1d()
    test_conv1d_groups()
    # test_conv1d_dilation()
    # test_repeat_interleave()
    # test_permute()
