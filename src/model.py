import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NUM_CLASSES


class DBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=True):
        super().__init__()
        self.stride = stride
        self.batch_norm = batch_norm
        padding = kernel_size // 2 if stride == 1 else 0

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not batch_norm,
        )
        if self.batch_norm:
            self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        if self.stride != 1:
            x = F.pad(x, (0, 1, 0, 1))
        x = self.conv(x)
        if self.batch_norm:
            x = self.bn(x)
        return self.act(x)


class ResUnit(nn.Module):
    def __init__(self, in_channels, filters):
        super().__init__()
        self.dbl1 = DBL(in_channels, filters // 2, 1)
        self.dbl2 = DBL(filters // 2, filters, 3)

    def forward(self, x):
        return x + self.dbl2(self.dbl1(x))


class ResN(nn.Module):
    def __init__(self, in_channels, filters, blocks):
        super().__init__()
        self.dbl = DBL(in_channels, filters, kernel_size=3, stride=2)
        self.blocks = nn.Sequential(*[ResUnit(filters, filters) for _ in range(blocks)])

    def forward(self, x):
        return self.blocks(self.dbl(x))


class Darknet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.dbl1 = DBL(in_channels, 32, 3)
        self.res1 = ResN(32, 64, 1)
        self.res2 = ResN(64, 128, 2)
        self.res3 = ResN(128, 256, 8)
        self.res4 = ResN(256, 512, 8)
        self.res5 = ResN(512, 1024, 4)

    def forward(self, x):
        x = self.dbl1(x)
        x = self.res1(x)
        x = self.res2(x)
        route_1 = self.res3(x)
        route_2 = self.res4(route_1)
        route_3 = self.res5(route_2)
        return route_1, route_2, route_3


class YoloHead(nn.Module):
    def __init__(self, in_channels, skip_channels, filters):
        super().__init__()
        self.has_skip = skip_channels > 0
        if self.has_skip:
            self.dbl_up = DBL(in_channels, filters, 1)
            self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
            in_channels = filters + skip_channels

        self.block = nn.Sequential(
            DBL(in_channels, filters, 1),
            DBL(filters, filters * 2, 3),
            DBL(filters * 2, filters, 1),
            DBL(filters, filters * 2, 3),
            DBL(filters * 2, filters, 1),
        )

    def forward(self, x, x_skip=None):
        if self.has_skip and x_skip is not None:
            x = self.dbl_up(x)
            x = self.upsample(x)
            x = torch.cat([x, x_skip], dim=1)
        return self.block(x)


class YoloHeadOutput(nn.Module):
    def __init__(self, in_channels, filters, anchors_count, classes):
        super().__init__()
        self.anchors_count = anchors_count
        self.classes = classes
        self.dbl = DBL(in_channels, filters * 2, 3)
        self.conv_out = nn.Conv2d(filters * 2, anchors_count * (classes + 5), 1)

    def forward(self, x):
        x = self.dbl(x)
        x = self.conv_out(x)
        batch_size, _, grid_h, grid_w = x.shape
        x = x.view(batch_size, self.anchors_count, self.classes + 5, grid_h, grid_w)
        return x.permute(0, 3, 4, 1, 2)


class YOLOv3(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = Darknet(3)

        self.head1 = YoloHead(1024, 0, 512)
        self.out1 = YoloHeadOutput(512, 512, 3, num_classes)

        self.head2 = YoloHead(512, 512, 256)
        self.out2 = YoloHeadOutput(256, 256, 3, num_classes)

        self.head3 = YoloHead(256, 256, 128)
        self.out3 = YoloHeadOutput(128, 128, 3, num_classes)

    def forward(self, x):
        route_1, route_2, route_3 = self.backbone(x)

        h1 = self.head1(route_3)
        out1 = self.out1(h1)

        h2 = self.head2(h1, route_2)
        out2 = self.out2(h2)

        h3 = self.head3(h2, route_1)
        out3 = self.out3(h3)

        return out1, out2, out3