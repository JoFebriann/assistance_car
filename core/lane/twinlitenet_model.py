from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_CHANNELS_BY_SIZE = {
    "nano": {"p": 1, "q": 1, "channels": [4, 8, 16, 32, 64]},
    "small": {"p": 2, "q": 3, "channels": [8, 16, 32, 64, 128]},
    "medium": {"p": 3, "q": 5, "channels": [16, 32, 64, 128, 256]},
    "large": {"p": 5, "q": 7, "channels": [32, 64, 128, 256, 512]},
}


class _Config:
    channel_img = 3

    @staticmethod
    def get(model_size: str) -> dict:
        if model_size not in _CHANNELS_BY_SIZE:
            raise ValueError(f"Unsupported TwinLiteNetPlus model_size: {model_size}")
        return _CHANNELS_BY_SIZE[model_size]


def patch_split(inp: torch.Tensor, bin_size: tuple[int, int]) -> torch.Tensor:
    b, c, h, w = inp.size()
    bin_num_h = bin_size[0]
    bin_num_w = bin_size[1]
    rh = h // bin_num_h
    rw = w // bin_num_w
    out = inp.view(b, c, bin_num_h, rh, bin_num_w, rw)
    out = out.permute(0, 2, 4, 3, 5, 1).contiguous()
    return out.view(b, -1, rh, rw, c)


def patch_recover(inp: torch.Tensor, bin_size: tuple[int, int]) -> torch.Tensor:
    b, _, rh, rw, c = inp.size()
    bin_num_h = bin_size[0]
    bin_num_w = bin_size[1]
    h = rh * bin_num_h
    w = rw * bin_num_w
    out = inp.view(b, bin_num_h, bin_num_w, rh, rw, c)
    out = out.permute(0, 5, 1, 3, 2, 4).contiguous()
    return out.view(b, c, h, w)


class GCN(nn.Module):
    def __init__(self, num_node: int, num_channel: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_node, num_node, kernel_size=1, bias=False)
        self.relu = nn.PReLU(num_node)
        self.conv2 = nn.Linear(num_channel, num_channel, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.relu(out + x)
        out = self.conv2(out)
        return out


class CAAM(nn.Module):
    def __init__(self, feat_in: int, num_classes: int, bin_size: tuple[int, int], norm_layer):
        super().__init__()
        feat_inner = feat_in // 2
        self.bin_size = bin_size
        self.conv_cam = nn.Conv2d(feat_in, num_classes, kernel_size=1)
        self.pool_cam = nn.AdaptiveAvgPool2d(bin_size)
        self.sigmoid = nn.Sigmoid()

        bin_num = bin_size[0] * bin_size[1]
        self.gcn = GCN(bin_num, feat_in)
        self.fuse = nn.Conv2d(bin_num, 1, kernel_size=1)
        self.proj_query = nn.Linear(feat_in, feat_inner)
        self.proj_key = nn.Linear(feat_in, feat_inner)
        self.proj_value = nn.Linear(feat_in, feat_inner)

        self.conv_out = nn.Sequential(
            nn.Conv2d(feat_inner, feat_in, kernel_size=1, bias=False),
            norm_layer(feat_in),
            nn.PReLU(feat_in),
        )
        self.relu = nn.PReLU(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cam = self.conv_cam(x)
        cls_score = self.sigmoid(self.pool_cam(cam))

        residual = x
        cam = patch_split(cam, self.bin_size)
        x = patch_split(x, self.bin_size)

        b = cam.shape[0]
        rh = cam.shape[2]
        rw = cam.shape[3]
        k = cam.shape[-1]
        c = x.shape[-1]
        cam = cam.view(b, -1, rh * rw, k)
        x = x.view(b, -1, rh * rw, c)

        bin_confidence = cls_score.view(b, k, -1).transpose(1, 2).unsqueeze(3)
        pixel_confidence = F.softmax(cam, dim=2)

        local_feats = torch.matmul(pixel_confidence.transpose(2, 3), x) * bin_confidence
        local_feats = self.gcn(local_feats)
        global_feats = self.fuse(local_feats)
        global_feats = self.relu(global_feats).repeat(1, x.shape[1], 1, 1)

        query = self.proj_query(x)
        key = self.proj_key(local_feats)
        value = self.proj_value(global_feats)

        aff_map = torch.matmul(query, key.transpose(2, 3))
        aff_map = F.softmax(aff_map, dim=-1)
        out = torch.matmul(aff_map, value)

        out = out.view(b, -1, rh, rw, value.shape[-1])
        out = patch_recover(out, self.bin_size)

        out_conv = self.conv_out(out)
        return residual + out_conv


class ConvBatchnormRelu(nn.Module):
    def __init__(
        self,
        n_in: int,
        n_out: int,
        k_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        padding = (k_size - 1) // 2
        self.conv = nn.Conv2d(
            n_in,
            n_out,
            k_size,
            stride=stride,
            padding=padding,
            bias=False,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(n_out)
        self.act = nn.PReLU(n_out)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else None

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        out = self.conv(inp)
        out = self.bn(out)
        out = self.act(out)
        if self.dropout:
            out = self.dropout(out)
        return out


class DilatedConv(nn.Module):
    def __init__(self, n_in: int, n_out: int, k_size: int, stride: int = 1, d: int = 1, groups: int = 1):
        super().__init__()
        padding = ((k_size - 1) // 2) * d
        self.conv = nn.Conv2d(
            n_in,
            n_out,
            k_size,
            stride=stride,
            padding=padding,
            bias=False,
            dilation=d,
            groups=groups,
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.conv(inp)


class BatchnormRelu(nn.Module):
    def __init__(self, n_out: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(n_out, eps=1e-3)
        self.act = nn.PReLU(n_out)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(inp))


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, n_in: int, n_out: int, kernel_size: int = 3, stride: int = 1, dilation: int = 1):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.depthwise = nn.Conv2d(
            n_in,
            n_in,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=n_in,
            bias=False,
        )
        self.pointwise = nn.Conv2d(n_in, n_out, 1, 1, 0, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        return self.pointwise(x)


class StrideESP(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        n = n_out // 5
        n1 = n_out - 4 * n
        self.c1 = DilatedConv(n_in, n, 3, 2)
        self.d1 = DilatedConv(n, n1, 3, 1, 1)
        self.d2 = DilatedConv(n, n, 3, 1, 2)
        self.d4 = DilatedConv(n, n, 3, 1, 4)
        self.d8 = DilatedConv(n, n, 3, 1, 8)
        self.d16 = DilatedConv(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(n_out, eps=1e-3)
        self.act = nn.PReLU(n_out)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        out1 = self.c1(inp)
        d1 = self.d1(out1)
        d2 = self.d2(out1)
        d4 = self.d4(out1)
        d8 = self.d8(out1)
        d16 = self.d16(out1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        return self.act(self.bn(combine))


class DepthwiseESP(nn.Module):
    def __init__(self, n_in: int, n_out: int, add: bool = True):
        super().__init__()
        n = max(n_out // 5, 1)
        n1 = max(n_out - 4 * n, 1)
        self.c1 = DepthwiseSeparableConv(n_in, n, 1, 1)
        self.d1 = DepthwiseSeparableConv(n, n1, 3, 1, 1)
        self.d2 = DepthwiseSeparableConv(n, n, 3, 1, 2)
        self.d4 = DepthwiseSeparableConv(n, n, 3, 1, 4)
        self.d8 = DepthwiseSeparableConv(n, n, 3, 1, 8)
        self.d16 = DepthwiseSeparableConv(n, n, 3, 1, 16)
        self.bn = BatchnormRelu(n_out)
        self.add = add

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        out1 = self.c1(inp)
        d1 = self.d1(out1)
        d2 = self.d2(out1)
        d4 = self.d4(out1)
        d8 = self.d8(out1)
        d16 = self.d16(out1)

        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        if self.add:
            combine = inp + combine
        return self.bn(combine)


class AvgDownsampler(nn.Module):
    def __init__(self, sampling_times: int):
        super().__init__()
        self.pool = nn.ModuleList([nn.AvgPool2d(3, stride=2, padding=1) for _ in range(sampling_times)])

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        for pool in self.pool:
            inp = pool(inp)
        return inp


class Encoder(nn.Module):
    def __init__(self, model_size: str):
        super().__init__()
        model_cfg = _Config.get(model_size)
        channels = model_cfg["channels"]

        self.level1 = ConvBatchnormRelu(_Config.channel_img, channels[0], stride=2)
        self.sample1 = AvgDownsampler(1)
        self.sample2 = AvgDownsampler(2)

        self.b1 = ConvBatchnormRelu(channels[0] + _Config.channel_img, channels[1])
        self.level2_0 = StrideESP(channels[1], channels[2])
        self.level2 = nn.ModuleList([DepthwiseESP(channels[2], channels[2]) for _ in range(model_cfg["p"])])
        self.b2 = ConvBatchnormRelu(channels[3] + _Config.channel_img, channels[3] + _Config.channel_img)

        self.level3_0 = StrideESP(channels[3] + _Config.channel_img, channels[3])
        self.level3 = nn.ModuleList([DepthwiseESP(channels[3], channels[3]) for _ in range(model_cfg["q"])])
        self.b3 = ConvBatchnormRelu(channels[4], channels[2])

    def forward(self, inp: torch.Tensor):
        out0 = self.level1(inp)
        inp1 = self.sample1(inp)
        inp2 = self.sample2(inp)
        out0_cat = self.b1(torch.cat([out0, inp1], 1))

        out1_0 = self.level2_0(out0_cat)
        out1 = out1_0
        for layer in self.level2:
            out1 = layer(out1)

        out1_cat = self.b2(torch.cat([out1, out1_0, inp2], 1))
        out2_0 = self.level3_0(out1_cat)
        out2 = out2_0
        for layer in self.level3:
            out2 = layer(out2)

        out2_cat = torch.cat([out2_0, out2], 1)
        out_encoder = self.b3(out2_cat)
        return out_encoder, inp1, inp2


class UpSimpleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, padding=0, output_padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)
        self.act = nn.PReLU(out_channels)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        out = self.deconv(inp)
        out = self.bn(out)
        return self.act(out)


class UpConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, sub_dim: int = 3, last: bool = False, kernel_size: int = 3):
        super().__init__()
        self.last = last
        self.up_conv = UpSimpleBlock(in_channels, out_channels)
        if not last:
            self.conv1 = ConvBatchnormRelu(out_channels + sub_dim, out_channels, kernel_size)
        self.conv2 = ConvBatchnormRelu(out_channels, out_channels, kernel_size)

    def forward(self, x: torch.Tensor, ori_img: torch.Tensor | None = None) -> torch.Tensor:
        x = self.up_conv(x)
        if not self.last:
            if ori_img is None:
                raise RuntimeError("ori_img is required when last=False")
            x = torch.cat([x, ori_img], dim=1)
            x = self.conv1(x)
        return self.conv2(x)


class TwinLiteNetPlus(nn.Module):
    def __init__(self, model_size: str = "small"):
        super().__init__()
        channels = _Config.get(model_size)["channels"]
        self.encoder = Encoder(model_size)
        self.caam = CAAM(feat_in=channels[2], num_classes=channels[2], bin_size=(2, 4), norm_layer=nn.BatchNorm2d)
        self.conv_caam = ConvBatchnormRelu(channels[2], channels[1])

        self.up_1_da = UpConvBlock(channels[1], channels[0])
        self.up_2_da = UpConvBlock(channels[0], 8)
        self.out_da = UpConvBlock(8, 2, last=True)

        self.up_1_ll = UpConvBlock(channels[1], channels[0])
        self.up_2_ll = UpConvBlock(channels[0], 8)
        self.out_ll = UpConvBlock(8, 2, last=True)

    def forward(self, inp: torch.Tensor):
        out_encoder, inp1, inp2 = self.encoder(inp)

        out_caam = self.caam(out_encoder)
        out_caam = self.conv_caam(out_caam)

        out_da = self.up_1_da(out_caam, inp2)
        out_da = self.up_2_da(out_da, inp1)
        out_da = self.out_da(out_da)

        out_ll = self.up_1_ll(out_caam, inp2)
        out_ll = self.up_2_ll(out_ll, inp1)
        out_ll = self.out_ll(out_ll)

        return out_da, out_ll


def net_params(model: nn.Module) -> int:
    return int(np.sum([np.prod(parameter.size()) for parameter in model.parameters()]))
