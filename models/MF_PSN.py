import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils
import numpy as np
import pandas as pd
from collections import OrderedDict


class _DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_channels))
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv2d(in_channels, bn_size * growth_rate,
                                           kernel_size=1,
                                           stride=1, bias=False))
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True))
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3,
                                           stride=1, padding=1, bias=False))

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            self.add_module('denselayer%d' % (i + 1),
                            _DenseLayer(in_channels + growth_rate * i,
                                        growth_rate, bn_size))


class _Transition(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, out_channels,
                                          kernel_size=1,
                                          stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=1, stride=1))


class DenseNet_BC(nn.Module):
    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16),
                 bn_size=4, theta=0.5, num_classes=10):
        super(DenseNet_BC, self).__init__()

        # 初始的卷积为filter:2倍的growth_rate
        num_init_feature = 2 * growth_rate

        # 针对cifar-10
        if num_classes == 10:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_feature,
                                    kernel_size=7, stride=1,
                                    padding=3, bias=False)),
            ]))
        else:
            self.features = nn.Sequential(OrderedDict([
                ('conv0', nn.Conv2d(3, num_init_feature,
                                    kernel_size=3, stride=1,
                                    padding=1, bias=False)),
                ('norm0', nn.BatchNorm2d(num_init_feature)),
                ('relu0', nn.ReLU(inplace=True)),
                ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            ]))

        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):
            self.features.add_module('denseblock%d' % (i + 1),
                                     _DenseBlock(num_layers, num_feature,
                                                 bn_size, growth_rate))
            num_feature = num_feature + growth_rate * num_layers
            if i != len(block_config) - 1:
                self.features.add_module('transition%d' % (i + 1),
                                         _Transition(num_feature,
                                                     int(num_feature * theta)))
                num_feature = int(num_feature * theta)

        self.features.add_module('norm5', nn.BatchNorm2d(num_feature))
        self.features.add_module('relu5', nn.ReLU(inplace=True))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.features(x)
        return out


# DenseNet_BC for ImageNet
def DenseNet121():
    return DenseNet_BC(growth_rate=32, block_config=(1, 2, 4, 3), num_classes=1000)


def DenseNet169():
    return DenseNet_BC(growth_rate=32, block_config=(6, 12, 32, 32), num_classes=1000)


def DenseNet201():
    return DenseNet_BC(growth_rate=32, block_config=(6, 12, 48, 32), num_classes=1000)


def DenseNet161():
    return DenseNet_BC(growth_rate=48, block_config=(6, 12, 36, 24), num_classes=1000, )


# DenseNet_BC for cifar
def densenet_BC_100():
    return DenseNet_BC(growth_rate=12, block_config=(16, 16, 16))


# 从TransXNet导入的关键组件
class DynamicConv2d(nn.Module):
    def __init__(self, dim, kernel_size=3, reduction_ratio=4, num_groups=1, bias=True):
        super().__init__()
        assert num_groups > 1, f"num_groups {num_groups} should > 1."
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim // reduction_ratio, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim // reduction_ratio),
            nn.GELU(),
            nn.Conv2d(dim // reduction_ratio, dim * num_groups, kernel_size=1),
        )

        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        scale = self.proj(self.pool(x)).reshape(B, self.num_groups, C, self.K, self.K)
        scale = torch.softmax(scale, dim=1)
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K, self.K)

        if self.bias is not None:
            scale = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
            scale = torch.softmax(scale.reshape(B, self.num_groups, C), dim=1)
            bias = scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None

        x = torch.nn.functional.conv2d(
            x.reshape(1, -1, H, W),
            weight=weight,
            padding=self.K // 2,
            groups=B * C,
            bias=bias
        )

        return x.reshape(B, C, H, W)


class Attention(nn.Module):
    def __init__(self, dim, num_heads=1, qk_scale=None, attn_drop=0, sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio = sr_ratio
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.attn_drop = nn.Dropout(attn_drop)
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio + 3, stride=sr_ratio,
                          padding=(sr_ratio + 3) // 2, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
            )
        else:
            self.sr = nn.Identity()
        self.local_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x, relative_pos_enc=None):
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        kv = self.sr(x)
        kv = self.local_conv(kv) + kv
        k, v = torch.chunk(self.kv(kv), chunks=2, dim=1)
        k = k.reshape(B, self.num_heads, C // self.num_heads, -1)
        v = v.reshape(B, self.num_heads, C // self.num_heads, -1).transpose(-1, -2)
        attn = (q @ k) * self.scale
        if relative_pos_enc is not None:
            if attn.shape[2:] != relative_pos_enc.shape[2:]:
                relative_pos_enc = torch.nn.functional.interpolate(
                    relative_pos_enc, size=attn.shape[2:], mode='bicubic', align_corners=False)
            attn = attn + relative_pos_enc
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(-1, -2)
        return x.reshape(B, C, H, W)


class HybridTokenMixer(nn.Module):
    def __init__(self, dim, kernel_size=3, num_groups=2, num_heads=1, sr_ratio=1, reduction_ratio=8):
        super().__init__()
        assert dim % 2 == 0, f"dim {dim} should be divided by 2."

        self.local_unit = DynamicConv2d(
            dim=dim // 2, kernel_size=kernel_size, num_groups=num_groups)
        self.global_unit = Attention(
            dim=dim // 2, num_heads=num_heads, sr_ratio=sr_ratio)

        inner_dim = max(16, dim // reduction_ratio)
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, inner_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(inner_dim),
            nn.Conv2d(inner_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x, relative_pos_enc=None):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        x1 = self.local_unit(x1)
        x2 = self.global_unit(x2, relative_pos_enc)
        x = torch.cat([x1, x2], dim=1)
        x = self.proj(x) + x  # Skip-connection with Transformation Enhancement
        return x


class DynamicConvFeatureFusion(nn.Module):
    def __init__(self, dim, kernel_size=3, num_groups=2):
        super().__init__()
        self.dynamic_conv = DynamicConv2d(dim=dim, kernel_size=kernel_size, num_groups=num_groups)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, features):
        # 将特征转换为相同形状
        reshaped_features = []
        for feat in features:
            if len(feat.shape) == 1:  # 如果是扁平化的特征
                # 假设我们知道原始形状
                B, C, H, W = feat.shape[0], features[0].shape[1], features[0].shape[2], features[0].shape[3]
                reshaped_features.append(feat.view(B, C, H, W))
            else:
                reshaped_features.append(feat)

        # 简单融合作为初始特征
        stacked_features = torch.stack(reshaped_features, dim=0)
        avg_features = stacked_features.mean(dim=0)

        # 使用动态卷积增强
        enhanced = self.dynamic_conv(avg_features)
        return self.norm(enhanced)


class HybridFeatureFusion(nn.Module):
    def __init__(self, dim, kernel_size=3, num_groups=2, num_heads=1, sr_ratio=1):
        super().__init__()
        self.mixer = HybridTokenMixer(
            dim=dim,
            kernel_size=kernel_size,
            num_groups=num_groups,
            num_heads=num_heads,
            sr_ratio=sr_ratio
        )

    def forward(self, features, relative_pos_enc=None):
        # 将特征转换为相同形状
        reshaped_features = []
        for feat in features:
            if len(feat.shape) == 1:  # 如果是扁平化的特征
                # 假设我们知道原始形状
                B, C, H, W = feat.shape[0], features[0].shape[1], features[0].shape[2], features[0].shape[3]
                reshaped_features.append(feat.view(B, C, H, W))
            else:
                reshaped_features.append(feat)

        # 简单融合作为初始特征
        stacked_features = torch.stack(reshaped_features, dim=0)
        avg_features = stacked_features.mean(dim=0)

        # 使用混合Token混合器增强
        enhanced = self.mixer(avg_features, relative_pos_enc)
        return enhanced


class FeatExtractor1(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor1, self).__init__()
        self.other = other
        self.conv1 = model_utils.conv(batchNorm, c_in, 64, k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 64, 128, k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out_feat = self.conv3(out)
        n, c, h, w = out_feat.data.shape
        out_feat = out_feat.view(-1)
        return out_feat, [n, c, h, w]


class FeatExtractor2(nn.Module):
    def __init__(self, batchNorm=False, other={}):
        super(FeatExtractor2, self).__init__()
        self.other = other
        self.conv4 = model_utils.conv(batchNorm, 256, 256, k=3, stride=1, pad=1)
        self.conv5 = model_utils.conv(batchNorm, 256, 256, k=3, stride=1, pad=1)
        self.conv6 = model_utils.deconv(256, 128)
        self.conv7 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)

    def forward(self, x):
        out = self.conv4(x)
        out = self.conv5(out)
        out = self.conv6(out)
        out_feat = self.conv7(out)
        n, c, h, w = out_feat.data.shape
        out_feat = out_feat.view(-1)
        return out_feat, [n, c, h, w]


class Regressor(nn.Module):
    def __init__(self, batchNorm=False, other={}):
        super(Regressor, self).__init__()
        self.other = other
        self.deconv1 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)
        self.deconv2 = model_utils.deconv(128, 64)
        self.deconv3 = self._make_output(64, 3, k=3, stride=1, pad=1)
        self.deconv4 = DenseNet121()
        self.deconv5 = model_utils.deconv(188, 64)
        self.deconv6 = model_utils.conv(batchNorm, 64, 64, k=3, stride=2, pad=1)
        self.est_normal = self._make_output(64, 3, k=3, stride=1, pad=1)
        self.other = other

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x, shape):
        x = x.view(shape[0], shape[1], shape[2], shape[3])
        out = self.deconv1(x)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.deconv5(out)
        out = self.deconv6(out)
        normal = self.est_normal(out)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal


class MF_PSN(nn.Module):
    def __init__(self, fuse_type='advanced', batchNorm=False, c_in=3, other={}):
        super(MF_PSN, self).__init__()
        self.extractor1 = FeatExtractor1(batchNorm, c_in, other)
        self.extractor2 = FeatExtractor2(batchNorm, other)
        self.regressor = Regressor(batchNorm, other)
        self.c_in = c_in
        self.fuse_type = fuse_type
        self.other = other

        # 添加特征融合模块
        if fuse_type == 'advanced':
            self.fusion1 = DynamicConvFeatureFusion(dim=128, kernel_size=3, num_groups=2)
            self.fusion2 = HybridFeatureFusion(dim=128, kernel_size=3, num_groups=2, num_heads=1, sr_ratio=1)

            # 用于HybridTokenMixer的相对位置编码
            self.relative_pos_enc = nn.Parameter(torch.zeros(1, 1, 32, 32), requires_grad=True)
            nn.init.trunc_normal_(self.relative_pos_enc, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        img = x[0]
        img_split = torch.split(img, 3, 1)
        if len(x) > 1:  # 有光照信息
            light = x[1]
            light_split = torch.split(light, 3, 1)

            # 第一阶段特征提取和融合
        feats = []
        shapes = []
        for i in range(len(img_split)):
            net_in = img_split[i] if len(x) == 1 else torch.cat([img_split[i], light_split[i]], 1)
            feat, shape = self.extractor1(net_in)
            feats.append(feat)
            shapes.append(shape)

        # 使用高级特征融合或原始融合方法
        if self.fuse_type == 'advanced':
            # 将特征转换为原始形状用于融合
            reshaped_feats = [f.view(shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3])
                              for i, f in enumerate(feats)]
            feat_fused = self.fusion1(reshaped_feats)
        elif self.fuse_type == 'mean':
            feat_fused = torch.stack(feats, 1).mean(1)
            feat_fused = feat_fused.view(shapes[0][0], shapes[0][1], shapes[0][2], shapes[0][3])
        elif self.fuse_type == 'max':
            feat_fused, _ = torch.stack(feats, 1).max(1)
            feat_fused = feat_fused.view(shapes[0][0], shapes[0][1], shapes[0][2], shapes[0][3])

        # 第二阶段特征处理和融合
        featss = []
        shapess = []
        for i in range(len(img_split)):
            # 将第一阶段的特征重塑为2D形式
            feat = feats[i].view(shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3])
            # 连接特征
            featt = torch.cat((feat, feat_fused), 1)
            # 通过第二个特征提取器
            featt, shapee = self.extractor2(featt)
            featss.append(featt)
            shapess.append(shapee)

        # 第二阶段特征融合
        if self.fuse_type == 'advanced':
            # 将特征转换为原始形状用于融合
            reshaped_featss = [f.view(shapess[i][0], shapess[i][1], shapess[i][2], shapess[i][3])
                               for i, f in enumerate(featss)]
            feat_fusedd = self.fusion2(reshaped_featss,
                                       self.relative_pos_enc if hasattr(self, 'relative_pos_enc') else None)
            # 转换回1D形式
            feat_fusedd = feat_fusedd.view(-1)
        elif self.fuse_type == 'mean':
            feat_fusedd = torch.stack(featss, 1).mean(1)
        elif self.fuse_type == 'max':
            feat_fusedd, _ = torch.stack(featss, 1).max(1)

        # 回归器处理
        normal = self.regressor(feat_fusedd, shapess[0])
        return normal