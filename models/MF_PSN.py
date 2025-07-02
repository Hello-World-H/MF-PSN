import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_
from . import model_utils
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
        num_init_feature = 2 * growth_rate

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
        return self.features(x)


def DenseNet121():
    return DenseNet_BC(growth_rate=32, block_config=(1, 2, 4, 3), num_classes=1000)


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
        out = self.conv3(out)
        return out  # 直接返回特征图 [B, 128, H/2, W/2]


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
        out = self.conv7(out)
        return out  # [B, 128, H/2, W/2] (输入尺寸的一半)


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

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False)
        )

    def forward(self, x):
        # 输入x是4D特征图 [B, C, H, W]
        out = self.deconv1(x)
        out = self.deconv2(out)
        out = self.deconv3(out)
        out = self.deconv4(out)
        out = self.deconv5(out)
        out = self.deconv6(out)
        normal = self.est_normal(out)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal


class LightAttentionFusion(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(LightAttentionFusion, self).__init__()
        self.channel_att = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, feats):
        """
        feats: 光源特征列表, 每个是形状为 (B, C, H, W) 的张量
        返回: 融合后的特征 (B, C, H, W)
        """
        # 确保所有特征图尺寸相同
        max_h = max([f.shape[2] for f in feats])
        max_w = max([f.shape[3] for f in feats])
        target_size = (max_h, max_w)

        aligned_feats = []
        for f in feats:
            # 仅当尺寸不匹配时才插值
            if f.shape[2:] != target_size:
                f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=True)
            aligned_feats.append(f)

        # 通道注意力
        channel_weights = []
        for feat in aligned_feats:
            gap = torch.mean(feat, dim=[2, 3])  # (B, C)
            channel_weight = self.channel_att(gap)  # (B, C)
            channel_weights.append(channel_weight.unsqueeze(-1).unsqueeze(-1))  # (B, C, 1, 1)

        # 空间注意力
        spatial_weights = [self.spatial_att(feat) for feat in aligned_feats]  # 每个 (B, 1, H, W)

        # 加权融合
        fused_feat = 0
        for i in range(len(aligned_feats)):
            weighted_feat = aligned_feats[i] * channel_weights[i] * spatial_weights[i]
            fused_feat += weighted_feat

        return fused_feat


class MF_PSN(nn.Module):
    def __init__(self, fuse_type='attn', batchNorm=False, c_in=3, other={}):
        super(MF_PSN, self).__init__()
        self.extractor1 = FeatExtractor1(batchNorm, c_in, other)
        self.extractor2 = FeatExtractor2(batchNorm, other)
        self.regressor = Regressor(batchNorm, other)
        self.c_in = c_in
        self.fuse_type = fuse_type
        self.other = other

        # 添加注意力融合模块
        if fuse_type == 'attn':
            self.attn_fusion1 = LightAttentionFusion(in_channels=128)
            self.attn_fusion2 = LightAttentionFusion(in_channels=128)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        img = x[0]  # 输入图像 [B, 3*N, H, W]
        num_lights = img.shape[1] // 3
        img_split = torch.split(img, 3, dim=1)  # 分割为单光源图像列表

        if len(x) > 1:  # 如果有光照信息
            light = x[1]
            light_split = torch.split(light, 3, dim=1)
        else:
            light_split = [None] * num_lights

        # ===== 第一阶段：提取基础特征 =====
        feats = []  # 存储每个光源的初级特征 [B, 128, H/2, W/2]
        for i in range(num_lights):
            # 如果有光照信息，与图像拼接
            if light_split[i] is not None:
                net_in = torch.cat([img_split[i], light_split[i]], dim=1)
            else:
                net_in = img_split[i]

            feat = self.extractor1(net_in)
            feats.append(feat)

        # ===== 第一阶段融合 =====
        if self.fuse_type == 'mean':
            feat_fused = torch.stack(feats, dim=0).mean(dim=0)
        elif self.fuse_type == 'max':
            feat_fused, _ = torch.stack(feats, dim=0).max(dim=0)
        elif self.fuse_type == 'attn':
            feat_fused = self.attn_fusion1(feats)

        # ===== 第二阶段：特征增强 =====
        featss = []  # 存储每个光源的增强特征 [B, 128, H/2, W/2]
        for i in range(num_lights):
            # 拼接基础特征和融合特征 (128+128=256通道)
            featt = torch.cat([feats[i], feat_fused], dim=1)
            enhanced_feat = self.extractor2(featt)
            featss.append(enhanced_feat)

        # ===== 第二阶段融合 =====
        if self.fuse_type == 'mean':
            feat_fusedd = torch.stack(featss, dim=0).mean(dim=0)
        elif self.fuse_type == 'max':
            feat_fusedd, _ = torch.stack(featss, dim=0).max(dim=0)
        elif self.fuse_type == 'attn':
            feat_fusedd = self.attn_fusion2(featss)

        # ===== 法线回归 =====
        normal = self.regressor(feat_fusedd)
        return normal