import torch
import torch.nn as nn
import torch.nn.functional as F
from .efficientnet_lite import EfficientNetLite
import sys
sys.path.append("..")

class LDRNet(EfficientNetLite):
    def __init__(self,
                 n_points,
                 widthi_multiplier,
                 depth_multiplier,
                 num_classes,
                 drop_connect_rate,
                 dropout_rate,
                 use_feature_fusion):
        
        super().__init__(widthi_multiplier,
                         depth_multiplier,
                         num_classes,
                         drop_connect_rate,
                         dropout_rate)
        del self.fc

        self.corner_detector = nn.Linear(1280, 8)
        self.line = nn.Linear(1280, (n_points - 4) * 2)
        self.classifier = nn.Linear(1280, num_classes)
        self.use_feature_fusion = use_feature_fusion

        if use_feature_fusion:
            self.fusion = FeatureFusionModule([16, 24, 40, 80, 112, 192, 320], 320)

    def efficient_net_forward(self, x):
        x = self.backbone_model.stem(x)
        feature_maps = []
        idx = 0
        for stage in self.backbone_model.blocks:
            for block in stage:
                drop_connect_rate = self.backbone_model.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / len(self.backbone_model.blocks)
                x = block(x, drop_connect_rate)
                idx +=1
            feature_maps.append(x)
        
        x = self.backbone_model.head(x)
        x = self.backbone_model.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.backbone_model.dropout is not None:
            x = self.backbone_model.dropout(x)
        
        corners = self.backbone_model.classifier(x)
        points = self.backbone_model.border(x)

        return corners, points

    def forward(self, x):
        x = self.stem(x)
        feature_maps = []
        idx = 0
        for stage in self.blocks:
            for block in stage:
                drop_connect_rate = self.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / len(self.blocks)
                x = block(x, drop_connect_rate)
                idx +=1
            if self.use_feature_fusion:
                feature_maps.append(x)

        if self.use_feature_fusion:
            x = self.fusion(feature_maps)
        
        x = self.head(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.dropout is not None:
            x = self.dropout(x)
        
        corners = self.corner_detector(x)
        line_points = self.line(x)
        cls = self.classifier(x)

        return corners, line_points, cls

class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        
        self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list])

    def forward(self, input_list):
        adjusted_features = [conv(x) for conv, x in zip(self.convs, input_list)]
        
        target_size = max([x.shape[2:] for x in adjusted_features])
        interpolated_features = [F.interpolate(x, size=target_size, mode='bilinear', align_corners=False) for x in adjusted_features]
        
        fused_features = sum(interpolated_features)
        
        return fused_features