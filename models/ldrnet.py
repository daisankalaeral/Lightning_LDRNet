import torch
import torch.nn as nn
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
                 dropout_rate):
        
        super().__init__(widthi_multiplier,
                         depth_multiplier,
                         num_classes,
                         drop_connect_rate,
                         dropout_rate)
        del self.fc

        self.corner_detector = nn.Linear(1280, 8)
        self.line = nn.Linear(1280, (n_points - 4) * 2)
        self.classifier = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.stem(x)
        idx = 0
        for stage in self.blocks:
            for block in stage:
                drop_connect_rate = self.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / len(self.blocks)
                x = block(x, drop_connect_rate)
                idx +=1
        x = self.head(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.dropout is not None:
            x = self.dropout(x)
        
        corners = self.corner_detector(x)
        line_points = self.line(x)
        cls = self.classifier(x)

        return corners, line_points, cls