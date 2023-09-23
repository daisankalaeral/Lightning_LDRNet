import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
import lightning as pl
import cv2 as cv
from loss import calculate_total_loss
import configs
from models.efficientnet_lite import EfficientNetLite

class CustomSigmoid(nn.Module):
    def forward(self, x):
        return 3.0 * torch.sigmoid(x) - 1.0

class LDRNet(pl.LightningModule):
    def __init__(self, n_points = 100, lr = 1e-3, **kwargs):
        super().__init__()
        self.n_points = n_points
        self.lr = lr
        self.cnt = 0

        # mobilenet
#         self.backbone_model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.DEFAULT, **kwargs)
#         self.backbone_model.classifier[1] = nn.Linear(self.backbone_model.last_channel, 8)
#         torch.nn.init.xavier_uniform_(self.backbone_model.classifier[1].weight)
        
#         self.backbone_model.border = nn.Sequential(
#             nn.Dropout(0),
#             nn.Linear(self.backbone_model.last_channel, (n_points - 4) * 2)
#         )
#         torch.nn.init.xavier_uniform_(self.backbone_model.border[1].weight)
        
        # efficientnet
        efficientnet_lite = EfficientNetLite(1.0, 1.0, 224, 0.2, 0.2)
        del efficientnet_lite.fc
        
        state_dict = torch.load("efficientnet_lite0.pth")
        del state_dict["fc.weight"]
        del state_dict["fc.bias"]
        
        efficientnet_lite.load_state_dict(state_dict)
        self.backbone_model = efficientnet_lite

        self.backbone_model.classifier = nn.Linear(1280, 8)
        self.backbone_model.border = nn.Linear(1280, (n_points - 4) * 2)
        
        self.tuning = False

    def custom_forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.backbone_model.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        corners = self.backbone_model.classifier(x)
        points = self.backbone_model.border(x)
        
        return corners, points
    
    def efficient_net_forward(self, x):
        x = self.backbone_model.stem(x)
        idx = 0
        for stage in self.backbone_model.blocks:
            for block in stage:
                drop_connect_rate = self.backbone_model.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / len(self.backbone_model.blocks)
                x = block(x, drop_connect_rate)
                idx +=1
        x = self.backbone_model.head(x)
        x = self.backbone_model.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.backbone_model.dropout is not None:
            x = self.backbone_model.dropout(x)
        
        corners = self.backbone_model.classifier(x)
        points = self.backbone_model.border(x)

        return corners, points
    
    def forward(self, inputs):
        x = self.efficient_net_forward(inputs)
        return x
    
    def _common_step(self, batch, which_loss):
        image, corner_coords_true = batch

        corner_coords_pred, border_coords_pred = self(image)

        loss = calculate_total_loss(corner_coords_true, corner_coords_pred, border_coords_pred)
        
        self.log_dict(
            {
                "lr": self.optimizers().param_groups[0]['lr'],
            },  
            on_step=True,
            on_epoch=False,
            prog_bar=True
        )
        
        self.log_dict(
            {
                which_loss: loss
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        if not self.tuning:
            if self.trainer.global_step < configs.warmup_step:
                lr_scale = min(1., float(self.trainer.global_step + 1) / configs.warmup_step)
                optimizer = self.optimizers()
                for pg in optimizer.param_groups:
                    pg['lr'] = lr_scale * self.lr
            # elif self.trainer.global_step % 10 == 0:
            #     self.scheduler.step()
            if batch_idx == 0 and self.trainer.global_step >= configs.warmup_step:
                self.scheduler.step()
            
        loss = self._common_step(batch, "train_loss")
        
        return loss
    
    def validation_step(self, batch, batch_idx):
#         image, corner_coords_true = batch
#         corner_coords_pred, border_coords_pred = self(image)
#         loss = calculate_total_loss(corner_coords_true, corner_coords_pred, border_coords_pred)
#         if batch_idx == 0 or batch_idx == 1 or batch_idx == 2:
#             image = image.cpu().numpy() * 255
#             image = image.transpose((0,2,3,1))[0]
#             huhu = image.copy()
#             huhu = cv.cvtColor(huhu, cv.COLOR_RGB2BGR)

#             temp = corner_coords_pred.cpu().numpy()
#             x = temp[0, 0::2] * 224
#             y = temp[0, 1::2] * 224
            
#             colors = [(0,0,255), (0,255,0), (255,0,0), (255,0,255)]
#             for i, (a,b) in enumerate(zip(x,y)):
#                 huhu = cv.circle(huhu, (int(a),int(b)), 3, colors[i], 2)
#             cv.imwrite(f"/notebooks/log_images/{self.cnt}.jpg", huhu)
#             self.cnt += 1

        loss = self._common_step(batch, "val_loss")

        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, "test_loss")

        return loss
    
    def predict_step(self, batch, batch_idx):
        image, corner_coords_true = batch
        corner_coords_pred, border_coords_pred = self(image)
        haha = image.numpy() * 255
        corner_coords_pred = corner_coords_pred.numpy()
    
    def configure_optimizers(self):
        # optimizer = torch.optim.RMSprop(self.parameters(), eps=1e-7, lr = self.lr)
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)
        return [optimizer]

if __name__ == '__main__':


    model = LDRNet(100, 1.0, dropout = 0.2)
    print(model.backbone_model)