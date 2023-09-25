import torch
import lightning as pl
from models.ldrnet import LDRNet
from loss import calculate_total_loss
import configs

class Lightning_LDRNet(pl.LightningModule):
    def __init__(self,
                  n_points,
                  num_classes = 224,
                  widthi_multiplier = 1.0,
                  depth_multiplier = 1.0, 
                  drop_connect_rate = 0.2,
                  dropout_rate = 0.2,
                  lr = 3e-4,
                  backbone_pretrained_path = None,
                  use_feature_fusion = False):
        
        super().__init__()
        self.lr = lr

        self.ldrnet = LDRNet(n_points,
                             widthi_multiplier,
                             depth_multiplier,
                             num_classes,
                             drop_connect_rate,
                             dropout_rate,
                             use_feature_fusion)

        if backbone_pretrained_path:
            print(f"Loading efficientnetlite weights: {backbone_pretrained_path}")
            state_dict = torch.load(backbone_pretrained_path)
            del state_dict["fc.weight"]
            del state_dict["fc.bias"]
            self.ldrnet.load_state_dict(state_dict, strict=False)

        self.tuning = False
    
    def forward(self, inputs):
        outputs = self.ldrnet(inputs)
        return outputs
    
    def _common_step(self, batch, which_loss):
        image, corner_coords_true, cls_true  = batch

        corner_coords_pred, border_coords_pred, cls_pred = self(image)

        loss = calculate_total_loss(corner_coords_true, corner_coords_pred, border_coords_pred, cls_true, cls_pred)
        
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
        loss = self._common_step(batch, "val_loss")

        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self._common_step(batch, "test_loss")

        return loss
    
    # def predict_step(self, batch, batch_idx):
    #     image, corner_coords_true = batch
    #     corner_coords_pred, border_coords_pred = self(image)
    #     haha = image.numpy() * 255
    #     corner_coords_pred = corner_coords_pred.numpy()
    
    def configure_optimizers(self):
        # optimizer = torch.optim.RMSprop(self.parameters(), eps=1e-7, lr = self.lr)
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.98)
        return [optimizer]

if __name__ == '__main__':
    x = torch.rand((1,3,224,224))
    model = Lightning_LDRNet(100, 6, backbone_pretrained_path="weights/pretrained_weights/efficientnet_lite0.pth", use_feature_fusion = True)
    a = input("Enter: ")
    print(model)
    out = model(x)