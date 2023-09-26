import torch
import random
import numpy as np
torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)
random.seed(123456)
np.random.seed(123456)
import lightning as pl
from lightning_model import Lightning_LDRNet
from data.dataset import DocDataModule
import configs
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from callback import CustomPrintingCallback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
# from pytorch_lightning.callbacks.quantizations import QuantizationAwareTraining

torch.set_float32_matmul_precision("medium") # to make lightning happy

if __name__ == "__main__":
    logger = TensorBoardLogger(configs.output_dir, name = "logs")
    checkpoint_callback = ModelCheckpoint(dirpath=configs.output_dir, save_top_k=1, monitor="val_loss", save_last = True)
    early_stop_callback = EarlyStopping(monitor="val_loss", patience=50)
  
    trainer = pl.Trainer(
        logger = logger,
        accelerator="gpu", 
        devices=1, 
        min_epochs=1, 
        max_epochs=1000, 
        precision='16-mixed',
        default_root_dir="test",
        check_val_every_n_epoch=configs.valid_interval,
        enable_checkpointing = True,
        callbacks = [checkpoint_callback, CustomPrintingCallback(), early_stop_callback],
        gradient_clip_val = 5.0,
        # plugins='deepspeed'
        # detect_anomaly=True
    )

    model = Lightning_LDRNet(
                      configs.n_points,
                      num_classes = configs.num_classes,
                      widthi_multiplier = 1.1,
                      depth_multiplier = 1.2, 
                      drop_connect_rate = 0.2,
                      dropout_rate = 0.2,
                      lr = configs.lr,
                      backbone_pretrained_path = "weights/pretrained_weights/efficientnet_lite2.pth",
                      use_feature_fusion = False)
    
    dm = DocDataModule(
        train_json_path="/notebooks/LDRNet_dataset/train_with_class.json",
        valid_json_path="/notebooks/LDRNet_dataset/valid_with_class.json",
        data_dir="/notebooks/LDRNet_dataset",
        batch_size=configs.batch_size,
        num_workers=configs.num_workers
    )
    
    if configs.auto_initial_lr:
        model.tuning = True
        tuner = pl.pytorch.tuner.tuning.Tuner(trainer)
        lr_finder = tuner.lr_find(model, dm, num_training = 500)
    
    model.tuning = False
    trainer.fit(model, dm) # ckpt_path="some/path/to/my_checkpoint.ckpt"
