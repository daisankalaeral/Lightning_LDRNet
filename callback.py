from lightning.pytorch.callbacks import Callback

class CustomPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, *args, **kwargs):
        print("\n")
        
    def on_test_epoch_end(self, *args, **kwargs):
        print("\n")