n_points = 100
if n_points > 4:
    size_per_border = int((n_points-4)/4)
num_classes = 6

reg_ratio = 50 # 0.001 when not scaled
beta = 0.01
gamma = 0.01

auto_initial_lr = False
lr = 0.0003
batch_size = 128
num_workers = 4

valid_interval = 1

warmup_step = 7500