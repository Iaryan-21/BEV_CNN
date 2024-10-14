import os
from data import compile_data

# Define your configurations
data_aug_conf = {
    'resize_lim': (0.193, 0.225),
    'final_dim': (128, 352),
    'rot_lim': (-5.4, 5.4),
    'H': 900, 'W': 1600,
    'rand_flip': True,
    'bot_pct_lim': (0.0, 0.22),
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 5,
}

grid_conf = {
    'xbound': [-50.0, 50.0, 0.5],
    'ybound': [-50.0, 50.0, 0.5],
    'zbound': [-10.0, 10.0, 20.0],
    'dbound': [4.0, 45.0, 1.0],
}

# Specify the dataroot
dataroot = '/home/amishr17/Desktop/BEV_CNN/data/v1.0-mini'

# Create data loaders
trainloader, valloader = compile_data(
    version='mini',
    dataroot=dataroot,
    data_aug_conf=data_aug_conf,
    grid_conf=grid_conf,
    bsz=4,
    nworkers=10,
    parser_name='vizdata'
)

# Test the data loader
for batch in trainloader:
    print(f"Batch size: {len(batch)}")
    print(f"Number of images: {batch[0].shape}")
    break

print("Data loading successful!")