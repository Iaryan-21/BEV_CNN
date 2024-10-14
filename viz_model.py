from explore import viz_model_preds

viz_model_preds(
    version='mini',
    modelf='/home/amishr17/Desktop/BEV_CNN/src/logdir/model525000.pt',
    dataroot='/home/amishr17/Desktop/BEV_CNN/data/v1.0-mini',
    map_folder='/home/amishr17/Desktop/BEV_CNN/data/v1.0-mini',
    gpuid=-1,  # Use -1 for CPU, or the appropriate GPU id if using GPU
    viz_train=False,  # Set to True if you want to visualize training data instead of validation data
    H=900, W=1600,
    resize_lim=(0.193, 0.225),
    final_dim=(128, 352),
    bot_pct_lim=(0.0, 0.22),
    rot_lim=(-5.4, 5.4),
    rand_flip=True,
    xbound=[-50.0, 50.0, 0.5],
    ybound=[-50.0, 50.0, 0.5],
    zbound=[-10.0, 10.0, 20.0],
    dbound=[4.0, 45.0, 1.0],
    bsz=1,  # You might want to use a smaller batch size for visualization
    nworkers=0,  # Use 0 for debugging or if you're using CPU
)