class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/data2/saizhou777/OSTrack-main'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/data2/saizhou777/OSTrack-main/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/data2/saizhou777/OSTrack-main/pretrained_networks'
        self.lasot_dir = '/data2/saizhou777/data/lasot'
        self.got10k_dir = '/data2/saizhou777/data/got10k/train'
        self.got10k_val_dir = '/data2/saizhou777/data/got10k/val'
        self.lasot_lmdb_dir = '/data2/saizhou777/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/data2/saizhou777/data/got10k_lmdb'
        self.trackingnet_dir = '/data2/saizhou777/data/trackingnet'
        self.trackingnet_lmdb_dir = '/data2/saizhou777/data/trackingnet_lmdb'
        self.coco_dir = '/data2/saizhou777/data/coco'
        self.coco_lmdb_dir = '/data2/saizhou777/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/data2/saizhou777/data/vid'
        self.imagenet_lmdb_dir = '/data2/saizhou777/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
