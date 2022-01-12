import os
from cv2 import rotate
from .common import *
from .image_processing import *

rootPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
class DefaultArgs:
    def __init__(self):
        self.dataset = 'Hayao'
        self.data_dir = '{}/content/dataset'.format(rootPath)
        self.epochs = 100
        self.init_epochs = 5
        self.batch_size = 6
        self.checkpoint_dir ='{}/content/checkpoints'.format(rootPath)
        self.save_image_dir ='{}/content/images'.format(rootPath)
        self.gan_loss = "lsgan"
        self.resume="False"
        self.display_image = True
        self.save_interval = 1
        self.debug_samples = 0
        self.lr_g = 2e-4
        self.lr_d = 4e-4
        self.wadvg = 10.0
        self.wadvd = 10.0
        self.wcon = 1.5
        self.wgra = 3.0
        self.wcol = 30.0
        self.use_sn = False
