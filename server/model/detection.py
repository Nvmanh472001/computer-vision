from re import M
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo

import cv2
import os
from tqdm import tqdm
import numpy as np
from torch.nn.functional import instance_norm

VALID_FORMATS = {
    'jpeg', 'jpg', 'jpe',
    'png', 'bmp',
}

class Detection:
    def __init__(self):
        self.cfg = get_cfg()
        
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
        
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg.MODEL.DEVICE = "cpu"
        
        self.predictor = DefaultPredictor(self.cfg)
        
    def onImage(self, imagePath):
        image = cv2.imread(imagePath)
        predictions = self.predictor(image)
        
        viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
        instance_mode = ColorMode.IMAGE)
        
        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        
        img = output.get_image()[:,:,::-1]
        return img
    
    def onImageDir(self, img_dir, dest_dir):
        os.makedirs(dest_dir, exist_ok=True)
        
        files = os.listdir(img_dir)
        files = [f for f in files if self.is_valid_file(f)]
        print(f'Found {len(files)} images in {img_dir}')
        
        for fname in tqdm(files):
            img_path = os.path.join(img_dir, fname)
            detection = self.onImage(img_path)
            ext = fname.split('.')[-1]
            fname = fname.replace(f'.{ext}', '')
            cv2.imwrite(os.path.join(dest_dir, f'{fname}_detection.jpg'), detection)
    
    @staticmethod
    def is_valid_file(fname):
        ext = fname.split('.')[-1]
        return ext in VALID_FORMATS
    
    
    