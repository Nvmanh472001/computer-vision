import cog
from pathlib import Path
from inference import Transformer
from utils import read_image
import cv2
import tempfile
from model.detection import Detection
from utils.image_processing import resize_image, normalize_input, denormalize_input
import numpy as np

class Predictor(cog.Predictor):
    def setup(self):
        pass

    @cog.input("image", type=Path, help="input image")
    @cog.input("model", type=str, default='gan',options=['gan', 'detection'], help="Model to handler image using Computer Vision")
    def predict(self, image, model="gan"):
        if model == "gan":
            transformer = Transformer()
            img = read_image(str(image))
            digital_img = transformer.transform(resize_image(img))[0]
            digital_img = denormalize_input(digital_img, dtype=np.int16)
            out_path = Path(tempfile.mkdtemp()) / "out_{}.png".format(model)
            cv2.imwrite(str(out_path), digital_img[..., ::-1])
        else:
            detection = Detection()
            img = detection.onImage(str(image))
            out_path = Path(tempfile.mkdtemp()) / "out_{}.png".format(model)
            cv2.imwrite(str(out_path), img)

        
        
        
        return out_path
