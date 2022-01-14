import os
import argparse
from inference import Transformer
from model.detector import Detector
from utils import root_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default=root_path+'/content/datatest/real', help='source dir, contain real images')
    parser.add_argument('--dest', type=str, default=root_path+'/content/images/detector', help='destination dir to save generated images')

    return parser.parse_args()


def main(args):
    detector = Detector()

    if os.path.exists(args.src) and not os.path.isfile(args.src):
        detector.onImageDir(args.src, args.dest)
    else:
        detector.onImage(args.src, args.dest)

if __name__ == '__main__':
    args = parse_args()
    main(args)