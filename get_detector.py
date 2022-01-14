import os
import argparse
from model.detector import Detector
from utils import root_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='{}/content/dataset/real'.format(root_path), help='source dir, contain real images')
    parser.add_argument('--dest', type=str, default='{}/content/images/detector'.format(root_path), help='destination dir to save detector images')

    return parser.parse_args()


def main(args):
    detector = Detector()

    if os.path.exists(args.src) and not os.path.isfile(args.src):
        detector.onImageDir(args.src, args.dest)
    else:
        detector.onImage(args.src)

if __name__ == '__main__':
    args = parse_args()
    main(args)