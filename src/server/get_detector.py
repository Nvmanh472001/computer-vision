import os
import argparse
from model.detection import Detection
from utils import root_path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='{}/content/real/detection'.format(root_path), help='source dir, contain real images')
    parser.add_argument('--dest', type=str, default='{}/content/images/detection'.format(root_path), help='destination dir to save detection images')

    return parser.parse_args()


def main(args):
    detection = Detection()

    if os.path.exists(args.src) and not os.path.isfile(args.src):
        detection.onImageDir(args.src, args.dest)
    else:
        detection.onImage(args.src)

if __name__ == '__main__':
    args = parse_args()
    main(args)