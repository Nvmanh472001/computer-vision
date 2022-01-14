import os
import argparse
from inference import Transformer
from utils import root_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='{}/content/checkpoints/generator_hayao.pth'.format(root_path))
    parser.add_argument('--src', type=str, default='{}/content/dataset/real'.format(root_path), help='source dir, contain real images')
    parser.add_argument('--dest', type=str, default='{}/content/images/gan'.format(root_path), help='destination dir to save generated images')

    return parser.parse_args()


def main(args):
    transformer = Transformer(args.checkpoint)

    if os.path.exists(args.src) and not os.path.isfile(args.src):
        transformer.transform_in_dir(args.src, args.dest)
    else:
        transformer.transform_file(args.src, args.dest)

if __name__ == '__main__':
    args = parse_args()
    main(args)
