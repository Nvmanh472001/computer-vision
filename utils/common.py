import torch
import gc
import os
import torch.nn as nn
import urllib.request
import cv2
from tqdm import tqdm

HTTP_PREFIXES = [
    'http',
    'data:image/jpeg',
]

def read_image(path):
    """
    Read image from given path
    """

    if any(path.startswith(p) for p in HTTP_PREFIXES):
        urllib.request.urlretrieve(path, "temp.jpg")
        path = "temp.jpg"

    return cv2.imread(path)[: ,: ,::-1]


def save_checkpoint(model, optimizer, epoch, args, posfix=''):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    path = os.path.join(args.checkpoint_dir, f'{model.name}{posfix}.pth')
    torch.save(checkpoint, path)


def load_checkpoint(model, checkpoint_dir, posfix=''):
    path = os.path.join(checkpoint_dir, f'{model.name}{posfix}.pth')
    return load_weight(model, path)


def load_weight(model, weight):
    checkpoint = torch.load(weight,  map_location='cuda:0') if torch.cuda.is_available() else \
        torch.load(weight,  map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    epoch = checkpoint['epoch']
    del checkpoint
    torch.cuda.empty_cache()
    gc.collect()

    return epoch


def initialize_weights(net):
    for m in net.modules():
        try:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        except Exception as e:
            # print(f'SKip layer {m}, {e}')
            pass


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



