import torch
import argparse
import os
import cv2
import numpy as np
import torch.optim as optim
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from model.gan import Generator
from model.gan import Discriminator
from model.losses import GanLoss
from model.losses import LossSummary
from utils.common import load_checkpoint
from utils.common import save_checkpoint
from utils.common import set_lr
from utils.common import initialize_weights
from utils.image_processing import denormalize_input
from dataset import DataSet
from utils import DefaultArgs
from tqdm import tqdm

gaussian_mean = torch.tensor(0.0)
gaussian_std = torch.tensor(0.1)


def parse_args():
    
    default_args = DefaultArgs()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=default_args.dataset)
    parser.add_argument('--data-dir', type=str, default=default_args.data_dir)
    parser.add_argument('--epochs', type=int, default=default_args.epochs)
    parser.add_argument('--init-epochs', type=int, default=default_args.init_epochs)
    parser.add_argument('--batch-size', type=int, default=default_args.batch_size)
    parser.add_argument('--checkpoint-dir', type=str, default=default_args.checkpoint_dir)
    parser.add_argument('--save-image-dir', type=str, default=default_args.save_image_dir)
    parser.add_argument('--gan-loss', type=str, default=default_args.gan_loss, help='lsgan')
    parser.add_argument('--resume', type=str, default=default_args.resume)
    parser.add_argument('--use_sn', action='store_true')
    parser.add_argument('--save-interval', type=int, default=default_args.save_interval)
    parser.add_argument('--debug-samples', type=int, default=default_args.debug_samples)
    parser.add_argument('--lr-g', type=float, default=default_args.lr_d)
    parser.add_argument('--lr-d', type=float, default=default_args.lr_g)
    parser.add_argument('--init-lr', type=float, default=default_args.init_lr)
    parser.add_argument('--wadvg', type=float, default=default_args.wadvd, help='Adversarial loss weight for G')
    parser.add_argument('--wadvd', type=float, default=default_args.wadvd, help='Adversarial loss weight for D')
    parser.add_argument('--wcon', type=float, default=default_args.wcon, help='Content loss weight')
    parser.add_argument('--wgra', type=float, default=default_args.wgra, help='Gram loss weight')
    parser.add_argument('--wcol', type=float, default=default_args.wcol, help='Color loss weight')
    parser.add_argument('--d-layers', type=int, default=3, help='Discriminator conv layers')
    parser.add_argument('--d-noise', action='store_true')

    return parser.parse_args()


def collate_fn(batch):
    img, digital_img, digital_img_gray, digital_img_smt_gray = zip(*batch)
    return (
        torch.stack(img, 0),
        torch.stack(digital_img, 0),
        torch.stack(digital_img_gray, 0),
        torch.stack(digital_img_smt_gray, 0),
    )


def check_params(args):
    data_path = os.path.join(args.data_dir, args.dataset)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Dataset not found {data_path}')

    if not os.path.exists(args.save_image_dir):
        print(f'* {args.save_image_dir} does not exist, creating...')
        os.makedirs(args.save_image_dir)

    if not os.path.exists(args.checkpoint_dir):
        print(f'* {args.checkpoint_dir} does not exist, creating...')
        os.makedirs(args.checkpoint_dir)


def save_samples(generator, loader, args, max_imgs=2, subname='gen'):
    '''
    Generate and save images
    '''
    generator.eval()

    max_iter = (max_imgs // args.batch_size) + 1
    fake_imgs = []

    for i, (img, *_) in enumerate(loader):
        with torch.no_grad():
            fake_img = generator(img.cuda())
            fake_img = fake_img.detach().cpu().numpy()
            # Channel first -> channel last
            fake_img  = fake_img.transpose(0, 2, 3, 1)
            fake_imgs.append(denormalize_input(fake_img, dtype=np.int16))

        if i + 1 == max_iter:
            break

    fake_imgs = np.concatenate(fake_imgs, axis=0)

    for i, img in enumerate(fake_imgs):
        save_path = os.path.join(args.save_image_dir, f'{subname}_{i}.jpg')
        cv2.imwrite(save_path, img[..., ::-1])


def gaussian_noise():
    return torch.normal(gaussian_mean, gaussian_std)


def main(args):
    check_params(args)

    print("Init models...")

    G = Generator(args.dataset).cuda()
    D = Discriminator(args).cuda()

    loss_tracker = LossSummary()

    loss_fn = GanLoss(args)

    # Create DataLoader
    data_loader = DataLoader(
        DataSet(args),
        batch_size=args.batch_size,
        num_workers=cpu_count(),
        pin_memory=True,
        shuffle=True,
        collate_fn=collate_fn,
    )

    optimizer_g = optim.Adam(G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    start_e = 0
    if args.resume == 'GD':
        # Load G and D
        try:
            start_e = load_checkpoint(G, args.checkpoint_dir)
            print("G weight loaded")
            load_checkpoint(D, args.checkpoint_dir)
            print("D weight loaded")
        except Exception as e:
            print('Could not load checkpoint, train from scratch', e)
    elif args.resume == 'G':
        # Load G only
        try:
            start_e = load_checkpoint(G, args.checkpoint_dir, posfix='_init')
        except Exception as e:
            print('Could not load G init checkpoint, train from scratch', e)

    for e in range(start_e, args.epochs):
        print(f"Epoch {e}/{args.epochs}")
        bar = tqdm(data_loader)
        G.train()

        init_losses = []

        if e < args.init_epochs:
            # Train with content loss only
            set_lr(optimizer_g, args.init_lr)
            for img, *_ in bar:
                img = img.cuda()
                
                optimizer_g.zero_grad()

                fake_img = G(img)
                loss = loss_fn.content_loss_vgg(img, fake_img)
                loss.backward()
                optimizer_g.step()

                init_losses.append(loss.cpu().detach().numpy())
                avg_content_loss = sum(init_losses) / len(init_losses)
                bar.set_description(f'[Init Training G] content loss: {avg_content_loss:2f}')

            set_lr(optimizer_g, args.lr_g)
            save_checkpoint(G, optimizer_g, e, args, posfix='_init')
            save_samples(G, data_loader, args, subname='initg')
            continue

        loss_tracker.reset()
        for img, digital_img, digital_img_gray, digital_img_smt_gray in bar:
            # To cuda
            img = img.cuda()
            digital_img = digital_img.cuda()
            digital_img_gray = digital_img_gray.cuda()
            digital_img_smt_gray = digital_img_smt_gray.cuda()

            # ---------------- TRAIN D ---------------- #
            optimizer_d.zero_grad()
            fake_img = G(img).detach()

            # Add some Gaussian noise to images before feeding to D
            if args.d_noise:
                fake_img += gaussian_noise()
                digital_img += gaussian_noise()
                digital_img_gray += gaussian_noise()
                digital_img_smt_gray += gaussian_noise()

            fake_d = D(fake_img)
            real_digital_img_d = D(digital_img)
            real_digital_img_gray_d = D(digital_img_gray)
            real_digital_img_smt_gray_d = D(digital_img_smt_gray)

            loss_d = loss_fn.compute_loss_D(
                fake_d, real_digital_img_d, real_digital_img_gray_d, real_digital_img_smt_gray_d)

            loss_d.backward()
            optimizer_d.step()

            loss_tracker.update_loss_D(loss_d)

            # ---------------- TRAIN G ---------------- #
            optimizer_g.zero_grad()

            fake_img = G(img)
            fake_d = D(fake_img)

            adv_loss, con_loss, gra_loss, col_loss = loss_fn.compute_loss_G(
                fake_img, img, fake_d, digital_img_gray)

            loss_g = adv_loss + con_loss + gra_loss + col_loss

            loss_g.backward()
            optimizer_g.step()

            loss_tracker.update_loss_G(adv_loss, gra_loss, col_loss, con_loss)

            avg_adv, avg_gram, avg_color, avg_content = loss_tracker.avg_loss_G()
            avg_adv_d = loss_tracker.avg_loss_D()
            bar.set_description(f'loss G: adv {avg_adv:2f} con {avg_content:2f} gram {avg_gram:2f} color {avg_color:2f} / loss D: {avg_adv_d:2f}')

        if e % args.save_interval == 0:
            save_checkpoint(G, optimizer_g, e, args)
            save_checkpoint(D, optimizer_d, e, args)
            save_samples(G, data_loader, args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
