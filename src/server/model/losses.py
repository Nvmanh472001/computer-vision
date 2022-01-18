import torch
import torch.nn.functional as F
import torch.nn as nn
from model.vgg import Vgg19
from utils.image_processing import gram, rgb_to_yuv


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.huber = nn.SmoothL1Loss()

    def forward(self, image, image_g):
        image = rgb_to_yuv(image)
        image_g = rgb_to_yuv(image_g)

        # After convert to yuv, both images have channel last

        return (self.l1(image[:, :, :, 0], image_g[:, :, :, 0]) +
                self.huber(image[:, :, :, 1], image_g[:, :, :, 1]) +
                self.huber(image[:, :, :, 2], image_g[:, :, :, 2]))

class GanLoss:
    def __init__(self, args):
        self.content_loss = nn.L1Loss().cuda()
        self.gram_loss = nn.L1Loss().cuda()
        self.color_loss = ColorLoss().cuda()
        self.wadvg = args.wadvg
        self.wadvd = args.wadvd
        self.wcon = args.wcon
        self.wgra = args.wgra
        self.wcol = args.wcol
        self.vgg19 = Vgg19().cuda().eval()
        self.adv_type = args.gan_loss

    def compute_loss_G(self, fake_img, img, fake_logit, digital_gray):
        '''
        Compute loss for Generator

        @Arugments:
            - fake_img: generated image
            - img: image
            - fake_logit: output of Discriminator given fake image
            - digital_gray: grayscale of digital image

        @Returns:
            loss
        '''
        fake_feat = self.vgg19(fake_img)
        digital_feat = self.vgg19(digital_gray)
        img_feat = self.vgg19(img).detach()

        return [
            self.wadvg * self.adv_loss_g(fake_logit),
            self.wcon * self.content_loss(img_feat, fake_feat),
            self.wgra * self.gram_loss(gram(digital_feat), gram(fake_feat)),
            self.wcol * self.color_loss(img, fake_img),
        ]

    def compute_loss_D(self, fake_img_d, real_digital_d, real_digital_gray_d, real_digital_smooth_gray_d):
        return self.wadvd * (
            self.adv_loss_d_real(real_digital_d) +
            self.adv_loss_d_fake(fake_img_d) +
            self.adv_loss_d_fake(real_digital_gray_d) +
            0.2 * self.adv_loss_d_fake(real_digital_smooth_gray_d)
        )


    def content_loss_vgg(self, image, recontruction):
        feat = self.vgg19(image)
        re_feat = self.vgg19(recontruction)

        return self.content_loss(feat, re_feat)

    def adv_loss_d_real(self, pred):
        return torch.mean(torch.square(pred - 1.0))

    def adv_loss_d_fake(self, pred):
        return torch.mean(torch.square(pred))


    def adv_loss_g(self, pred):
        return torch.mean(torch.square(pred - 1.0))
        


class LossSummary:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss_g_adv = []
        self.loss_content = []
        self.loss_gram = []
        self.loss_color = []
        self.loss_d_adv = []

    def update_loss_G(self, adv, gram, color, content):
        self.loss_g_adv.append(adv.cpu().detach().numpy())
        self.loss_gram.append(gram.cpu().detach().numpy())
        self.loss_color.append(color.cpu().detach().numpy())
        self.loss_content.append(content.cpu().detach().numpy())

    def update_loss_D(self, loss):
        self.loss_d_adv.append(loss.cpu().detach().numpy())

    def avg_loss_G(self):
        return (
            self._avg(self.loss_g_adv),
            self._avg(self.loss_gram),
            self._avg(self.loss_color),
            self._avg(self.loss_content),
        )

    def avg_loss_D(self):
        return self._avg(self.loss_d_adv)


    @staticmethod
    def _avg(losses):
        return sum(losses) / len(losses)