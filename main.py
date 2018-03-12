import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as f

from torch.autograd import Variable
import numpy as np
import argparse
from PIL import Image

from patchmatch import PatchMatch
from patchmatch2 import PatchMatch2
from vgg import Vgg19
from bds import bds_vote
import utils

parser = argparse.ArgumentParser(description='Neural Color Transfer between Images')
parser.add_argument('--source-image', type=str, default='images/source_b.jpg',
                                help='path to source-image')
parser.add_argument('--style-image', type=str, default='images/style_b.jpg',
                                help='path to style-image')
parser.add_argument('--scale', type=float, default=None,
                                help='factor to scale input images, eg. 0.5')
parser.add_argument("--cuda", dest='feature', action='store_true')
parser.set_defaults(cuda=False)

def main():
    args = parser.parse_args()

    source_image = utils.load_image(args.source_image, scale=args.scale)
    style_image = utils.load_image(args.style_image, scale=args.scale)

    #min_width = min(source_image.width, style_image.width)
    #min_height = min(source_image.height, style_image.height)

    #source_image = source_image.crop((0, 0, min_width, min_height))
    #style_image = style_image.crop((0, 0, min_width, min_height))

    #nnf = NNF(np.asarray(source_image), np.asarray(style_image))
    #nnf.solve()
    #c = nnf.reconstruct()
    #print(c)
    #image = Image.fromarray(c.astype('uint8'))
    #image.save('image.png')

    #print(source_image)


    # print(args)
    # vgg19 = models.vgg19(pretrained = True)
    # print(vgg19.features)
    # relu_1 = vgg19.features[1]

    # print(relu_1)
    # res = relu_1(Variable(source_image, requires_grad=False))
    # res_image = transforms.ToPILImage()(res.data)
    # res_image.save('image.png')

    # veggie = Vgg19()
    # source_image = source_image.unsqueeze(0)
    # veggie_res = veggie(Variable(source_image, requires_grad=False))


    # denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
    # img = veggie_res.clone().cpu().squeeze()
    # img = denorm(img.data)
    # res_image = transforms.ToPILImage()(img)
    # res_image.save('image2.png')

    color_transfer = ColorTransfer(source_image, style_image, args.cuda)
    color_transfer.run()

#def denormalize(img):
#    denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
#    return denorm(img)

class ColorTransfer(object):
    def __init__(self, source_image, style_image, cuda):
        self.source_image = source_image
        self.style_image = style_image
        self.cuda = cuda
        self.vgg19 = Vgg19()
        if self.cuda:
            self.vgg19.cuda()

    def run(self):
        S = utils.image_to_tensor(self.source_image)
        R = utils.image_to_tensor(self.style_image)
        print(S)
        self._color_transfer(S, R)

    def _resized_S(self, size):
        return utils.image_to_tensor(self.source_image, [transforms.Resize(size)])

    def _resized_R(self, size):
        return utils.image_to_tensor(self.style_image, [transforms.Resize(size)])

    def _color_transfer(self, S, R, level=5):
        """
        Color transfer function, calls itself recursively

        S -- source image at the given level (normalized tensor)
        R -- reference image (normalized tensor)
        level -- between 1 and 5 (default 5)
        """

        if level == 0:
            return

        F_S = self.vgg19(Variable(S.unsqueeze(0), requires_grad=False))[level - 1].data.squeeze()
        F_R = self.vgg19(Variable(R.unsqueeze(0), requires_grad=False))[level - 1].data.squeeze()

        print(F_S.size())
        print(F_R.size())

        snn = PatchMatch2(f.normalize(F_S, p=2, dim=0).numpy(),
                         f.normalize(F_R, p=2, dim=0).numpy())

        snn.solve()

        rnn = PatchMatch2(f.normalize(F_R, p=2, dim=0).numpy(),
                         f.normalize(F_S, p=2, dim=0).numpy())

        rnn.solve()

        # Resize R to feature map dimensions
        R_L = self._resized_R(F_R.size()[1:3])

        print(R_L)

        G = bds_vote(snn.nnf.transpose(2,1,0), rnn.nnf.transpose(2,1,0), snn.nnd.transpose(1,0), rnn.nnd.transpose(1,0), R_L)

        print(G)

        utils.save_image(f'g{level}.png', G*255.99)

        #F_G = bds_vote(snn.nnf, rnn.nnf, snn.nnfd, rnn.nnfd, F_R)

        #snn_img = snn.reconstruct()
        #snn_img = Image.fromarray(snn_img.astype('uint8'))


        # image_r_to_s.save('image_r_to_s.png')


        #F_S_img = self._feature_map_to_nnf(F_S, level)
        #F_S5_img = self._feature_map_to_nnf(F_S5, level)
        #F_R_img = self._feature_map_to_nnf(F_R, level)

        #F_S_img.save('blahblah.png')
        #F_S_img.save('blahblah.png')
        #F_R_img.save('blablahbla.png')

        # L_S_to_R_nnf = NNF(np.asarray(F_S_img), np.asarray(F_R_img))
        # L_S_to_R_nnf.solve()
        # L_S_to_R_img = L_S_to_R_nnf.reconstruct()
        # image_s_to_r = Image.fromarray(L_S_to_R_img.astype('uint8'))
        # image_s_to_r.save('image_s_to_r.png')

        # L_R_to_S_nnf = NNF(np.asarray(F_R_img), np.asarray(F_S_img))
        # L_R_to_S_nnf.solve()
        # L_R_to_S_img = L_R_to_S_nnf.reconstruct()
        # image_r_to_s = Image.fromarray(L_R_to_S_img.astype('uint8'))
        # image_r_to_s.save('image_r_to_s.png')


        #G = bds_vote(pm)
        #S = downscale(source_original, G) # BILINEAR

if __name__ == "__main__":
    main()


