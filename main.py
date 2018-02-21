import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import argparse

from patchmatch import NNF
import utils
from vgg import Vgg19

from PIL import Image
parser = argparse.ArgumentParser(description='Neural Color Transfer between Images')
parser.add_argument('--source-image', type=str, default='images/source_b.jpg',
                                  help='path to source-image')
parser.add_argument('--style-image', type=str, default='images/style_b.jpg',
                                  help='path to style-image')

def main():
    args = parser.parse_args()

    source_image = utils.load_image(args.source_image)
    style_image = utils.load_image(args.style_image)

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

    color_transfer = ColorTransfer(source_image, style_image)
    color_transfer.run()

#def denormalize(img):
#    denorm = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
#    return denorm(img)

class ColorTransfer(object):
    def __init__(self, source_image, style_image):
        self.source_image = source_image
        self.style_image = style_image
        self.vgg19 = Vgg19()

    def run(self):
        S = utils.image_to_tensor(self.source_image)
        R = utils.image_to_tensor(self.style_image)
        self._color_transfer(S, R)

    def _feature_map_to_nnf(self, feature_map, layer = 5):
        denormalize = transforms.Normalize((-2.12, -2.04, -1.80), (4.37, 4.46, 4.44))
        scale_factor = 1 / (layer - 1)

        to_img = transforms.Compose([
            denormalize,
            transforms.ToPILImage()
        ])
        nnf_img = to_img(feature_map)
        #nnf_img = nnf_img.resize((int(nnf_img.width * scale_factor), int(nnf_img.height * scale_factor)))
        return nnf_img

    def _color_transfer(self, S, R, level = 5):
        F_S = self.vgg19(Variable(S.unsqueeze(0), requires_grad=False))[0].data.squeeze()
        F_S5 = self.vgg19(Variable(S.unsqueeze(0), requires_grad=False))[4].data.squeeze()
        F_R = self.vgg19(Variable(R.unsqueeze(0), requires_grad=False))[3].data.squeeze()

        print(F_S.size())
        print(F_R.size())

        F_S_img = self._feature_map_to_nnf(F_S, level)
        F_S5_img = self._feature_map_to_nnf(F_S5, level)
        F_R_img = self._feature_map_to_nnf(F_R, level)

        F_S_img.save('blahblah.png')
        F_S_img.save('blahblah.png')
        F_R_img.save('blablahbla.png')

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


