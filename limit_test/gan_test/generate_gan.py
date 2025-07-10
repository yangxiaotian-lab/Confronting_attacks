# encoding:utf-8
import os
import torch
from torch import nn
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
import sys
sys.path.append('../../../')
from model import model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def imResize(image, size=112):
    resizeLayer = torch.nn.AdaptiveAvgPool2d(output_size=size)
    imageResize = resizeLayer(image)
    return imageResize


# 对图像数据类型转换为tensor类型
data_transforms = transforms.Compose([
    # transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


class GEN(nn.Module):

    def __init__(self):
        super(GEN, self).__init__()

        self.gen_model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=5),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_adv = self.gen_model(x)
        return x_adv


class GAN_attack(object):

    def __init__(self, img_orign, model, conf=0.8):
        self.img_orign = img_orign
        self.model = model
        self.conf = conf

    def attack(self, img_no):
        # attack_model = GEN()
        attack_model = torch.load('../../../model/adv_gen_model_59.pth', map_location='cuda:0')
        attack_model.to(device)
        attack_model.eval()
        img_attack = attack_model(self.img_orign.unsqueeze(0)).squeeze(0)

        originFeature = self.model(imResize(self.img_orign).unsqueeze(0))
        attackFeature = self.model(imResize(img_attack).unsqueeze(0))
        originFeatureNormlized = originFeature / torch.norm(originFeature)
        attackFeatureNormlized = attackFeature / torch.norm(attackFeature)
        score = 0.5 * (originFeatureNormlized.matmul(attackFeatureNormlized.permute(1, 0)) + 1)
        print("得分 = {:.2f},".format(score.item() * 100))

        if score < self.conf:
            print("img_no: " + str(img_no) + "攻击失败")
        else:
            print("img_no: " + str(img_no) + "攻击失败")
        return img_attack


if __name__ == "__main__":
    # 模型加载
    featuresNum = 512
    net = model.MobileFacenet(featuresNum)  # mobileFace
    ckpt = torch.load("../../../model/mobileFace-ArcFace.ckpt")
    print('----------------------------加载MobileFace模型----------------------------')

    net.load_state_dict(ckpt['net_state_dict'])
    net = net.cuda()
    net.eval()

    ori_dir = '../../../dataset'
    print('加载的数据集为: ', ori_dir)
    img_no = 0
    dirlist = os.listdir(ori_dir)
    for dir in dirlist:
        files = os.listdir(os.path.join(ori_dir, dir))
        for f in files:
            img_name = os.path.join(ori_dir, dir, f)
            img = Image.open(img_name).convert("RGB")
            img_origin = data_transforms(img).to(device)

            img_attack = GAN_attack(img_origin, net).attack(img_no)

            img_attack = 127.5 * (img_attack.detach().cpu().numpy().transpose(1, 2, 0) + 1)
            img_attack = Image.fromarray(img_attack.astype('uint8'))

            save_folder = '../../../output/limit_data/gan_data/' + dir
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_path = save_folder + '/' + 'GAN+' + f[:-4] + '.png' 
            img_attack.save(save_path)
            img_no += 1
