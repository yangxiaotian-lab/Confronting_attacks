# encoding:utf-8
import os

import torch
from torch import nn
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
import sys
sys.path.append('../../../../')
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


class White_box_attack:
    def __init__(self, img_orign, model, sourceFeatureNormlized=None, conf=0.8):
        self.img_orign = img_orign
        self.model = model
        self.sourceFeatureNormlized = sourceFeatureNormlized
        self.conf = conf

    def PGD_attack(self, img_no, iters=100, eps=0.1 / 255 * 2, eps_max=6 / 255 * 2):
        img_orign = self.img_orign.clone()
        originFeature = self.model(imResize(img_orign).unsqueeze(0)).detach()
        originFeatureNormlized = originFeature / torch.norm(originFeature)

        flag = False
        img_attack = img_orign + 0.01 * torch.randn(img_orign.size()).to(img_orign.device)
        img_attack = torch.clamp(img_attack, -1, 1)
        for i in range(iters):
            img_attack.requires_grad = True
            attackFeature = self.model(imResize(img_attack).unsqueeze(0))
            attackFeatureNormlized = attackFeature / torch.norm(attackFeature)
            score = 0.5 * (originFeatureNormlized.matmul(attackFeatureNormlized.permute(1, 0)) + 1)

            # print('{:04d}: '.format(i) + '得分={:.2f}'.format(score.item() * 100))

            if score < self.conf:
                flag = True
                # break
            score.backward()
            grad = img_attack.grad.data
            img_attack = img_attack - eps * torch.sign(grad)  # FGSM
            img_noise = img_attack - img_orign
            img_noise = np.clip(img_noise.detach().cpu().numpy(), -eps_max, eps_max)  # Gradient Project
            img_attack = img_orign.detach().cpu().numpy() + img_noise
            img_attack = np.clip(img_attack, -1, 1)
            img_attack = torch.Tensor(img_attack).to(img_orign.device)

        if flag:
            print("img_no: " + str(img_no) + " 攻击成功")
        else:
            print("img_no: " + str(img_no) + " 攻击失败")
        return img_attack


if __name__ == "__main__":
    # 模型加载
    featuresNum = 512
    net = model.MobileFacenet(featuresNum)  # mobileFace
    ckpt = torch.load("../../../../model/mobileFace-ArcFace.ckpt")
    print('----------------------------加载MobileFace模型----------------------------')

    net.load_state_dict(ckpt['net_state_dict'])
    net = net.cuda()
    net.eval()

    pgd_para = np.linspace(1, 10, 10).astype('double')

    ori_dir = '../../../../dataset'
    print('加载的数据集为: ', ori_dir)

    dirlist = os.listdir(ori_dir)
    for value in pgd_para:
        img_no = 0
        for dir in dirlist:
            files = os.listdir(os.path.join(ori_dir, dir))
            for f in files:
                img_name = os.path.join(ori_dir, dir, f)
                img = Image.open(img_name).convert("RGB")
                img_origin = data_transforms(img).to(device)

                img_attack = White_box_attack(img_origin, net).PGD_attack(img_no, eps=0.01 / 255 * value)

                img_attack = 127.5 * (img_attack.detach().cpu().numpy().transpose(1, 2, 0) + 1)
                img_attack = Image.fromarray(img_attack.astype('uint8'))

                save_folder = '../../../../output/limit_data/whitebox_data/pgd_∞/PGD_∞_' + str(value) + '/' + dir
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                save_path = save_folder + '/' + 'PGD_∞_' + str(value) + '+' + f[:-4] + '.png'
                img_attack.save(save_path)
                img_no += 1

        print("参数为" + str(value) + "的已经生成完成")
