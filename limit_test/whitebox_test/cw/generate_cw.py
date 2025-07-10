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


class White_box_attack(object):

    def __init__(self, img_orign, model, sourceFeatureNormlized=None, targetFeatureNormlized=None, featureSet=None,
                 conf=0.7):
        self.img_orign = img_orign
        self.model = model
        self.sourceFeatureNormlized = sourceFeatureNormlized
        self.targetFeatureNormlized = targetFeatureNormlized
        self.featureSet = featureSet
        self.conf = conf

    def CW_attack(self, img_no, iters=50, norm_p=2, lamb=1e-3, lr=6e-4):
        img_orign = self.img_orign.clone()
        originFeature = self.model(imResize(img_orign).unsqueeze(0)).detach()
        originFeatureNormlized = originFeature / torch.norm(originFeature)

        # img_orign = torch.clamp(img_orign, 0 + 1e-16, 1 - 1e-16)
        # wn = 0.5 * torch.log1p(2 * (2 * img_orign - 1) / (1 - (2 * img_orign - 1)))
        img_orign = torch.clamp(img_orign, -1 + 1e-16, 1 - 1e-16)
        wn = 0.5 * torch.log1p(2 * img_orign / (1 - img_orign))
        wn.requires_grad = True
        optimizer = torch.optim.Adam([wn], lr=lr)

        flag = False
        for i in range(iters):
            rn = torch.tanh(wn) - img_orign
            img_attack = torch.tanh(wn)
            attackFeature = self.model(imResize(img_attack).unsqueeze(0))
            attackFeatureNormlized = attackFeature / torch.norm(attackFeature)
            score = 0.5 * (originFeatureNormlized.matmul(attackFeatureNormlized.permute(1, 0)) + 1)
            loss = score + lamb * torch.norm(rn, p=norm_p)
            # print('{:04d}：'.format(i) + '得分={:.2f}, '.format(score.item() * 100))
            if score < self.conf:
                flag = True
                # break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

    cw_para = np.linspace(1, 10, 10).astype('double')

    ori_dir = '../../../../dataset'
    print('加载的数据集为: ', ori_dir)

    dirlist = os.listdir(ori_dir)
    for value in cw_para:
        img_no = 0
        for dir in dirlist:
            files = os.listdir(os.path.join(ori_dir, dir))
            for f in files:
                img_name = os.path.join(ori_dir, dir, f)
                img = Image.open(img_name).convert("RGB")
                img_origin = data_transforms(img).to(device)

                img_attack = White_box_attack(img_origin, net).CW_attack(img_no, lamb= 1e-4 * value)     # lamb最大为1e-3

                img_attack = 127.5 * (img_attack.detach().cpu().numpy().transpose(1, 2, 0) + 1)
                img_attack = Image.fromarray(img_attack.astype('uint8'))

                save_folder = '../../../../output/limit_data/whitebox_data/cw/CW_' + str(value) + '/' + dir
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                save_path = save_folder + '/' + 'CW_' + str(value) + '+' + f[:-4] + '.png'
                img_attack.save(save_path)
                img_no += 1

        print("参数为" + str(value) + "的已经生成完成")
