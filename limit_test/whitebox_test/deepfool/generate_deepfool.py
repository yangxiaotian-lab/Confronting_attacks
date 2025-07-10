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

    def __init__(self, img_origin, model, sourceFeatureNormlized=None, targetFeatureNormlized=None, featureSet=None,
                 conf=0.8):
        self.img_origin = img_origin
        self.model = model
        self.sourceFeatureNormlized = sourceFeatureNormlized
        self.targetFeatureNormlized = targetFeatureNormlized
        self.featureSet = featureSet
        self.conf = conf

    def DeepFool_attack(self, img_no, iters=120, eps=0.01 / 255, eps_max=1 / 255):
        img_origin = self.img_origin
        img_origin.requires_grad = True
        originFeature = self.model(imResize(img_origin).unsqueeze(0))
        originFeatureNormlized = originFeature / torch.norm(originFeature)
        score_ori = 0.5 * (originFeatureNormlized.matmul(originFeatureNormlized.permute(1, 0)) + 1)
        grad_ori = torch.autograd.grad(score_ori, img_origin, retain_graph=True, create_graph=True, only_inputs=True)

        img_origin = img_origin.detach()
        originFeatureNormlized = originFeatureNormlized.detach()
        flag = False
        # img_attack = img_origin.clone()
        img_attack = img_origin + 0.03 * torch.randn(img_origin.size()).to(img_origin.device)
        img_attack = torch.clamp(img_attack, -1, 1)

        max_value = 0
        r = None
        for i in range(iters):
            img_attack.requires_grad = True
            attackFeature = self.model(imResize(img_attack).unsqueeze(0))
            attackFeatureNormlized = attackFeature / torch.norm(attackFeature)
            score_attack = 0.5 * (originFeatureNormlized.matmul(attackFeatureNormlized.permute(1, 0)) + 1)
            # print('{:04d}: '.format(i) + '得分={:.2f}'.format(score_attack.item() * 100))
            if score_attack < self.conf:
                flag = True
                # break
            # score_attack.backward()
            # grad_attack = img_attack.grad.data
            grad_attack = torch.autograd.grad(score_attack, img_attack, retain_graph=True, create_graph=True, only_inputs=True)

            grad_prime = (grad_attack[0] - grad_ori[0]) * eps
            # grad_prime = grad_attack[0] - grad_ori[0]       # grad_ori是元组
            value = ((score_ori - score_attack) / (torch.norm(grad_prime) ** 2)).item()
            if max_value < value:
                r = ((score_attack - score_ori) / (torch.norm(grad_prime) ** 2)) * grad_prime
                max_value = value

            r = torch.clamp(r, -eps_max, eps_max)
            img_attack = torch.clamp(img_origin + r, -1, 1).to(img_origin.device).detach()

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

    df_para = np.linspace(1, 10, 10).astype('int')

    ori_dir = '../../../../dataset'
    print('加载的数据集为: ', ori_dir)

    dirlist = os.listdir(ori_dir)
    for value in df_para:
        img_no = 0
        for dir in dirlist:
            files = os.listdir(os.path.join(ori_dir, dir))
            for f in files:
                img_name = os.path.join(ori_dir, dir, f)
                img = Image.open(img_name).convert("RGB")
                img_origin = data_transforms(img).to(device)

                img_attack = White_box_attack(img_origin, net).DeepFool_attack(img_no, iters=20, eps_max=1 / 255 * value)

                img_attack = 127.5 * (img_attack.detach().cpu().numpy().transpose(1, 2, 0) + 1)
                img_attack = Image.fromarray(img_attack.astype('uint8'))

                save_folder = '../../../../output/limit_data/whitebox_data/deepfool/DeepFool_' + str(value) + '/' + dir
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                save_path = save_folder + '/' + 'DeepFool_' + str(value) + '+' + f[:-4] + '.png'
                img_attack.save(save_path)
                img_no += 1

        print("参数为" + str(value) + "的已经生成完成")
