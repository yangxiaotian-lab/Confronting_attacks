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

# 对图像数据类型转换为tensor类型
data_transforms = transforms.Compose([
    # transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


def imResize(image, size=112):
    resizeLayer = torch.nn.AdaptiveAvgPool2d(output_size=size)
    imageResize = resizeLayer(image)
    return imageResize


class Black_box_attack(object):
    def __init__(self, img_orign, model, sourceFeatureNormlized, conf=0.8, do_noise_surppression=False):
        self.img_orign = img_orign
        self.model = model
        self.sourceFeatureNormlized = sourceFeatureNormlized
        self.conf = conf
        self.do_noise_surppression = do_noise_surppression

    def noise_surppression(self, SNR=10):
        img_in = self.img_orign.clone()
        kernel = cv2.getGaussianKernel(5, 1.5)
        window = np.outer(kernel, kernel.transpose())
        img = img_in.detach().cpu().numpy()
        img_sigma_2 = []
        for i in range(np.size(img, 0)):
            mu = cv2.filter2D(img[i], -1, window)
            mu_sq = mu ** 2
            sigma_sq = cv2.filter2D(img[i] ** 2, -1, window) - mu_sq
            img_sigma_2.append(sigma_sq)
        img_sigma_2 = np.array(img_sigma_2) + 1e-6
        noise_std = torch.sqrt(torch.Tensor(img_sigma_2 / np.power(10, SNR / 10))).to(self.img_orign.device)
        noise = noise_std * torch.randn_like(self.img_orign)
        return noise

    @staticmethod
    def getNormlizedFeature(img_path, model, device):
        img = Image.open(img_path).convert("RGB")
        img = data_transforms(img).to(device)
        feature = model(imResize(img).unsqueeze(0)).detach()
        return feature / torch.norm(feature)

    def GA_attack(self, img_no, eps=2 / 255 * 2, iters=100, pop_nums=50, elite_nums=10, variation_rate=1 / 5):
        img_orign = self.img_orign.clone()
        # pop init
        img_attack_pop = []
        for i in range(pop_nums):
            img_attack_pop.append(img_orign)

        # evolution
        flag = False
        for i in range(iters):
            # choice
            with torch.no_grad():
                attackFeatures = self.model(imResize(torch.stack(img_attack_pop)))
                attackFeaturesNormlized = attackFeatures / torch.norm(attackFeatures, dim=1).unsqueeze(1)
                loss = (attackFeaturesNormlized.matmul(self.sourceFeatureNormlized.permute(1, 0)) + 1) * 0.5
                topk_value, topk_index = torch.topk(loss.squeeze(1), elite_nums, largest=False)

            # print('{:04d}：'.format(i) + '得分={:.2f}, '.format(topk_value[0].item() * 100))
            if topk_value[0] < self.conf:
                # print("攻击成功!")
                flag = True
                # break
            # inheritance
            for j in range(elite_nums):
                img_attack_pop[j] = img_attack_pop[topk_index[j]]
            for j in range(elite_nums, pop_nums):       # 精英个体数量为10
                rand_index = torch.randint(elite_nums, (2,))    # 每次从0-10之间取两个数出来
                img_attack_father = img_attack_pop[topk_index[rand_index[0]]]   # 取出两个精英个体
                img_attack_mother = img_attack_pop[topk_index[rand_index[1]]]
                img_mask = ((torch.rand(img_orign.size()) < 0.5).float()).to(img_orign.device)
                img_attack_pop[j] = img_mask * img_attack_father + (
                            torch.ones_like(img_mask) - img_mask) * img_attack_mother  # 将父亲的一部分加母亲的一部分叠加在一起
            # variaton
            for j in range(elite_nums, pop_nums):
                if self.do_noise_surppression:  # 是否噪声抑制
                    noise = self.noise_surppression(SNR=10)
                else:
                    noise = eps * torch.sign(torch.rand(img_orign.size()) - 0.5).to(img_orign.device)
                img_mask = ((torch.rand(img_orign.size()) < variation_rate).float()).to(img_orign.device)
                img_attack_pop[j] = img_attack_pop[j] + img_mask * noise
                img_attack_pop[j] = torch.clamp(img_attack_pop[j], -1, 1)
        if flag:
            print('img_no: ' + str(img_no) + ' 攻击成功')
        else:
            print('img_no: ' + str(img_no) + ' 攻击失败')
        img_attack = img_attack_pop[topk_index[0]]
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

    ga_para = np.linspace(1, 10, 5).astype('double')

    ori_dir = '../../../../dataset'
    print('加载的数据集为: ', ori_dir)
    img_no = 0
    dirlist = os.listdir(ori_dir)
    for value in ga_para:
        for dir in dirlist:
            files = os.listdir(os.path.join(ori_dir, dir))
            for f in files:
                img_name = os.path.join(ori_dir, dir, f)
                featureNormlized = Black_box_attack.getNormlizedFeature(img_name, net, device)

                img = Image.open(img_name).convert("RGB")
                img_origin = data_transforms(img).to(device)
                img_attack = Black_box_attack(img_origin, net, featureNormlized).GA_attack(img_no, eps=1 / 255 * value)

                img_attack = 127.5 * (img_attack.detach().cpu().numpy().transpose(1, 2, 0) + 1)
                img_attack = Image.fromarray(img_attack.astype('uint8'))

                save_folder = '../../../../output/limit_data/blackbox_data/ga/GA_' + str(value) + '/' + dir
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                save_path = save_folder + '/' + 'GA_' + str(value) + '+' + f[:-4] + '.png'
                img_attack.save(save_path)
                img_no += 1

        print("参数为" + str(value) + "的已经生成完成")
