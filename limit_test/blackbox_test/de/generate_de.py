# encoding:utf-8
import os
import torch
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
import sys
sys.path.append('../../../../')
from model import model
import random


def imResize(image, size=112):
    resizeLayer = torch.nn.AdaptiveAvgPool2d(output_size=size)
    imageResize = resizeLayer(image)
    return imageResize


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 对图像数据类型转换为tensor类型
data_transforms = transforms.Compose([
    # transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])


class Black_box_attack(object):

    def __init__(self, img_origin, model, sourceFeatureNormlized, conf=0.8, do_noise_surppression=False):
        self.img_origin = img_origin
        self.model = model
        self.conf = conf
        self.sourceFeatureNormlized = sourceFeatureNormlized
        self.do_noise_surppression = do_noise_surppression

    @staticmethod
    def getNormlizedFeature(img_path, model, device):
        img = Image.open(img_path).convert("RGB")
        img = data_transforms(img).to(device)
        feature = model(imResize(img).unsqueeze(0)).detach()
        return feature / torch.norm(feature)

    def DE_attack(self, img_no, eps, iters=30, pop_nums=50, elite_nums=8, cross_rate=0.5, F=0.4, eps_max=24/255):
        img_origin = self.img_origin.clone()

        img_attack_pop = []
        for i in range(pop_nums):
            img_attack_pop.append(img_origin + 0.002 * torch.randn(img_origin.size()).to(img_origin.device))
            # img_attack_pop.append(img_origin)

        flag = False
        for i in range(iters):

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

            new_pop = []
            # 求变异解  求交叉解（由变异解与旧解生成）     比较交叉解与旧解得到子代
            for j in range(pop_nums):
                # 变异
                sample_list = list(np.linspace(0, pop_nums - 1, pop_nums).astype('int'))
                sample_list.remove(j)
                index_list = random.sample(sample_list, 3)
                # r0, r1, r2为属于[1,…, pop_nums]的三个随机数, 排除当前个体下标
                r1 = img_attack_pop[index_list[0]]
                r2 = img_attack_pop[index_list[1]]
                r3 = img_attack_pop[index_list[2]]
                v = r1 + F * (r2 - r3)
                # 交叉
                mask = ((torch.rand(img_origin.size()) < cross_rate).float()).to(img_origin.device)
                noise = eps * torch.sign(torch.rand(img_origin.size()) - 0.5).to(img_origin.device)
                # u = (torch.ones_like(mask) - mask) * img_attack_pop[j] + mask * (v + noise)  # 交叉解
                u = (torch.ones_like(mask) - mask) * img_attack_pop[j] + mask * v + noise
                # u_noise = u - img_origin
                # u_noise = torch.clamp(u_noise, -eps_max, eps_max)
                # u = img_origin + u_noise
                # 计算特征
                with torch.no_grad():
                    img_attack_pop[j] = torch.clamp(img_attack_pop[j], -1, 1)    # 旧解
                    oldFeature = self.model(imResize(img_attack_pop[j]).unsqueeze(0).detach())
                    oldFeaturesNormlized = oldFeature / torch.norm(oldFeature, dim=1).unsqueeze(1)

                    u_res = torch.clamp(u, -1, 1)   # 新解
                    u_Feature = self.model(imResize(u_res).unsqueeze(0).detach())
                    u_FeaturesNormlized = u_Feature / torch.norm(u_Feature, dim=1).unsqueeze(1)

                # 选择 img_attack_pop[j] or u ?
                score1 = (torch.sum(oldFeaturesNormlized * self.sourceFeatureNormlized) + 1) * 0.5     # 旧解得分
                score2 = (torch.sum(u_FeaturesNormlized * self.sourceFeatureNormlized) + 1) * 0.5     # 新解得分
                new_pop.append(img_attack_pop[j] if score1 < score2 else u)

                # img_attack_pop = [torch.clamp(img_origin + new_noise_pop[k], 0, 1) for k in range(pop_nums)]
            img_attack_pop = new_pop

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

    de_para = np.linspace(1, 10, 10).astype('double')

    ori_dir = '../../../../dataset'
    print('加载的数据集为: ', ori_dir)
    img_no = 0
    dirlist = os.listdir(ori_dir)
    for value in de_para:
        for dir in dirlist:
            files = os.listdir(os.path.join(ori_dir, dir))
            for f in files:
                img_name = os.path.join(ori_dir, dir, f)
                featureNormlized = Black_box_attack.getNormlizedFeature(img_name, net, device)

                img = Image.open(img_name).convert("RGB")
                img_origin = data_transforms(img).to(device)
                img_attack = Black_box_attack(img_origin, net, featureNormlized).DE_attack(img_no, eps=1 / 255 * value)

                img_attack = 127.5 * (img_attack.detach().cpu().numpy().transpose(1, 2, 0) + 1)
                img_attack = Image.fromarray(img_attack.astype('uint8'))

                save_folder = '../../../../output/limit_data/blackbox_data/de/DE_' + str(value) + '/' + dir
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                save_path = save_folder + '/' + 'DE_' + str(value) + '+' + f[:-4] + '.png'
                img_attack.save(save_path)
                img_no += 1

        print("参数为" + str(value) + "的已经生成完成")
