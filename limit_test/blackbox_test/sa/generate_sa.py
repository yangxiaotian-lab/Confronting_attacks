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

sys.path.append('..')

# 对图像数据类型转换为tensor类型
data_transforms = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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

    def SA_attack(self, img_no, eps=2 / 255 * 2, T_start=100, T_end=20, T_decay=0.9, k=1e-3, mark_chain=20):
        img_orign = self.img_orign.clone()
        img_attack_now = img_orign
        loss_now = 1
        T = T_start
        flag = False
        smart_count = 0
        j = 0
        with torch.no_grad():
            while T > T_end:
                # print("温度：" + str(T))
                for i in range(mark_chain):
                    if self.do_noise_surppression:
                        noise = self.noise_surppression(SNR=10)
                    else:
                        noise = eps * torch.sign(torch.rand(img_orign.size()) - 0.5).to(img_orign.device)
                    img_attack_new = img_attack_now + noise
                    img_attack_new = torch.clamp(img_attack_new, -1, 1)
                    attackFeature = self.model(imResize(img_attack_new).unsqueeze(0)).detach()
                    attackFeatureNormlized = attackFeature / torch.norm(attackFeature)
                    loss_new = 0.5 * (self.sourceFeatureNormlized.matmul(attackFeatureNormlized.permute(1, 0)) + 1)
                    if loss_new > loss_now:
                        smart_count += 1
                    if np.random.rand() < torch.exp((loss_now - loss_new) / k * T) or smart_count > 100:
                        img_attack_now = img_attack_new
                        loss_now = loss_new
                        smart_count = 0
                    # print('{:04d}：'.format(j) + '得分={:.2f}, '.format(loss_now.item() * 100) + '温度={:.2f}'.format(T))

                    j = j + 1
                    # if j >= 500:
                    #     flag = False
                    #     break
                    if loss_now < self.conf:
                        flag = True
                        # break
                # if flag:
                #     break
                # if not flag and j >= 500:
                #     break
                T = T * T_decay
        if flag:
            print('img_no: ' + str(img_no) + ' 攻击成功')
        else:
            print('img_no: ' + str(img_no) + ' 攻击失败')
        out = img_attack_now
        return out


if __name__ == "__main__":
    # 模型加载
    featuresNum = 512
    net = model.MobileFacenet(featuresNum)  # mobileFace
    ckpt = torch.load("../../../../model/mobileFace-ArcFace.ckpt")
    print('----------------------------加载MobileFace模型----------------------------')

    net.load_state_dict(ckpt['net_state_dict'])
    net = net.cuda()
    net.eval()

    sa_para = np.linspace(1, 10, 5).astype('double')

    ori_dir = '../../../../dataset'
    print('加载的数据集为: ', ori_dir)
    img_no = 0
    dirlist = os.listdir(ori_dir)
    for value in sa_para:
        for dir in dirlist:
            files = os.listdir(os.path.join(ori_dir, dir))
            for f in files:
                img_name = os.path.join(ori_dir, dir, f)

                featureNormlized = Black_box_attack.getNormlizedFeature(img_name, net, device)
                img = Image.open(img_name).convert("RGB")

                img_origin = data_transforms(img).to(device)

                img_attack = Black_box_attack(img_origin, net, featureNormlized).SA_attack(img_no, eps=1 / 255 * value)

                img_attack = 127.5 * (img_attack.detach().cpu().numpy().transpose(1, 2, 0) + 1)
                img_attack = Image.fromarray(img_attack.astype('uint8'))

                save_folder = '../../../../output/limit_data/blackbox_data/sa/SA_' + str(value) + '/' + dir
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                save_path = save_folder + '/' + 'SA_' + str(value) + '+' + f[:-4] + '.png'
                img_attack.save(save_path)
                img_no += 1

        print("参数为" + str(value) + "的已经生成完成")
