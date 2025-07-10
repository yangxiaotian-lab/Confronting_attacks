import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch
from PIL import Image
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
img_transform1 = transforms.Compose([transforms.RandomRotation((5, 5)),
                                     transforms.ToTensor()])
img_transform = transforms.Compose([transforms.Resize([112, 112]),
                                    transforms.ToTensor()])


def imResize(image, size=112):
    resizeLayer = torch.nn.AdaptiveAvgPool2d(output_size=size)
    imageResize = resizeLayer(image)
    return imageResize


def getGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
        # 根据 kernelsize 计算默认的 sigma，和 opencv 保持一致
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center)  # 元素与矩阵中心的横向距离
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2))  # 计算一维卷积核
    # 根据指数函数性质，利用矩阵乘法快速计算二维卷积核
    kernel = kernel1d[..., None] @ kernel1d[None, ...]
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum()  # 归一化
    return kernel


def GaussianBlur(batch_img, ksize, sigma=None):
    kernel = getGaussianKernel(ksize, sigma).cuda()  # 生成权重
    B, C, H, W = batch_img.shape
    # 生成 group convolution 的卷积核
    kernel = kernel.view(1, 1, ksize, ksize).repeat(C, 1, 1, 1)
    pad = (ksize - 1) // 2  # 保持卷积前后图像尺寸不变
    # mode=relfect 更适合计算边缘像素的权重
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')
    weighted_pix = F.conv2d(batch_img_pad, weight=kernel, bias=None, stride=1, padding=0, groups=C)
    return weighted_pix


if __name__ == "__main__":
    # input = torch.FloatTensor(1, 3, 111, 111).to(device)
    # ga = get_gaussian_blur(3, device)
    # x = ga(input)
    # print(x)
    # x = imResize(input, 112)
    # print(x.shape)

    image_path = "./0_0.jpg"
    image = Image.open(image_path).convert("RGB")
    # image.show()
    img = img_transform(image).unsqueeze(0).cuda()
    # img = img_transform1(image).unsqueeze(0).cuda()
    res = GaussianBlur(img, 5, 1.42)
    print(res.shape)
    out = res.squeeze().detach().cpu().numpy()
    print(out.shape)
    out = np.transpose(out, (1, 2, 0))
    plt.imshow(out)
    plt.show()
