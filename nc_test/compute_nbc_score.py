import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import math
import sys

sys.path.append('../../')
from model import model

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
img_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


def write_res(file_path, ans_dict):
    f = open(file_path, 'r')
    con = f.readlines()
    f.close()

    con = np.array(con)
    # 用来保存文件内数据的字典
    file_dict = {}

    for i in con:
        cur = i.strip('\n')
        # 获取文件中已经存在的key和value，写入字典
        key = cur.split('=')[0]
        value = cur.split('=')[1]
        file_dict[key] = value

    file_dict.update(ans_dict)

    f = open(file_path, 'w')
    for i in file_dict.keys():
        key = i
        value = file_dict[key]
        f.write(key + '={}'.format(value) + '\n')
    f.close()


def imResize(image, size=112):
    resizeLayer = torch.nn.AdaptiveAvgPool2d(output_size=size)
    imageResize = resizeLayer(image)
    return imageResize


def MFN_table(model):  # mobileFaceNet中间值记录
    d = {}
    neurons = ["conv1.prelu", "dw_conv1.prelu", "conv2.prelu", "linear7.bn", "linear1.bn", "blocks.1.conv.0",
               "blocks.2.conv.0", "blocks.3.conv.0", "blocks.4.conv.0", "blocks.5.conv.0", "blocks.6.conv.0",
               "blocks.7.conv.0", "blocks.8.conv.0", "blocks.9.conv.0", "blocks.10.conv.0", "blocks.11.conv.0",
               "blocks.12.conv.0", "blocks.13.conv.0", "blocks.14.conv.0", "conv2.conv"]

    def set_table(name):
        def hook(model, input, output):
            if name in ["conv1.prelu", "dw_conv1.prelu", "conv2.prelu", "linear7.bn", "linear1.bn"]:
                d[name] = output
            else:
                d[name] = input[0]  # 不取[0]的话，保存的input是元组，不知道为啥-_-

        return hook

    for name, layer in model.named_modules():  # 层名+层名对应的实际操作  eg：'conv1:conv2D(...)'
        if "fc" in name:  # 不记录全连接层
            continue
        if name in neurons:
            # 记录前向过程中层名为'name'的中间结果
            layer.register_forward_hook(set_table(name))
    return d


def init_coner(upperConer, lowerConer, output_table):
    for key in output_table.keys():
        upperConer[key] = torch.zeros(output_table[key][0].shape[0], dtype=torch.int)
        lowerConer[key] = torch.zeros(output_table[key][0].shape[0], dtype=torch.int)


def NBC(dg, dir):  # 神经元边界覆盖
    print('开始计算NBC')
    nbc = None
    img_no = 0
    files = os.listdir(dir)
    for f in files:
        pics = os.listdir(os.path.join(dir, f))
        for pic in pics:
            img_name = os.path.join(dir, f, pic)
            img = Image.open(img_name).convert("RGB")
            img_origin = img_transform(img).unsqueeze(0).to(device)
            img_resize = imResize(img_origin)

            dg.forward(img_resize)

            for key in dg.low.keys():
                out = dg.output_table[key][0]  # n * m * m  找到当前层的输出值并减去第一维
                mean_out = out.mean(dim=[1, 2])  # n     计算n维的平均值
                low = dg.low[key].to(device)
                high = dg.high[key].to(device)
                curr_low = dg.LowerConer[key].to(device).clone()
                curr_high = dg.UpperConer[key].to(device).clone()
                dg.LowerConer[key] = torch.clamp((low > mean_out).int() + curr_low, 0, 1)
                dg.UpperConer[key] = torch.clamp((high < mean_out).int() + curr_high, 0, 1)

            UpperCornerNeuron = 0
            LowerCornerNeuron = 0
            neuron_num = 0
            for key in dg.low.keys():
                LowerCornerNeuron += torch.sum(dg.LowerConer[key]).item()
                UpperCornerNeuron += torch.sum(dg.UpperConer[key]).item()
                neuron_num += len(dg.low[key])
            nbc = (LowerCornerNeuron + UpperCornerNeuron) / (2 * neuron_num)
            print('img_no:' + str(img_no) + '\tNBC: ' + str(nbc))
            img_no += 1
    return nbc

class deepGauge:
    def __init__(self, dnn):
        self.dnn = dnn  # 模型
        self.output_table = MFN_table(dnn)  # 每个模型每一层的输出值   20 * 1 * n * m * m
        self.low = torch.load('./Data_Low&High/MFN_low.pth')
        self.high = torch.load('./Data_Low&High/MFN_high.pth')
        # self.LowerConer = './Data_Coner/MFN_LowerConer.pth'
        # self.UpperConer = './Data_Coner/MFN_UpperConer.pth'
        self.LowerConer = {}
        self.UpperConer = {}

    def forward(self, img):
        with torch.no_grad():
            out = self.dnn(img)
        if not (self.LowerConer and self.UpperConer):
            init_coner(self.UpperConer, self.LowerConer, self.output_table)
        return out


if __name__ == "__main__":
    print('===============================神经元边界覆盖率鲁棒性测试===============================')
    # 模型加载
    featuresNum = 512
    net = model.MobileFacenet(featuresNum)  # mobileFace
    ckpt = torch.load("../../model/mobileFace-ArcFace.ckpt")
    print('----------------------------加载MobileFace模型----------------------------')

    net.load_state_dict(ckpt['net_state_dict'])
    net = net.to(device)
    net.eval()

    dg = deepGauge(net)  # 初始化deepgauge

    dir = '../../dataset'
    print('读入的数据集为: ', dir)

    nbc = NBC(dg, dir)  # KMNC, NBC, SNAC都与[low, high]相关

    # 写入数据
    nbc_score = round(nbc * 100, 2)
    ans_dict = {"nbc_score": nbc_score}
    write_res("../../res/result.txt", ans_dict)