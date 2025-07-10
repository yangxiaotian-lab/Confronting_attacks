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


def init_coverage(coverage_table, K_multi):
    for key in K_multi.keys():
        coverage_table[key] = torch.zeros(K_multi[key].shape, dtype=torch.int)


def get_K_multi(dg, low, high):
    dict = {}  # 用于记录各个层神经元的k多节区间
    temp = {}
    for k_low, v_low in low.items():  # low跟high为tensor
        for k_high, v_high in high.items():
            if k_low == k_high:
                temp[k_low] = []
                temp[k_low].append(low[k_low])
                temp[k_low].append(high[k_low])  # 将low, high组合起来

    for key, value in temp.items():
        dict[key] = []
        for i in range(len(value[0])):
            # section = torch.linspace(value[0][i], value[1][i], dg.K)    # value[0]代表low， value[1]代表high
            section = np.linspace(value[0][i].item(), value[1][i].item(), dg.K_num)
            dict[key].append(section)
        # dict[key] = torch.Tensor(dict[key]).to(device)
        dict[key] = torch.Tensor(dict[key])
    return dict


def binarySearch(section, searchNum):
    length = len(section)
    low = 0
    high = length - 1
    while low <= high:
        mid = low + int((high - low) / 2)
        if section[mid] > searchNum:
            high = mid - 1
        elif section[mid] < searchNum:
            low = mid + 1
        else:
            return mid
    return high


def KMNC(dg, dir):  # K多节神经元覆盖率
    print('开始计算KMNC')
    dg.K_multi = get_K_multi(dg, dg.low, dg.high)  # 进一步划分[low, high]区间
    print('区间划分完成')
    kmnc = None
    img_no = 0
    files = os.listdir(dir)
    for f in files:
        pics = os.listdir(os.path.join(dir, f))
        for pic in pics:
            img_name = os.path.join(dir, f, pic)
            img = Image.open(img_name).convert("RGB")
            img_origin = img_transform(img).unsqueeze(0).to(device)
            img_resize = imResize(img_origin)

            with torch.no_grad():
                dg.forward(img_resize)

            for key in dg.K_multi.keys():
                out = dg.output_table[key][0].detach().cpu()  # n * m * m  找到当前层的输出值并减去第一维
                mean_out = out.mean(dim=[1, 2])  # n     计算n维的平均值
                low = dg.low[key].detach().cpu()
                high = dg.high[key].detach().cpu()
                for i in range(dg.K_multi[key].shape[0]):  # 0-n      进入某一维
                    curr_value = mean_out[i]  # 获取当前维度对应的平均值
                    # if curr_value < dg.low[key][i].item() or curr_value > dg.high[key][i].item():  # 有可能落在[low, high]之外
                    if curr_value < low[i] or curr_value > high[i]:
                        continue
                    location = binarySearch(dg.K_multi[key][i], curr_value.item())  # 二分查找区间
                    dg.coverage_table[key][i][location] = 1

            activate = 0
            neurons_num = 0
            for key in dg.K_multi.keys():
                neurons_num += dg.K_multi[key].shape[0] * dg.K_num  # 计算该层总共有多少个区间
                activate += torch.sum(dg.coverage_table[key]).item()  # 计算该层总共激活了多少个区间
            kmnc = activate / neurons_num
            print('img_no:' + str(img_no) + '\tK_num = ' + str(dg.K_num) + '\tKMNC: ' + str(kmnc))  # 每进来一张图片算一次kmnc
            img_no += 1
    return kmnc


class deepGauge:
    def __init__(self, dnn, K_num):
        self.dnn = dnn  # 模型
        self.K_num = K_num
        self.K_multi = {}  # 记录每个神经元的k多节区间  20 * n * 1000
        self.output_table = MFN_table(dnn)  # 每个模型每一层的输出值   20 * 1 * n * m * m
        self.low = torch.load('./Data_Low&High/MFN_low.pth')
        self.high = torch.load('./Data_Low&High/MFN_high.pth')
        # self.coverage_table = torch.load('./Data_KMNC/MFN_KMNC_' + str(self.K_num) + '.pth')
        self.coverage_table = {}

    def forward(self, img):
        with torch.no_grad():
            out = self.dnn(img)
        if not self.coverage_table:  # ct一开始为空，进入该分支
            init_coverage(self.coverage_table, self.K_multi)  # 初始化覆盖设置为False
        return out


if __name__ == "__main__":
    print('===============================K-多区域神经元覆盖率鲁棒性测试===============================')
    # 模型加载
    featuresNum = 512
    net = model.MobileFacenet(featuresNum)  # mobileFace
    ckpt = torch.load("../../model/mobileFace-ArcFace.ckpt")
    print('----------------------------加载MobileFace模型----------------------------')

    net.load_state_dict(ckpt['net_state_dict'])
    net = net.to(device)
    net.eval()

    dg = deepGauge(net, K_num=100)  # 初始化deepgauge

    dir = '../../dataset'
    print('读入的数据集为: ', dir)

    kmnc = KMNC(dg, dir)  # KMNC[low, high]相关

    # 写入数据
    kmnc_score = round(kmnc * 100, 2)
    ans_dict = {"kmnc_score": kmnc_score}
    write_res("../../res/result.txt", ans_dict)
