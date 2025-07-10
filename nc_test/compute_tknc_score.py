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


def init_topk(topk_neuron, output_table):
    for key in output_table.keys():
        topk_neuron[key] = torch.zeros(output_table[key][0].shape[0], dtype=torch.int)


def TKNC(dg, dir):  # top-k神经元覆盖
    # 根据k的大小，每进入一个样本，点亮输出值最大的k个神经元， 跑完所有测试样本后，计算(点亮的 / 总神经元)
    tknc = None
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

            for key in dg.output_table.keys():
                out = dg.output_table[key][0]  # n * m * m  找到当前层的输出值并减去第一维
                mean_out = out.mean(dim=[1, 2])  # n     计算n维的平均值
                # topk返回最大的K个值，还有其对应的坐标， [0]表示k个最大值的集合，[1]表示k个最大值的集合的坐标
                index = torch.topk(mean_out, dg.topK_num)[1]
                for i in range(index.shape[0]):
                    location = index[i].item()
                    dg.topk_neuron[key][location] = 1

            topk_num = 0
            neuron_num = 0
            for key in dg.topk_neuron.keys():
                topk_num += torch.sum(dg.topk_neuron[key]).item()
                neuron_num += dg.topk_neuron[key].shape[0]
            tknc = topk_num / neuron_num
            print('img_no:' + str(img_no) + '\tk = ' + str(dg.topK_num) + '\tTKNC: ' + str(tknc))
            img_no += 1
    return tknc


class deepGauge:
    def __init__(self, dnn, topK_num):
        self.dnn = dnn  # 模型
        self.topK_num = topK_num
        self.output_table = MFN_table(dnn)  # 每个模型每一层的输出值   20 * 1 * n * m * m
        # self.topk_neuron = torch.load('./Data_TKNC/MFN_TKNC_' + str(self.topK_num) + '.pth')
        self.topk_neuron = {}

    def forward(self, img):
        with torch.no_grad():
            out = self.dnn(img)
        if not self.topk_neuron:
            init_topk(self.topk_neuron, self.output_table)
        return out


if __name__ == "__main__":
    print('===============================Top-K神经元覆盖率鲁棒性测试===============================')
    # 模型加载
    featuresNum = 512
    net = model.MobileFacenet(featuresNum)  # mobileFace
    ckpt = torch.load("../../model/mobileFace-ArcFace.ckpt")
    print('----------------------------加载MobileFace模型----------------------------')

    net.load_state_dict(ckpt['net_state_dict'])
    net = net.to(device)
    net.eval()

    dg = deepGauge(net, topK_num=3)  # 初始化deepgauge
    # get_low_high(dg, model_index)  # 获取low, high区间

    dir = '../../dataset'
    print('读入的数据集为: ', dir)

    tknc = TKNC(dg, dir)

    # 写入数据
    tknc_score = round(tknc * 100, 2)
    ans_dict = {"tknc_score": tknc_score}
    write_res("../../res/result.txt", ans_dict)
