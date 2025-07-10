import matplotlib.pyplot as plt
import torchvision
import torch.utils.data
from torchvision import transforms
import numpy as np
import zipfile
import math
import sys
sys.path.append('../../../')
from model import model
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BATCH_SIZE = 32
os.environ['CUDA_VISIBLE_DEVICE'] = '0'

img_transform = transforms.Compose([transforms.Resize((112, 112)),
                                    transforms.ToTensor(),
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


def get_succ_loss(net, loader_ori, loader_test):
    attack_succ = 0
    test_loss = 0
    for step, (ori, test) in enumerate(zip(loader_ori, loader_test)):
        with torch.no_grad():
            ori_feature = net(ori[0].cuda())
            test_feature = net(test[0].cuda())
            cos_ori = ori_feature / torch.norm(ori_feature, dim=1).unsqueeze(1)
            cos_test = test_feature / torch.norm(test_feature, dim=1).unsqueeze(1)

        score = 0.5 * (torch.sum(cos_ori * cos_test, dim=1) + 1)
        test_loss += torch.mean(torch.ones_like(score) - score).item()
        attack_succ += sum((score < 0.8).int()).item()

        # print("第" + str(step) + "批测试样本得分计算完成")

    return 1 - attack_succ / 150, test_loss / (step + 1)


if __name__ == '__main__':
    print('===============================环境模拟-磨皮鲁棒性测试===============================')
    # 模型加载
    featuresNum = 512
    net = model.MobileFacenet(featuresNum)  # mobileFace

    ckpt = torch.load("../../../model/mobileFace-ArcFace.ckpt")
    net.load_state_dict(ckpt['net_state_dict'])
    print('----------------------------加载MobileFace模型----------------------------')
    net = net.cuda()
    net.eval()

    data_ori = '../../../dataset'
    print('加载的数据集为: ', data_ori)
    case_ori = torchvision.datasets.ImageFolder(data_ori, img_transform)
    loader_ori = torch.utils.data.DataLoader(case_ori, num_workers=0, batch_size=BATCH_SIZE, shuffle=False,
                                             drop_last=False)

    filter_para = np.linspace(1.0, 3.0, 5)
    reco_ac = []
    reco_loss = []
    # 读数据集列表 循环数据集
    test_case_dirs = '../../../output/antijam_data/filter'
    dir_list = sorted(os.listdir(test_case_dirs), key=lambda str:(str[:6], float(str[7:])))
    no = 1
    for dir in dir_list:
        test_case_name = os.path.join(test_case_dirs, dir)  # ./Dataset/FGSM/FGSM_1.0
        # 使用dataloader读取数据集
        case_test = torchvision.datasets.ImageFolder(test_case_name, img_transform)
        loader_test = torch.utils.data.DataLoader(case_test, num_workers=0, batch_size=BATCH_SIZE, shuffle=False,
                                                  drop_last=False)

        ac, loss = get_succ_loss(net, loader_ori, loader_test)
        reco_ac.append(ac)
        reco_loss.append(loss)
        print("数据集" + str(no) + "准确率%.2f%%\t损失%.4f" % (ac * 100, loss))
        no = no + 1

    # print(reco_ac)
    # print(reco_loss)
    deltaAccRise = -np.mean(np.diff(reco_ac) / np.ones(len(filter_para) - 1))
    deltaLossRise = np.mean(np.diff(reco_loss) / np.ones(len(filter_para) - 1))
    print("准确率下降%.2f%%/磨皮程度增加10%%\t" % (deltaAccRise * 100))
    print("损失值增大%.4f/磨皮程度增加10%%" % (deltaLossRise))

    # 写入数据
    filter_ac_score = round(((math.pow(100000, 1 - deltaAccRise) - 1) / (100000 - 1)) * 100, 2)   # round(100 - deltaAccRise * 100, 2)
    filter_loss_score = round(100 - deltaLossRise * 100, 2)
    ans_dict = {"filter_ac_score": filter_ac_score, "filter_loss_score": filter_loss_score}
    write_res("../../../res/result.txt", ans_dict)
