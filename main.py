import torch
import torch.nn as nn
import copy
import numpy as np
import glob
import random
import math
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 鲁棒性得分
ROBUST_SCORE = 0
# 一级指标
ANTIJAM_TEST_SCORE = 0      # 抗干扰测试
LIMIT_TEST_SCORE = 0        # 极限测试

# 二级指标
vulnerability_score = 0     # 漏洞检出率
nc_score = 0                # 神经元覆盖率
es_score = 0                # 环境模拟
blackbox_score = 0             # 黑盒
whitebox_score = 0             # 白盒
gan_score = 0               # 生成式对抗网络

# 三级指标
kmnc_score = 0
nbc_score = 0
snac_score = 0
tknc_score = 0

block_score = 0
brightness_score = 0
filter_score = 0
white_score = 0

de_score = 0
sa_score = 0
ga_score = 0

pgd_score = 0
cw_score = 0
fgsm_score = 0
df_score = 0


def get_res(file_path):
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
    # print(file_dict)
    # print(len(file_dict))
    return file_dict


target_dict = get_res("../res/result.txt")      # 读入字典

for key in target_dict.keys():
    target_dict[key] = float(target_dict[key])

vulnerability_score = target_dict['vulnerability_score']

kmnc_score = target_dict['kmnc_score']
nbc_score = target_dict['nbc_score']
snac_score = target_dict['snac_score']
tknc_score = target_dict['tknc_score']
nc_score = np.mean([kmnc_score, nbc_score, snac_score, tknc_score])

block_score = np.mean([target_dict['block_ac_score'], target_dict['block_loss_score']])
brightness_score = np.mean([target_dict['brightness_ac_score'], target_dict['brightness_loss_score']])
filter_score = np.mean([target_dict['filter_ac_score'], target_dict['filter_loss_score']])
white_score = np.mean([target_dict['white_ac_score'], target_dict['white_loss_score']])
es_score = np.mean([block_score, brightness_score, filter_score, white_score])

de_score = np.mean([target_dict['de_ac_score'], target_dict['de_loss_score']])
sa_score = np.mean([target_dict['sa_ac_score'], target_dict['sa_loss_score']])
ga_score = np.mean([target_dict['ga_ac_score'], target_dict['ga_loss_score']])
blackbox_score = np.mean([de_score, sa_score, ga_score])

fgsm_score = np.mean([target_dict['fgsm_ac_score'], target_dict['fgsm_loss_score']])
cw_score = np.mean([target_dict['cw_ac_score'], target_dict['cw_loss_score']])
df_score = np.mean([target_dict['df_ac_score'], target_dict['df_loss_score']])
pgd_L2_score = np.mean([target_dict['pgd_L2_ac_score'], target_dict['pgd_L2_loss_score']])
pgd_Linf_score = np.mean([target_dict['pgd_Linf_ac_score'], target_dict['pgd_Linf_loss_score']])
pgd_score = np.mean([pgd_L2_score, pgd_Linf_score])
whitebox_score = np.mean([fgsm_score, cw_score, pgd_score, df_score])

gan_score = target_dict['gan_ac_score']

ANTIJAM_TEST_SCORE = 0.2 * vulnerability_score + 0.2 * nc_score + 0.6 * es_score
LIMIT_TEST_SCORE = 0.4 * blackbox_score + 0.4 * whitebox_score + 0.2 * gan_score

ROBUST_SCORE = 0.5 * ANTIJAM_TEST_SCORE + 0.5 * LIMIT_TEST_SCORE

print("鲁棒性综合评分：", ROBUST_SCORE)
print("\t抗干扰能力得分：", ANTIJAM_TEST_SCORE)
print("\t\t神经元覆盖率得分：", nc_score)
print("\t\t\tk-多区域神经元覆盖率得分：", kmnc_score)
print("\t\t\t边缘神经元法覆盖率得分：", nbc_score)
print("\t\t\t强神经元覆盖率得分：", snac_score)
print("\t\t\tTopk神经元覆盖率得分：", tknc_score)
print("\t\t漏洞检出率得分：", vulnerability_score)
print("\t\t基于环境模拟的抗干扰测试得分：", es_score)
print("\t\t\t镜头遮挡模拟抗干扰测试得分：", block_score)
print("\t\t\t亮度模拟抗干扰测试得分：", brightness_score)
print("\t\t\t磨皮模拟抗干扰测试得分：", filter_score)
print("\t\t\t美白模拟抗干扰测试得分：", white_score)
print("\t极限测试得分：", LIMIT_TEST_SCORE)
print("\t\t黑盒条件下极限测试得分：", blackbox_score)
print("\t\t\t基于模拟退火优化算法极限测试得分：", sa_score)
print("\t\t\t基于遗传算法极限测试得分：", ga_score)
print("\t\t\t基于差分进化算法极限测试得分：", de_score)
print("\t\t白盒条件下极限测试得分：", whitebox_score)
print("\t\t\t基于单步梯度算法极限测试得分：", fgsm_score)
print("\t\t\t基于多步梯度算法极限测试得分：", pgd_score)
print("\t\t\t基于梯度优化算法极限测试得分：", cw_score)
print("\t\t\t基于决策面分类算法极限测试得分：", df_score)
print("\t\t生成对抗网络下极限测试得分：", gan_score)