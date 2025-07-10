import os
import torch.utils.data
from torch import nn
import sys
import torchvision
sys.path.append('../../../')
from gaussfilter import *
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BATCH_SIZE = 256
os.environ['CUDA_VISIBLE_DEVICE'] = '0'

img_transform = transforms.Compose([transforms.Resize((112, 112)),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

if __name__ == '__main__':
    core_size = 11
    sigma = np.linspace(1.0, 3.0, 5)

    ori_dir = '../../../dataset'
    img_no = 0
    dirlist = os.listdir(ori_dir)
    for value in sigma:
        for dirs in dirlist:
            files = os.listdir(os.path.join(ori_dir, dirs))
            for f in files:
                img_name = os.path.join(ori_dir, dirs, f)
                img = Image.open(img_name).convert("RGB")

                ori = img_transform(img).unsqueeze(0).cuda()
                out = GaussianBlur(ori, core_size, value)
                out = (out + 1) / 2

                img_changed = Image.fromarray(
                    torch.clamp(out.squeeze() * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy())
                # plt.imshow(img_changed)
                # plt.show()

                save_folder = '../../../output/antijam_data/filter/filter_' + str(value) + '/' + dirs
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                save_path = save_folder + '/' + 'filter_' + str(value) + '+' + f[:-4] + '.png'
                img_changed.save(save_path)
                img_no += 1

        print("参数为" + str(value) + "的已经生成完成")
