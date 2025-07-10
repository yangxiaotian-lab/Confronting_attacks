import os
import torch.utils.data
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import sys
sys.path.append('../../../')
from RGB_HSV import RGB_HSV
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

BATCH_SIZE = 1024
os.environ['CUDA_VISIBLE_DEVICE'] = '0'

img_transform = transforms.Compose([transforms.Resize((112, 112)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

if __name__ == '__main__':
    brightness = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
    convertor = RGB_HSV()  # 转换类实例

    ori_dir = '../../../dataset'
    img_no = 0
    dirlist = os.listdir(ori_dir)
    for value in brightness:
        for dirs in dirlist:
            files = os.listdir(os.path.join(ori_dir, dirs))
            for f in files:
                img_name = os.path.join(ori_dir, dirs, f)
                img = Image.open(img_name).convert("RGB")

                ori = img_transform(img).unsqueeze(0)
                img_HSV = convertor.rgb_to_hsv((ori + 1) / 2)  # [-1, 1]->[0, 1] RGB 转到 HSV
                img_HSV[:, 2, :, :] = img_HSV[:, 2, :, :] * value  # [H, S, V] 更改光照V
                out = convertor.hsv_to_rgb(img_HSV)  # [0, 1] HSV 转到 RGB
                # out = out * 2 - 1  # [0, 1] -> [-1, 1]

                img_changed = Image.fromarray(torch.clamp(out.squeeze() * 255, min=0, max=255).byte().permute(1, 2, 0).cpu().numpy())
                # plt.imshow(img_changed)
                # plt.show()

                save_folder = '../../../output/antijam_data/brightness/brightness_' + str(value) + '/' + dirs
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                save_path = save_folder + '/' + 'brightness_' + str(value) + '+' + f[:-4] + '.png'
                img_changed.save(save_path)
                img_no += 1

        print("参数为" + str(value) + "的已经生成完成")