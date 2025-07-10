import torch
from PIL import Image
from torchvision.transforms import functional as TF
import os
import numpy as np
from torchvision import transforms

img_transform = transforms.Compose([transforms.Resize((112, 112)),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

if __name__ == '__main__':
    white_size = np.linspace(1, 4, 4).astype('int')

    ori_dir = '../../../dataset'
    img_no = 0
    dirlist = os.listdir(ori_dir)
    for value in white_size:
        for dirs in dirlist:
            files = os.listdir(os.path.join(ori_dir, dirs))
            for f in files:
                img_name = os.path.join(ori_dir, dirs, f)
                img = Image.open(img_name).convert("RGB")

                ori = img_transform(img).unsqueeze(0)


                img_changed = TF.adjust_contrast(img, value)


                save_folder = '../../../output/antijam_data/white/white_' + str(value) + '/' + dirs
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                save_path = save_folder + '/' + 'white_' + str(value) + '+' + f[:-4] + '.png'
                img_changed.save(save_path)
                img_no += 1

        print("参数为" + str(value) + "的已经生成完成")

