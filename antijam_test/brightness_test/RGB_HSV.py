"""
Pytorch implementation of RGB convert to HSV, and HSV convert to RGB,
RGB or HSV's shape: (B * C * H * W)
RGB or HSV's range: [0, 1)
"""
import torch
from torch import nn
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class RGB_HSV(nn.Module):
    def __init__(self, eps=1e-8):
        super(RGB_HSV, self).__init__()
        self.eps = eps

    def rgb_to_hsv(self, img):
        with torch.no_grad():
            hue = torch.Tensor(img.shape[0], img.shape[2], img.shape[3]).to(img.device)

            hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
                img[:, 2] == img.max(1)[0]]
            hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
                img[:, 1] == img.max(1)[0]]
            hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0] + self.eps))[
                img[:, 0] == img.max(1)[0]]) % 6

            hue[img.min(1)[0] == img.max(1)[0]] = 0.0
            hue = hue / 6

            saturation = (img.max(1)[0] - img.min(1)[0]) / (img.max(1)[0] + self.eps)
            saturation[img.max(1)[0] == 0] = 0

            value = img.max(1)[0]

            hue = hue.unsqueeze(1)
            saturation = saturation.unsqueeze(1)
            value = value.unsqueeze(1)
            hsv = torch.cat([hue, saturation, value], dim=1)
        return hsv

    def hsv_to_rgb(self, hsv):
        with torch.no_grad():
            h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
            # 对出界值的处理
            h = h % 1
            s = torch.clamp(s, 0, 1)
            v = torch.clamp(v, 0, 1)

            r = torch.zeros_like(h)
            g = torch.zeros_like(h)
            b = torch.zeros_like(h)

            hi = torch.floor(h * 6)
            f = h * 6 - hi
            p = v * (1 - s)
            q = v * (1 - (f * s))
            t = v * (1 - ((1 - f) * s))

            hi0 = hi == 0
            hi1 = hi == 1
            hi2 = hi == 2
            hi3 = hi == 3
            hi4 = hi == 4
            hi5 = hi == 5

            r[hi0] = v[hi0]
            g[hi0] = t[hi0]
            b[hi0] = p[hi0]

            r[hi1] = q[hi1]
            g[hi1] = v[hi1]
            b[hi1] = p[hi1]

            r[hi2] = p[hi2]
            g[hi2] = v[hi2]
            b[hi2] = t[hi2]

            r[hi3] = p[hi3]
            g[hi3] = q[hi3]
            b[hi3] = v[hi3]

            r[hi4] = t[hi4]
            g[hi4] = p[hi4]
            b[hi4] = v[hi4]

            r[hi5] = v[hi5]
            g[hi5] = p[hi5]
            b[hi5] = q[hi5]

            r = r.unsqueeze(1)
            g = g.unsqueeze(1)
            b = b.unsqueeze(1)
            rgb = torch.cat([r, g, b], dim=1)
        return rgb


if __name__ == '__main__':
    img = cv2.imread('./0_0.jpg')
    rgb = img[:, :, ::-1]  # 注意opencv是BGR顺序，必须转换成RGB
    rgb = rgb / 255

    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
    convertor = RGB_HSV()

    hsv_tensor = convertor.rgb_to_hsv(rgb_tensor)
    rgb1 = convertor.hsv_to_rgb(hsv_tensor)

    hsv_arr = hsv_tensor[0].permute(1, 2, 0).numpy()
    rgb1_arr = rgb1[0].permute(1, 2, 0).numpy()

    hsv_m = mcolors.rgb_to_hsv(rgb)
    rgb1_m = mcolors.hsv_to_rgb(hsv_m)

    print('MSE of my code and matplotlib:', ((rgb1_arr - rgb) ** 2).mean())
    plt.figure()
    plt.imshow(rgb)
    plt.title('origin image')
    plt.figure()
    plt.imshow(hsv_arr)
    plt.title('visual to hsv')
    plt.figure()
    plt.imshow(rgb1_arr)
    plt.title('convert back: my code')
    plt.figure()
    plt.imshow(rgb1_m)
    plt.title('convert back: matplotlib method')
