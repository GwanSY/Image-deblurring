import utils
import torch
from torch import nn
from run.models import Generator
import time
from PIL import Image
import os
import cv2
from run.utils import convert_image
import argparse
current_path = os.path.dirname(__file__) + '/'


# 模型参数
large_kernel_size = 9   # 第一层卷积和最后一层卷积的核大小
small_kernel_size = 3   # 中间层卷积的核大小
n_channels = 64         # 中间层通道数
n_blocks = 16           # 残差模块数量
scaling_factor = 4      # 放大比例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main(image_path:str,
         weights_path= current_path +'checkpoint_srgan.pth',
         out_dir= current_path ,
         ui = None  # 传入界面参数
         ):
    # 预训练模型
    #srgan_checkpoint = weights_path
    #srresnet_checkpoint = "./results/checkpoint_srresnet.pth"
    # 加载模型SRResNet 或 SRGAN
    checkpoint = torch.load(weights_path)
    generator = Generator(large_kernel_size=large_kernel_size,
                        small_kernel_size=small_kernel_size,
                        n_channels=n_channels,
                        n_blocks=n_blocks,
                        scaling_factor=scaling_factor)
    generator = generator.to(device)
    generator.load_state_dict(checkpoint['generator'])
   
    generator.eval()
    model = generator
    print("转移数据至设备")
    # 加载图像
    img = Image.open(image_path, mode='r')
    img = img.convert('RGB')
    # img = Image.open(imgPath, mode='r')
    # img = img.convert('RGB')

    # 双线性上采样
    Bicubic_img = img.resize((int(img.width * scaling_factor),int(img.height * scaling_factor)),Image.BICUBIC)
    Bicubic_img.save(out_dir+"SRGandeblur.jpg")

    # 图像预处理
    lr_img = convert_image(img, source='pil', target='imagenet-norm')
    lr_img.unsqueeze_(0)

    # 记录时间
    start = time.time()

    # 转移数据至设备
    lr_img = lr_img.to(device)  # (1, 3, w, h ), imagenet-normed
    print("转移数据至设备")
    # 模型推理
    with torch.no_grad():
        sr_img = model(lr_img).squeeze(0).cpu().detach()  # (1, 3, w*scale, h*scale), in [-1, 1]   
        sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
        sr_img.save(out_dir+"srgan.jpg")
        if ui is not None:
            ui.ms.message.emit([sr_img])
    print('用时  {:.3f} 秒'.format(time.time()-start))

def get_files():
    list=[]
    for filepath,dirnames,filenames in os.walk("./dataset1"):
        for filename in filenames:
            list.append(os.path.join(filepath,filename))
    return list

