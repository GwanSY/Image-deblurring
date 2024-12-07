import os
import argparse
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
import torch
import DeepblurGan.model.model as module_arch
from data_loader.data_loader import CustomDataLoader
from utils.util import denormalize
import cv2
import numpy as np
current_path = os.path.dirname(__file__) + '/'

def main(blurred_dir,
         weights_path=current_path + 'pretrained_weights/checkpoint-epoch300.pth',
         out_dir=current_path ,
         ui = None  # 传入界面参数# 传入界面参数
         ):
    # load checkpoint
    checkpoint = torch.load(weights_path)
    config = checkpoint['config']
    # setup data_loader instances
    data_loader = CustomDataLoader(data_dir=blurred_dir)

    # build model architecture
    generator_class = getattr(module_arch, config['generator']['type'])
    generator = generator_class(**config['generator']['args'])

    # prepare model for deblurring
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    generator.to(device)

    generator.load_state_dict(checkpoint['generator'])

    generator.eval()

    # start to deblur
    with torch.no_grad():
        for batch_idx, sample in enumerate(tqdm(data_loader, ascii=True)):
            blurred = sample['blurred'].to(device)
            image_name = sample['image_name'][0]
            deblurred = generator(blurred)
            deblurred_np = np.array(to_pil_image(denormalize(deblurred).squeeze().cpu()))
            deblurred_array = np.array(deblurred_np)
            # 将 numpy 数组转换为 BGR 格式
            deblurred_bgr = cv2.cvtColor(deblurred_array, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(out_dir, 'deblurred ' + image_name), deblurred_bgr)
            #deblurred_img.save(os.path.join(out_dir, 'deblurred ' + image_name))
            if ui is not None:
                ui.ms.message.emit([ deblurred_bgr])
            print('weights_path', weights_path)
if __name__ == '__main__':
    def main(weights_path=current_path + 'pretrained_weights/checkpoint-epoch300.pth',
         out_dir=current_path + "../result",
         ui = None):
        print(weights_path)
        parser = argparse.ArgumentParser(description='Deblur your own image!')
        parser.add_argument('-b', '--blurred', default=ui, type=str, help='dir of blurred images')
        parser.add_argument('-d', '--deblurred', default=out_dir, type=str, help='dir to save deblurred images')
        parser.add_argument('-r', '--resume', default=weights_path, type=str, help='path to latest checkpoint')
        parser.add_argument('--device', default=None, type=str, help='indices of GPUs to enable (default: all)')
        args = parser.parse_args()
        if args.device:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        #传入args.blurred, args.deblurred, args.weights_path三个参数
        main(args.blurred, args.deblurred, args.weights_path)
