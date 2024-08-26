import imageio
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.EMFFNet_Res2Net import EMFFNet
from utils.dataloader import test_dataset
from skimage import img_as_ubyte

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default=r'C:\Users\inspur\Downloads\EMFFNet\snapshots\EMFFNet-Res2Net\EMFFNet-29.pth')

for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Nice1', 'Nice2']:
    # 'CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'Nice1', 'Nice2'
    data_path = './data/TestDataset/{}/'.format(_data_name)
    save_path = './results/EMFFNet/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = EMFFNet()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()git in

        res5, res4, res3, res2, res1 = model(image)
        res = res2 + res1
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res_uint8 = img_as_ubyte(res)
        print('> {} - {}'.format(_data_name, name))
        imageio.imwrite(save_path+name, res_uint8)