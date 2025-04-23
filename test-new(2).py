# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import scipy.io
import yaml
import math
from model import ft_net, two_view_net, three_view_net
from utils import load_network

#fp16
try:
    from apex.fp16_utils import *
except ImportError: # will be 3.x series
    print('This is not an error. If you want to use low precision, i.e., fp16, please install the apex with cuda support (https://github.com/NVIDIA/apex) and update pytorch to 1.0')
######################################################################
# Options
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='./data/test', type=str, help='./test_data')
parser.add_argument('--name', default='three_view_long_share_d0.75_256_s1_google', type=str, help='save model path')
parser.add_argument('--pool', default='avg', type=str, help='avg|max')
parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--h', default=256, type=int, help='height')
parser.add_argument('--w', default=256, type=int, help='width')
parser.add_argument('--views', default=2, type=int, help='views')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--PCB', action='store_true', help='use PCB')
parser.add_argument('--multi', action='store_true', help='use multiple query')
parser.add_argument('--fp16', action='store_true', help='use fp16.')
parser.add_argument('--ms', default='1', type=str, help='multiple_scale: e.g. 1 1,1.1  1,1.1,1.2')
# 添加新的命令行参数
parser.add_argument('--query_name', default='query_street_name.txt', type=str, help='load query image')

opt = parser.parse_args()
###load config###
# load the training config
config_path = os.path.join('./model', opt.name, 'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)
opt.fp16 = config['fp16']
opt.use_dense = config['use_dense']
opt.use_NAS = config['use_NAS']
opt.stride = config['stride']
opt.views = config['views']

if 'h' in config:
    opt.h = config['h']
    opt.w = config['w']

if 'nclasses' in config:  # tp compatible with old config files
    opt.nclasses = config['nclasses']
else:
    opt.nclasses = 729

str_ids = opt.gpu_ids.split(',')
#which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

print('We use the scale: %s' % opt.ms)
str_ms = opt.ms.split(',')
ms = []
for s in str_ms:
    s_f = float(s)
    ms.append(math.sqrt(s_f))

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

######################################################################
# Load Data
# ---------
#
# We will use torchvision and torch.utils.data packages for loading the
# data.
#
data_transforms = transforms.Compose([
    transforms.Resize((opt.h, opt.w), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

if opt.PCB:
    data_transforms = transforms.Compose([
        transforms.Resize((384, 192), interpolation=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

data_dir = test_dir

# 自定义数据集类
class SimpleImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, query_name=None):
        self.root = root
        self.transform = transform
        # 定义支持的图像文件扩展名
        IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.webp']
        self.imgs = []
        if query_name is None:
            for f in os.listdir(root):
                if os.path.isfile(os.path.join(root, f)) and any(f.lower().endswith(ext) for ext in IMG_EXTENSIONS):
                    self.imgs.append(os.path.join(root, f))
        else:
            try:
                with open(query_name, 'r') as f:
                    for line in f:
                        img_name = line.strip()
                        img_path = os.path.join(root, img_name)
                        if os.path.isfile(img_path) and any(img_path.lower().endswith(ext) for ext in IMG_EXTENSIONS):
                            self.imgs.append(img_path)
            except FileNotFoundError:
                print(f"Error: The file {query_name} was not found.")

    def __getitem__(self, index):
        path = self.imgs[index]
        img = torchvision.datasets.folder.default_loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, 0  # 这里标签设为 0，因为没有分类信息

    def __len__(self):
        return len(self.imgs)

image_datasets = {
    'gallery_satellite': SimpleImageDataset(os.path.join(data_dir, 'gallery_satellite'), data_transforms),
    'query_street': SimpleImageDataset(os.path.join(data_dir, 'query_street'), data_transforms, query_name=opt.query_name)
}
dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batchsize,
                                   shuffle=False, num_workers=16) for x in ['gallery_satellite', 'query_street']
}
use_gpu = torch.cuda.is_available()

######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def which_view(name):
    if 'satellite' in name:
        return 1
    elif 'street' in name:
        return 2
    elif 'drone' in name:
        return 3
    else:
        print('unknown view')
    return -1

def extract_feature(model,dataloaders, view_index = 1):
    features = torch.FloatTensor()
    count = 0
    for data in dataloaders:
        img, label = data
        n, c, h, w = img.size()
        count += n
        print(count)
        ff = torch.FloatTensor(n,512).zero_().cuda()

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bilinear', align_corners=False)
                if opt.views ==2:
                    if view_index == 1:
                        outputs, _ = model(input_img, None) 
                    elif view_index ==2:
                        _, outputs = model(None, input_img) 
                elif opt.views ==3:
                    if view_index == 1:
                        outputs, _, _ = model(input_img, None, None)
                    elif view_index ==2:
                        _, outputs, _ = model(None, input_img, None)
                    elif view_index ==3:
                        _, _, outputs = model(None, None, input_img)
                ff += outputs
        # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features,ff.data.cpu()), 0)
    return features

def get_id(img_path):
    labels = []
    paths = []
    for path in img_path:
        # 假设文件名包含标签信息，例如：label_123.jpg
        base_name = os.path.basename(path)
        name_without_ext = os.path.splitext(base_name)[0]
        label = name_without_ext.split('_')[0]
        labels.append(label)
        paths.append(path)
    return labels, paths
    
def get_SatId_160k(img_path):
    labels = []
    paths = []
    for path,v in img_path:
        labels.append(v)
        paths.append(path)
    return labels, paths

def get_result_rank10(qf,gf,gl):
    query = qf.view(-1,1)
    score = torch.mm(gf, query)
    score = score.squeeze(1).cpu()
    score = score.numpy()
    index = np.argsort(score)
    index = index[::-1]
    rank10_index = index[0:10]
    result_rank10 = gl[rank10_index]
    return result_rank10

######################################################################
# Load Collected data Trained model
print('-------test-----------')

model, _, epoch = load_network(opt.name, opt)
model.classifier.classifier = nn.Sequential()
model = model.eval()
if use_gpu:
    model = model.cuda()

# Extract feature
since = time.time()

gallery_name = 'gallery_satellite'
query_name = 'query_street'

which_gallery = which_view(gallery_name)
which_query = which_view(query_name)
print('%d -> %d:'%(which_query, which_gallery))

gallery_path = image_datasets[gallery_name].imgs
f = open('gallery_name.txt','w')
for p in gallery_path:
    f.write(p+'\n')
query_path = image_datasets[query_name].imgs
f = open('query_name.txt','w')
for p in query_path:
    f.write(p+'\n')

gallery_label, gallery_path  = get_id(gallery_path)
query_label, query_path  = get_id(query_path)

if __name__ == "__main__":
    
    with torch.no_grad():
        print('-------------------extract query feature----------------------')
        query_feature = extract_feature(model,dataloaders[query_name], which_query)
        print('-------------------extract gallery feature----------------------')
        gallery_feature = extract_feature(model,dataloaders[gallery_name], which_gallery)        
        print('--------------------------ending extract-------------------------------')

    time_elapsed = time.time() - since
    print('Test complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    query_feature = query_feature.cuda()
    gallery_feature = gallery_feature.cuda()

    save_filename = 'answer.txt' #'results_rank10.txt' 
    if os.path.isfile(save_filename):
        os.remove(save_filename) 
    results_rank10 = []
    print(len(query_feature))
    gallery_label = np.array(gallery_label)
    for i in range(len(query_feature)):
        result_rank10 = get_result_rank10(query_feature[i], gallery_feature, gallery_label)
        results_rank10.append(result_rank10)
        
    results_rank10 = np.row_stack(results_rank10)
    if os.path.isfile(save_filename):
        os.remove(save_filename)
    with open(save_filename, 'w') as f:
        for row in results_rank10:
            f.write('\t'.join(map(str, row)) + '\n')
    print('You need to compress the file as *.zip before submission.')