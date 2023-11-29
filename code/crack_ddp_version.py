import warnings
warnings.filterwarnings(action='ignore')

import random
import pandas as pd
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import argparse
import shutil
import time
import builtins
from time import time
import visdom

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import torchvision
import torchvision.models as models

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from tqdm.auto import tqdm
from torchmetrics.detection import MeanAveragePrecision

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
import torch.distributed as dist

warnings.filterwarnings("ignore")


class CustomDataset(Dataset):

    def __init__(self, root, train=True, valid=False, transforms=None):
        # root path
        self.root = root
        self.train = train
        self.valid = valid
        self.transform_args = transforms
        
        # image data
        self.imgs = sorted(glob.glob(root+'/*.png'))

        if train or valid: # both train, valid requires txt annotation.
            self.boxes = sorted(glob.glob(root+'/*.txt'))
        

    def parse_boxes(self, box_path):
        with open(box_path, 'r') as file:
            lines = file.readlines()

        boxes = []
        labels = []

        for line in lines:
            values = list(map(float, line.strip().split(' ')))
            class_id = int(values[0])
            x_min, y_min = int(round(values[1])), int(round(values[2]))
            x_max, y_max = int(round(max(values[3], values[5], values[7]))), int(round(max(values[4], values[6], values[8])))

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id)

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx):
        
        img_path = self.imgs[idx]
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)

        height, width = img.shape[0], img.shape[1]
        
        if self.train or self.valid:
            box_path = self.boxes[idx]
            boxes, labels = self.parse_boxes(box_path)
            labels += 1

            transformed = self.transform_args(image=img, bboxes=boxes, labels=labels)
            img, boxes, labels = transformed["image"], transformed["bboxes"], transformed["labels"]

            return img, torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)
        else:
            transformed = self.transform_args(image=img)
            img = transformed["image"]
            file_name = img_path.split('/')[-1]
            return file_name, img, width, height

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    images, targets_boxes, targets_labels = tuple(zip(*batch))
   
    images = torch.stack(images, 0)
    targets = []

    for i in range(len(targets_boxes)):
        target = {
            "boxes": targets_boxes[i],
            "labels": targets_labels[i]
        }
        
        targets.append(target)

    return images, targets

class CustomDatasetV2(Dataset):

    def __init__(self, root, transforms=None):

        self.root = root
        self.transform_args = transforms
        self.imgs = sorted(glob.glob(root+'/*.png'))
        self.boxes = sorted(glob.glob(root+'/*.txt'))

    def parse_boxes(self, box_path):
        with open(box_path, 'r') as file:
            lines = file.readlines()

        boxes = []
        labels = []

        for line in lines:
            values = list(map(float, line.strip().split(' ')))
            class_id = int(values[0])
            x_min, y_min = int(round(values[1])), int(round(values[2]))
            x_max, y_max = int(round(max(values[3], values[5], values[7]))), int(round(max(values[4], values[6], values[8])))

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(class_id)

        return boxes, torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
        height, width = img.shape[0], img.shape[1]
        box_path = self.boxes[idx]
        boxes, labels = self.parse_boxes(box_path)
        labels += 1

        box_index = idx%len(boxes)
        transformed = self.transform_args(image=img, bboxes=boxes, labels=labels, cropping_bbox=[boxes[box_index][0], boxes[box_index][1], boxes[box_index][2], boxes[box_index][3]])
        img, boxes, labels = transformed["image"], transformed["bboxes"], transformed["labels"]
        return img, torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)
    
    def __len__(self):
        return len(self.imgs)

CFG={'IMG_HEIGHT_SIZE' : 800, 'IMG_WIDTH_SIZE' : 1200}

def default_train_transforms():
    return A.Compose([
        A.Resize(CFG['IMG_HEIGHT_SIZE'], CFG['IMG_WIDTH_SIZE']),
        A.augmentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def verticalflip_transforms():
    return A.Compose([
        A.augmentations.geometric.transforms.VerticalFlip(p=0.8),
        A.Resize(CFG['IMG_HEIGHT_SIZE'], CFG['IMG_WIDTH_SIZE']), 
        A.augmentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(), 
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def horizontalflip_transforms():
    return A.Compose([
        A.augmentations.geometric.transforms.HorizontalFlip(p=0.8),
        A.Resize(CFG['IMG_HEIGHT_SIZE'], CFG['IMG_WIDTH_SIZE']), # baseline resize to (512 x 512)    
        A.augmentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(), # albumentations pytorch transforms ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def pixeldrop_transforms():
    return A.Compose([
        A.augmentations.transforms.PixelDropout(dropout_prob=0.7, drop_value=0, p=1.0),
        A.Resize(CFG['IMG_HEIGHT_SIZE'], CFG['IMG_WIDTH_SIZE']),
        A.augmentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def colorjitter_transforms():
    return A.Compose([
        A.augmentations.geometric.transforms.VerticalFlip(p=0.2),
        A.Resize(CFG['IMG_HEIGHT_SIZE'], CFG['IMG_WIDTH_SIZE']),
        A.augmentations.transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5),
                                               saturation=(0.5, 1.5), hue=0.0,
                                               always_apply=False, p=1.0),
        
        A.augmentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def bboxsafe_cropping_transforms():
    return A.Compose([
            A.augmentations.crops.transforms.BBoxSafeRandomCrop(p=1.0),
            A.Resize(CFG['IMG_HEIGHT_SIZE'], CFG['IMG_WIDTH_SIZE']),
            A.augmentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'],
                                    min_area=36000, min_visibility=0.2))

def bboxsafe_cropping_transforms_v2():
    return A.Compose([
            A.augmentations.crops.transforms.BBoxSafeRandomCrop(p=1.0),
            A.augmentations.transforms.PixelDropout(dropout_prob=0.5, drop_value=0, p=0.5),
            A.Resize(CFG['IMG_HEIGHT_SIZE'], CFG['IMG_WIDTH_SIZE']),
            A.augmentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'],
                                    min_area=36000, min_visibility=0.2))


def bboxsafe_cropping_transforms_v3():
    return A.Compose([
            A.augmentations.geometric.rotate.Rotate(limit=50, p=1.0, border_mode=cv2.BORDER_DEFAULT, value=0, crop_border=False, rotate_method='ellipse'),
            A.augmentations.crops.transforms.BBoxSafeRandomCrop(p=1.0),
            A.Resize(CFG['IMG_HEIGHT_SIZE'], CFG['IMG_WIDTH_SIZE']),
            A.augmentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'],
                                    min_area=36000, min_visibility=0.2))


def bboxsafe_cropping_transforms_v4():
    return A.Compose([
            A.augmentations.geometric.transforms.Affine(scale=(1.0, 1.5),
                                                    translate_percent=(-0.1, 0.1),
                                                    rotate=(-30, 30),
                                                    shear=0,
                                                    cval=0,
                                                    fit_output=True,
                                                    keep_ratio=False,
                                                    rotate_method='ellipse',p=1.0),
            A.augmentations.crops.transforms.BBoxSafeRandomCrop(p=1.0),
            A.Resize(CFG['IMG_HEIGHT_SIZE'], CFG['IMG_WIDTH_SIZE']),
            A.augmentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def bbox_nearcrop_transforms():
    return A.Compose([
            A.augmentations.crops.transforms.RandomCropNearBBox(max_part_shift=(0.0, 0.0), p=1.0),
            A.Resize(CFG['IMG_HEIGHT_SIZE'], CFG['IMG_WIDTH_SIZE']),
            A.augmentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
def bbox_nearcrop_transforms_v2():
    return A.Compose([
            A.augmentations.crops.transforms.RandomCropNearBBox(max_part_shift=(0.0, 0.0), p=1.0),
            A.augmentations.dropout.cutout.Cutout(num_holes=4, max_h_size=80, max_w_size=40, p=1.0),
            A.Resize(CFG['IMG_HEIGHT_SIZE'], CFG['IMG_WIDTH_SIZE']),
            A.augmentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def scaling_noise_transforms():
    scale_ratio = 0.25 # for better effect on decreasing quality
    return A.Compose([
        A.augmentations.transforms.Downscale(scale_min=scale_ratio, scale_max=scale_ratio, p=1.0),
        A.Resize(CFG['IMG_HEIGHT_SIZE'], CFG['IMG_WIDTH_SIZE']),
        A.augmentations.transforms.GaussNoise(p=1.0), # for more dirty image quality...
        A.augmentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def rotate_transforms():
    return A.Compose([
        A.augmentations.geometric.rotate.Rotate(limit=70, p=1.0),
        A.Resize(CFG['IMG_HEIGHT_SIZE'], CFG['IMG_WIDTH_SIZE']),
        A.augmentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def perspective_transforms():
    return A.Compose([
        A.augmentations.geometric.transforms.Perspective(scale=(0.10, 0.25), keep_size=True, p=1.0),
        A.Resize(CFG['IMG_HEIGHT_SIZE'], CFG['IMG_WIDTH_SIZE']),
        A.augmentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_valid_transforms():
    return A.Compose([
        A.Resize(CFG['IMG_HEIGHT_SIZE'], CFG['IMG_WIDTH_SIZE']),
        A.augmentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

def get_test_transforms():
    return A.Compose([
        A.Resize(CFG['IMG_HEIGHT_SIZE'], CFG['IMG_WIDTH_SIZE']),
        A.augmentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
from itertools import product
import logging
import random
import pickle
import shutil
import json
import yaml
import csv
import os

'''
File IO
'''

def save_pickle(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def save_json(path, obj, sort_keys=True)-> str:  
    try:
        with open(path, 'w') as f:    
            json.dump(obj, f, indent=4, sort_keys=sort_keys)
        msg = f"Json saved {path}"
    except Exception as e:
        msg = f"Fail to save {e}"
    return msg

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
'''
Logger
'''
def get_logger(name: str, dir_: str, stream=False) -> logging.RootLogger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(dir_, f'{name}.log'))
    
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    if stream:
        logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    
    return logger

import os
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import logging
import csv

class Recorder():
    def __init__(self,
                record_dir:str,
                model: object,
                optimizer: object,
                scheduler: object,
                amp: object,
                logger: logging.RootLogger=None):
        self.record_dir = record_dir
        self.plot_dir = os.path.join(record_dir, 'plots')
        self.record_filepath = os.path.join(self.record_dir, 'record.csv')
        self.weight_path = os.path.join(record_dir, 'model.pt')
        
        self.logger = logger
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.amp = amp
        
        os.makedirs(self.plot_dir, exist_ok = True)
        
    def set_model(self, model: 'model'):
        self.model = model
        
    def set_logger(self, logger: logging.RootLogger):
        self.logger = logger
        
    def create_record_directory(self):
        os.makedirs(self.record_dir, exist_ok=True)
        
        msg = f"Create directory {self.record_dir}"
        self.logger.info(msg) if self.logger else None
        
    def add_row(self, row_dict: dict):
        fieldnames = list(row_dict.keys())
        
        with open(self.record_filepath, newline='', mode='a') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if f.tell() == 0:
                writer.writeheader()
                
            writer.writerow(row_dict)
            msg = f"Write row {row_dict['epoch_index']}"
            self.logger.info(msg) if self.logger else None
            
    def save_weight(self, epoch:int) -> None:
        if self.amp is not None:
            check_point = {
                'epoch': epoch+1,
                'model': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler else None,
                'amp': self.amp.state_dict()
            }
        else:
            check_point = {
                'epoch': epoch + 1,
                'model': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            }
        torch.save(check_point, self.weight_path)
        msg = f"Recorder, epoch {epoch} Model saved: {self.weight_path}"
        self.logger.info(msg) if self.logger else None
        
    def save_plot(self, plots: list):

        record_df = pd.read_csv(self.record_filepath)
        current_epoch = record_df['epoch_index'].max()
        epoch_range = list(range(0, current_epoch+1))
        color_list = ['red', 'blue']  # train, val

        for plot_name in plots:
            columns = [f'train_{plot_name}']

            fig = plt.figure(figsize=(20, 8))
            
            for id_, column in enumerate(columns):
                values = record_df[column].tolist()
                plt.plot(epoch_range, values, marker='.', c=color_list[id_], label=column)
             
            plt.title(plot_name, fontsize=15)
            plt.legend(loc='upper right')
            plt.grid()
            plt.xlabel('epoch')
            plt.ylabel(plot_name)
            plt.xticks(epoch_range, [str(i) for i in epoch_range])
            plt.close(fig)
            fig.savefig(os.path.join(self.plot_dir, plot_name +'.png'))

###############################################################################
###############################################################################

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--lr', type=float, default=0.00015)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--vis_step', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    # parser.add_argument('--gpu_ids', nargs="+", default=['0'])
    parser.add_argument('--gpu_ids', nargs="+", default=['0', '1'])
    # parser.add_argument('--gpu_ids', nargs="+", default=['0', '1', '2'])
    # parser.add_argument('--gpu_ids', nargs="+", default=['0', '1', '2', '3'])
    parser.add_argument('--world_size', type=int, default=0)
    parser.add_argument('--port', type=int, default=2022)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='/home/kwy00/ysy/results/train/20231111_213442/')
    parser.add_argument('--save_file_name', type=str, default='model')
    # usage : --gpu_ids 0, 1,
    return parser

###############################################################################
###############################################################################


def init_for_distributed(rank, opts):

    # 1. setting for distributed training
    opts.rank = rank
    local_gpu_id = int(opts.gpu_ids[opts.rank])
    torch.cuda.set_device(local_gpu_id)
    if opts.rank is not None:
        print("Use GPU: {} for training".format(local_gpu_id))

    # 2. init_process_group
    dist.init_process_group(backend='nccl',
                            init_method='tcp://0.0.0.0:23456',
                            world_size=opts.world_size,
                            rank=opts.rank)

    # if put this function, the all processes block at all.
    torch.distributed.barrier()
    # convert print fn iif rank is zero
    setup_for_distributed(opts.rank == 0)
    print(opts)
    return local_gpu_id


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def main_worker(rank, opts):
    
    local_gpu_id = init_for_distributed(rank, opts)
    # vis = visdom.Visdom(port=opts.port)
    
    
    train_dataset = CustomDatasetV2('/home/kwy00/ysy/train',  transforms=default_train_transforms())
    colorjitter_train_dataset = CustomDatasetV2('/home/kwy00/ysy/train',  transforms=colorjitter_transforms())
    colorjitter_train_dataset2 = CustomDatasetV2('/home/kwy00/ysy/train', transforms=colorjitter_transforms())
    pixeldrop_train_dataset = CustomDatasetV2('/home/kwy00/ysy/train',  transforms=pixeldrop_transforms())
    # horizontalfliped_train_dataset = CustomDatasetV2('/home/kwy00/ysy/train', transforms=horizontalflip_transforms())
    # verticalfliped_train_dataset = CustomDatasetV2('/home/kwy00/ysy/train', transforms=verticalflip_transforms())
    # perspective_train_dataset = CustomDatasetV2('/home/kwy00/ysy/train', transforms=perspective_transforms())
    # scaling_noise_dataset = CustomDatasetV2('/home/kwy00/ysy/train', transforms=scaling_noise_transforms())
    bboxsafe_cropping_train_dataset = CustomDatasetV2('/home/kwy00/ysy/train', transforms=bboxsafe_cropping_transforms_v4())
    bbox_nearcropping_train_dataset = CustomDatasetV2('/home/kwy00/ysy/train', transforms=bbox_nearcrop_transforms_v2())

    valid_dataset = CustomDatasetV2('/home/kwy00/ysy/valid', transforms=default_train_transforms())
    val_colorjitter_train_dataset = CustomDatasetV2('/home/kwy00/ysy/valid', transforms=colorjitter_transforms())
    val_colorjitter_train_dataset2 = CustomDatasetV2('/home/kwy00/ysy/valid', transforms=colorjitter_transforms())
    val_pixeldrop_train_dataset = CustomDatasetV2('/home/kwy00/ysy/valid', transforms=pixeldrop_transforms())
    # val_horizontalfliped_train_dataset = CustomDatasetV2('/home/kwy00/ysy/valid', transforms=horizontalflip_transforms())
    # val_verticalfliped_train_dataset = CustomDatasetV2('/home/kwy00/ysy/valid', transforms=verticalflip_transforms())
    # val_perspective_train_dataset = CustomDatasetV2('/home/kwy00/ysy/valid', transforms=perspective_transforms())
    # val_scaling_noise_dataset = CustomDatasetV2('/home/kwy00/ysy/valid', transforms=scaling_noise_transforms())
    val_bboxsafe_cropping_train_dataset = CustomDatasetV2('/home/kwy00/ysy/valid', transforms=bboxsafe_cropping_transforms_v4())
    val_bbox_nearcropping_train_dataset = CustomDatasetV2('/home/kwy00/ysy/valid', transforms=bbox_nearcrop_transforms_v2())

    # concat dataset with train and valid -> only train used
    total_dataset = ConcatDataset([train_dataset,
                                colorjitter_train_dataset,
                                colorjitter_train_dataset2,
                                pixeldrop_train_dataset,
                                # horizontalfliped_train_dataset,
                                #  verticalfliped_train_dataset,
                                #    perspective_train_dataset,
                                #  scaling_noise_dataset,
                                bboxsafe_cropping_train_dataset,
                                bbox_nearcropping_train_dataset,
                                 valid_dataset,
                                 val_colorjitter_train_dataset,
                                 val_colorjitter_train_dataset2,
                                 val_pixeldrop_train_dataset,
                                #  val_horizontalfliped_train_dataset,
                                #  val_verticalfliped_train_dataset,
                                #    val_perspective_train_dataset,
                                #  val_scaling_noise_dataset,
                                 val_bboxsafe_cropping_train_dataset,
                                 val_bbox_nearcropping_train_dataset
                                ])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    sampler_train = DistributedSampler(total_dataset, shuffle=False)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, opts.batch_size, drop_last = True)
    
    train_loader = DataLoader(total_dataset, batch_sampler=batch_sampler_train,
                            num_workers=opts.num_workers, pin_memory=True, 
                            collate_fn=collate_fn)
    
    model = torchvision.models.detection.retinanet_resnet50_fpn_v2(pretrained=True, pretrained_backbone=True)
    model = model.cuda(local_gpu_id)
    model = DDP(module=model, device_ids=[local_gpu_id])
    
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts.lr,
                                    weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    from datetime import datetime, timezone, timedelta

    # root directory
    ROOT_DIR = '/home/kwy00/ysy/'

    # define train serial
    kst = timezone(timedelta(hours=9))
    train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

    # recorder directory
    RECORDER_DIR = os.path.join(ROOT_DIR, 'results', 'train', train_serial)
    os.makedirs(RECORDER_DIR, exist_ok=True)
    
    
    logger = get_logger(name='train', dir_=RECORDER_DIR, stream=False)
    logger.info(f"Set Logger {RECORDER_DIR}")
    recorder = Recorder(record_dir=RECORDER_DIR,
                    model = model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    amp= None,
                    logger=logger)
    

    interval=100
    EPOCHS = opts.epoch
    
    if opts.start_epoch != 0:
        checkpoint = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.pt',
                                map_location=torch.device('cuda:{}'.format(local_gpu_id)))
        model.load_state_dict(checkpoint['model'])  # load model state dict
        optimizer.load_state_dict(checkpoint['optimizer'])  # load optim state dict
        scheduler.load_state_dict(checkpoint['scheduler'])  # load sched state dict
        if opts.rank == 0:
            print('\nLoaded checkpoint from epoch %d.\n' % (int(opts.start_epoch) - 1))
    
    for epoch in range(opts.start_epoch, opts.epoch):
        
        model.train()
        train_sampler.set_epoch(epoch)

        print("current lr: ", optimizer.param_groups[0]['lr'])
        
        tic = time()

        if opts.rank==0:
            row_dict = dict()
            row_dict['epoch_index'] = epoch
            row_dict['train_serial'] = train_serial
            start_timestamp = time()
            train_loss = []
        """
        Train start
        """
        print(f"Train {epoch}/{EPOCHS}")
        logger.info(f"--Train {epoch}/{EPOCHS}")

        # data batch processing
        for batch_index, (images, targets) in enumerate(tqdm(train_loader)):

            images = [img.to(local_gpu_id) for img in images]
            targets = [{k: v.to(local_gpu_id) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
                
            toc = time()
            
            if opts.rank==0:
                train_loss.append(losses.item())

                if batch_index % interval == 0:
                    msg = f"batch: {batch_index}/{len(train_loader)} loss: {losses.item()}"
                    logger.info(msg)
                    
                if (batch_index%opts.vis_step==0 or batch_index==len(train_loader)-1):
                    print('Epoch [{0}/{1}], Iter [{2}/{3}], Loss: {4:.4f}, LR: {5:.8f}, Time: {6:.2f}'.format(epoch,
                                                                                                            opts.epoch, batch_index,
                                                                                                            len(train_loader),
                                                                                                            losses.item(), lr, 
                                                                                                            toc - tic))
        if opts.rank==0:
            tr_loss = np.mean(train_loss)
            end_timestamp = time()
            row_dict['train_loss'] = tr_loss
            row_dict['train_elapsed_time'] = end_timestamp - start_timestamp
            recorder.add_row(row_dict)
            recorder.save_plot(['loss', 'elapsed_time'])
            
            if epoch%4 == 0:
                recorder.save_weight(epoch=epoch)
                
            print(f'Epoch [{epoch}] Train loss : [{tr_loss:.5f}]') 

        scheduler.step()

    if opts.rank==0:
        recorder.save_weight(epoch=epoch)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CarDetection torchivision model training', parents=[get_args_parser()])
    opts = parser.parse_args()
    
    opts.world_size = len(opts.gpu_ids)
    opts.num_workers = len(opts.gpu_ids)*4
    ngpus_per_node = torch.cuda.device_count()
    
    mp.spawn(main_worker,nprocs=opts.world_size, args=(opts, ), join=True)

    
