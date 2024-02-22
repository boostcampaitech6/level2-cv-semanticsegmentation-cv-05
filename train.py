# python native
import os
import json
import random
import datetime
from functools import partial

# external library
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
import albumentations as A

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models

import argparse
import wandb

# visualization
from dataset import *
# model
from model import DeepLabV3p

#smp 사용시
import segmentation_models_pytorch as smp

###ARGPARSE###
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", default=4, type=int) 
    parser.add_argument("-lr", "--lr", dest="lr", default=1e-4, type=float)   
    parser.add_argument("--name", type=str, default="exp")   
    parser.add_argument("-s", "--seed", dest="seed", default=5252, type=int)   
    parser.add_argument("-e", "--epoch", dest="epoch", default=200, type=int) 
    parser.add_argument("-v", "--val_every", dest="val_every", default=20, type=int)     
    parser.add_argument("-dir", "--saved_dir", dest="saved_dir", default="/data/ephemeral/home/segmentation/model", type=str)    
    parser.add_argument("-name", "--model_name", dest="model_name", default="val2_fcn_aug_model.pt", type=str)    
    parser.add_argument("-w", "--weight_decay", dest="weight_decay", default=1e-6, type=float)   
    parser.add_argument("-l", "--loss", default='bce_loss', type=str) 
    parser.add_argument("-wb", "--entity", default= 'exp1', type=str)

    args = parser.parse_args()
    return args

args = parse_args()

CLASSES = [
    'finger-1', 'finger-2', 'finger-3', 'finger-4', 'finger-5',
    'finger-6', 'finger-7', 'finger-8', 'finger-9', 'finger-10',
    'finger-11', 'finger-12', 'finger-13', 'finger-14', 'finger-15',
    'finger-16', 'finger-17', 'finger-18', 'finger-19', 'Trapezium',
    'Trapezoid', 'Capitate', 'Hamate', 'Scaphoid', 'Lunate',
    'Triquetrum', 'Pisiform', 'Radius', 'Ulna',
]

if not os.path.exists(args.saved_dir):                                                           
    os.makedirs(args.saved_dir)

# Initialize WandB
wandb.init(
        name=args.name,  # exp_name
        project="segmentation",
        config=args,
    )

#Augmentation 추가 
tf = A.Compose([A.Resize(1024, 1024), 
                A.HorizontalFlip(p=0.8)
                ])
train_dataset = XRayDataset(is_train=True, transforms=tf)

#validation dataset
tf = A.Resize(1024, 1024)
valid_dataset = XRayDataset(is_train=False, transforms=tf)

#train Dataloader
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,
    drop_last=True,
)

#valid Dataloader 주의 : num_workers 커지지 않도록 
valid_loader = DataLoader(
    dataset=valid_dataset, 
    batch_size=1,
    shuffle=False,
    num_workers=2,
    drop_last=False
)

#Define Functions for Training - loss function으로 빼기 
def dice_coef(y_true, y_pred):
    y_true_f = y_true.flatten(2)
    y_pred_f = y_pred.flatten(2)
    intersection = torch.sum(y_true_f * y_pred_f, -1)
    
    eps = 0.0001
    return (2. * intersection + eps) / (torch.sum(y_true_f, -1) + torch.sum(y_pred_f, -1) + eps)

def save_model(model, file_name=args.model_name):
    output_path = os.path.join(args.saved_dir, file_name)
    torch.save(model, output_path)

def set_seed():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

def validation(epoch, model, data_loader, criterion, thr=0.5):
    print(f'Start validation #{epoch:2d}')
    model.eval()

    dices = []
    with torch.no_grad():
        n_class = len(CLASSES)
        total_loss = 0
        cnt = 0

        for step, (images, masks) in tqdm(enumerate(data_loader), total=len(data_loader)):
            images, masks = images.cuda(), masks.cuda()         
            model = model.cuda()
            
            outputs = model(images)
            
            output_h, output_w = outputs.size(-2), outputs.size(-1)
            mask_h, mask_w = masks.size(-2), masks.size(-1)
            
            # gt와 prediction의 크기가 다른 경우 prediction을 gt에 맞춰 interpolation 합니다.
            if output_h != mask_h or output_w != mask_w:
                outputs = F.interpolate(outputs, size=(mask_h, mask_w), mode="bilinear")
            
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > thr).detach().cpu()
            masks = masks.detach().cpu()
            
            dice = dice_coef(outputs, masks)
            dices.append(dice)
                
    dices = torch.cat(dices, 0)
    dices_per_class = torch.mean(dices, 0)
    dice_str = [
        f"{c:<12}: {d.item():.4f}"
        for c, d in zip(CLASSES, dices_per_class)
    ]
    dice_str = "\n".join(dice_str)
    print(dice_str)
    
    avg_dice = torch.mean(dices_per_class).item()
    
    return avg_dice

#dice loss 추가 
def dice_loss(pred, target, smooth = 1.):
    pred = pred.contiguous()
    target = target.contiguous()   
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

#combined loss
def calc_loss(pred, target, bce_weight = 0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)
    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss

def train(model, data_loader, val_loader, criterion, optimizer):
    print(f'Start training..')
    
    n_class = len(CLASSES)
    best_dice = 0.
    
    for epoch in range(args.epoch):
        model.train()

        for step, (images, masks) in enumerate(data_loader):            
            # gpu 연산을 위해 device 할당합니다.
            images, masks = images.cuda(), masks.cuda()
            model = model.cuda()
            
            outputs = model(images)
            
            # loss를 계산합니다.
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({'train_loss': loss.item()})
            
            # step 주기에 따라 loss를 출력합니다.
            if (step + 1) % 25 == 0:
                print(
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | '
                    f'Epoch [{epoch+1}/{args.epoch}], '
                    f'Step [{step+1}/{len(train_loader)}], '
                    f'Loss: {round(loss.item(),4)}'
                )
             
        # validation 주기에 따라 loss를 출력하고 best model을 저장합니다.
        if (epoch + 1) % args.val_every == 0:
            dice = validation(epoch + 1, model, val_loader, criterion)
            wandb.log({'val_loss': dice})
            
            if best_dice < dice:
                print(f"Best performance at epoch: {epoch + 1}, {best_dice:.4f} -> {dice:.4f}")
                print(f"Save model in {args.saved_dir}")
                best_dice = dice
                save_model(model)

##TRAINING##
#Custom model
#model = models.segmentation.fcn_resnet50(pretrained=True)
           
#smp model 

model = smp.FPN(
     encoder_name="resnet34", # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
     classes=29,                     # model output channels (number of classes in your dataset)
 )

wandb.watch(model)

# Loss function
criterion = calc_loss

# Optimizer
optimizer = optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# seed
set_seed()

# train
train(model, train_loader, valid_loader, criterion, optimizer)

if __name__ == '__main__':
    args = parse_args()