#####################
# IMPORT LIBRAIRIES #
#####################

import os
import glob
import cv2
import pandas as pd
import numpy as np
import torch
from random import shuffle

from config import cfg
from utils.dataloader import WMHDataset
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import KFold

from utils.cactus import CACTUS
from utils.utils import EpochCallback
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import BinaryRecall, BinaryPrecision, BinaryMatthewsCorrCoef

import datetime
import time
from tqdm import tqdm


#####################
#   CONFIGURATION   #
#####################


# get the start datetime
st = datetime.datetime.now()
t = time.time()
dt = datetime.datetime.now()
d = f'{dt.year}-{dt.month}-{dt.day}_{dt.hour}-{dt.minute}'

data_dir = cfg.data_dir

train_transforms=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
 
val_transforms=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

device = 'cpu:0'
if torch.cuda.is_available():
    device = 'cuda:0'
    print('[INFO] Using GPU :', torch.cuda.get_device_name(0), '\n')
    
kfold = KFold(n_splits=cfg.kfold, shuffle=True, random_state=42)

cv_scores = {
    'kfold' : [],
    'avg_train_loss': [],
    'avg_train_acc': [],
    'avg_train_recall': [],
    'avg_train_precision': [],
    'avg_train_MCC': [],
    'avg_val_loss': [],
    'avg_val_acc': [],
    'avg_val_recall': [],
    'avg_val_precision': [],
    'avg_val_MCC': []
}


########################
#   CROSS-VALIDATION   #
########################


print('[INFO] Starting cross-validation...\n')

PATLIST = cfg.train_patlist

for k, (train_idx, val_idx) in enumerate(kfold.split(PATLIST)):
    
    config = f'config{k}'

    VAL_PATLIST = val_idx
    TRAIN_PATLIST = train_idx
    shuffle(TRAIN_PATLIST)

    print('\n[INFO]', config)
    print(f'\n{len(TRAIN_PATLIST)} patients dans train : {TRAIN_PATLIST}')
    print(f'{len(VAL_PATLIST)} patients dans valid : {VAL_PATLIST}\n')

    kfold_train_loss =  []
    kfold_train_acc = []
    kfold_train_recall = []
    kfold_train_precision = []
    kfold_train_MCC = []

    kfold_val_loss =  []
    kfold_val_acc = []
    kfold_val_recall = []
    kfold_val_precision = []
    kfold_val_MCC = []

    #-----------------
    #  Loading Data
    #-----------------
    
    # add paths to collect train images
    train_list = []
    for patient in TRAIN_PATLIST:
        for img in glob.glob(data_dir+f'/*/Patient{patient}/*.npy',recursive=True):
            train_list.append(img)

    # add paths to collect valid images
    val_list = []
    for patient in VAL_PATLIST:
        for img in glob.glob(data_dir+f'/*/Patient{patient}/*.npy',recursive=True):
            val_list.append(img)

    print('--------------------------------------------------------------\n')
    print(f'[KFOLD {config}] train : {len(train_list)} examples, valid : {len(val_list)} examples')

    train_dataset = WMHDataset(data_dir, train_list, train_transforms)
    val_dataset = WMHDataset(data_dir, val_list, val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=True)


    #-----------------
    #  Loading Model
    #-----------------

    model = CACTUS(
            image_size = cfg.cactus.image_size,
            channels = cfg.cactus.channels,
            num_classes = cfg.cactus.num_classes,
            patch_size_axial = cfg.cactus.patch_size_axial, 
            patch_size_coronal = cfg.cactus.patch_size_coronal,
            axial_dim = cfg.cactus.axial_dim,
            coronal_dim = cfg.cactus.coronal_dim, 
            axial_depth = cfg.cactus.axial_depth, 
            coronal_depth = cfg.cactus.coronal_depth,
            cross_attn_depth = cfg.cactus.cross_attn_depth,
            multi_scale_enc_depth = cfg.cactus.multi_scale_enc_depth,
            heads = cfg.cactus.heads
            )

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('[INFO] Trainable Parameters: %.3fM' % parameters)

    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    def binary_acc(y_pred, y_test):
        y_pred_tag = torch.log_softmax(y_pred, dim = 1)
        _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
        correct_results_sum = (y_pred_tags == y_test).sum().float()
        acc = correct_results_sum/y_test.shape[0]
        acc = torch.round(acc * 100)
        return acc

    binary_rec = BinaryRecall()
    binary_prec = BinaryPrecision()
    binary_MCC = BinaryMatthewsCorrCoef()

    stats = {
        'train_loss': [],
        'train_acc': [],
        'train_recall': [],
        'train_precision': [],
        'train_MCC': [],
        'val_loss': [],
        'val_acc': [],
        'val_recall': [],
        'val_precision': [],
        'val_MCC': []
    }

    n_epochs = cfg.epoch

    model_path = os.path.join(cfg.model_path, f'{d}/{config}/')
    if not os.path.isdir(os.path.join(cfg.model_path, f'{d}/{config}/')):
        os.makedirs(os.path.join(cfg.model_path, f'{d}/{config}/'))

    model_name = f'{model_path}cactus_{config}.pth'

    callback = EpochCallback(model_name, n_epochs,
                            model, optimizer, 'val_loss', cfg.patience)

    print(f'[INFO] Starting training...')
    for epoch in range(n_epochs):
        epoch_loss = 0
        epoch_acc = 0
        epoch_recall = 0
        epoch_precision = 0
        epoch_MCC = 0

        #-------------
        #  Training
        #-------------
        
        # iterate over batches
        for i ,data in tqdm(enumerate(train_loader), total = len(train_loader)):
            x_batch , y_batch = data
            x_batch = x_batch.to(device) # move to gpu
            y_batch = y_batch
            y_batch = y_batch.to(device) # move to gpu

            model.train()

            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            train_acc = binary_acc(y_pred, y_batch)

            y_pred_tag = torch.log_softmax(y_pred, dim = 1)
            _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
            train_recall = binary_rec(y_pred_tags, y_batch)
            train_precision = binary_prec(y_pred_tags, y_batch)
            train_MCC = binary_MCC(y_pred_tags, y_batch)
            
            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += train_acc.item()
            epoch_recall += train_recall.item()
            epoch_precision += train_precision.item()
            epoch_MCC += train_MCC.item()
        
        stats['train_loss'].append(epoch_loss/(len(train_loader)))
        stats['train_acc'].append(epoch_acc/(len(train_loader)))
        stats['train_recall'].append(epoch_recall/(len(train_loader)))
        stats['train_precision'].append(epoch_precision/(len(train_loader)))
        stats['train_MCC'].append(epoch_MCC/(len(train_loader)))

        # print('\nEpoch : {}, train loss : {}, train acc : {}'.format(epoch+1,epoch_loss/(len(train_loader)), epoch_acc/(len(train_loader))))

        #--------------
        #  Validation
        #--------------

        # Validation doesnt requires gradient
        with torch.no_grad():
            epoch_val_loss = 0
            epoch_val_acc = 0
            epoch_val_recall = 0
            epoch_val_precision = 0
            epoch_val_MCC = 0

            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch
                y_batch = y_batch.to(device)

                # model to eval mode
                model.eval()

                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                val_acc = binary_acc(y_pred, y_batch)

                y_pred_tag = torch.log_softmax(y_pred, dim = 1)
                _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
                val_recall = binary_rec(y_pred_tags, y_batch)
                val_precision = binary_prec(y_pred_tags, y_batch)
                val_MCC = binary_MCC(y_pred_tags, y_batch)

                epoch_val_loss += loss.item()
                epoch_val_acc += val_acc.item()
                epoch_val_recall += val_recall.item()
                epoch_val_precision += val_precision.item()
                epoch_val_MCC += val_MCC.item()

        stats['val_loss'].append(epoch_val_loss/(len(val_loader)))
        stats['val_acc'].append(epoch_val_acc/(len(val_loader)))
        stats['val_recall'].append(epoch_val_recall/(len(val_loader)))
        stats['val_precision'].append(epoch_val_precision/(len(val_loader)))
        stats['val_MCC'].append(epoch_val_MCC/(len(val_loader)))
        # print('\nEpoch : {}, val loss : {}, val acc : {}'.format(epoch+1,val_loss/(len(val_loader)), val_acc/(len(val_loader))))

        scheduler.step(epoch_val_loss)

        callback.epoch_end(epoch + 1,
                                {'loss': epoch_loss / len(train_loader),
                                'recall': epoch_recall / len(train_loader),
                                'precision': epoch_precision / len(train_loader),
                                'val_loss': epoch_val_loss / len(val_loader),
                                'val_recall': epoch_val_recall / len(val_loader),
                                'val_precision': epoch_val_precision / len(val_loader)})
        
        kfold_train_loss.append(epoch_loss / len(train_loader))
        kfold_train_acc.append(epoch_acc / len(train_loader))
        kfold_train_recall.append(epoch_recall / len(train_loader))
        kfold_train_precision.append(epoch_precision / len(train_loader))
        kfold_train_MCC.append(epoch_MCC / len(train_loader))

        kfold_val_loss.append(epoch_val_loss / len(val_loader))
        kfold_val_acc.append(epoch_val_acc / len(val_loader))
        kfold_val_recall.append(epoch_val_recall / len(val_loader))
        kfold_val_precision.append(epoch_val_precision / len(val_loader))
        kfold_val_MCC.append(epoch_val_MCC / len(val_loader))

        if callback.end_training:

            break
    
    pd.DataFrame.from_dict(stats).to_csv(f'{model_path}/stats_{config}.csv') 
 
    cv_scores['kfold'].append(config)
    cv_scores['avg_train_acc'].append(np.median(kfold_train_acc))
    cv_scores['avg_train_loss'].append(np.median(kfold_train_loss))
    cv_scores['avg_train_recall'].append(np.median(kfold_train_recall))
    cv_scores['avg_train_precision'].append(np.median(kfold_train_precision))
    cv_scores['avg_train_MCC'].append(np.median(kfold_train_MCC))
    cv_scores['avg_val_acc'].append(np.median(kfold_val_acc))
    cv_scores['avg_val_loss'].append(np.median(kfold_val_loss))
    cv_scores['avg_val_recall'].append(np.median(kfold_val_recall))
    cv_scores['avg_val_precision'].append(np.median(kfold_val_precision))
    cv_scores['avg_val_MCC'].append(np.median(kfold_val_MCC))

    print(f'\n[KFOLD {config}]')
    print(f'TRAIN : recall = {np.round(np.median(kfold_train_recall), 2)}, precision = {np.round(np.median(kfold_train_precision), 2)}, MCC = {np.round(np.median(kfold_train_MCC), 2)}')
    print(f'VALID : recall = {np.round(np.median(kfold_val_recall), 2)}, precision = {np.round(np.median(kfold_val_precision), 2)}, MCC = {np.round(np.median(kfold_val_MCC), 2)}')
    print('\n--------------------------------------------------------------')

pd.DataFrame.from_dict(cv_scores).to_csv(os.path.join(cfg.model_path, f'{d}/crossval_global_stats.csv'), index=False) 
    
print('\n[INFO] End of cross-validation !')

# get the end datetime
et = datetime.datetime.now()

def convert_timedelta(duration):
    days, seconds = duration.days, duration.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 60)
    return hours, minutes, seconds

# get execution time
elapsed_time = et - st
hours, minutes, seconds = convert_timedelta(elapsed_time)
print('Execution time:', hours, 'hours', minutes, 'minutes and', seconds, 'seconds.')