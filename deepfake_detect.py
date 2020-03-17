import os
import argparse
import json
import cv2
from PIL import Image
import numpy as np
import glob
from tqdm import tqdm
import pickle as pkl
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorchcv.model_provider import get_model

from sklearn.model_selection import train_test_split
import sklearn.metrics

from dataprepare import DeepFakeFrame, DeepFakeFrame2
from network import Xception, XceptionFusion

import albumentations
from albumentations.augmentations.transforms import ShiftScaleRotate, HorizontalFlip, Normalize, RandomBrightnessContrast, \
    MotionBlur, Blur, GaussNoise, JpegCompression


def criterion_1(pred1, targets):
    l1 = F.binary_cross_entropy(torch.sigmoid(pred1), targets.float())
    return l1


def criterion_2(pred, targets, gamma=5, reduced=False):
    '''
    Focal Loss
    :param pred: predicted probability
    :param targets: labels
    :return: focal loss
    '''
    if reduced:
        l = -torch.sum(targets*(1-pred)**gamma*torch.log(pred) + (1-targets)*(pred)**gamma*torch.log(1-pred))
    else:
        l = -(targets*(1-pred)**gamma*torch.log(pred) + (1-targets)*(pred)**gamma*torch.log(1-pred))
    return l


def train(epoch, model, optimizer, criterion, trainloader, scheduler, device, args):
    total_loss = 0

    tq = tqdm(trainloader, ascii=True, ncols=50)
    for i, (images, labels, _) in enumerate(tq):

        if not (images.shape[0] == args.batch_size):
            continue

        images = np.transpose(images, [0, 3, 1, 2])
        images, labels = images.to(device), labels.to(device).float()
        outputs = model(images)

        # loss = criterion_1(outputs.reshape(-1), labels)
        loss = criterion(torch.sigmoid(outputs).reshape(-1), labels.float())
        # weight = 1 - 0.5*labels
        weight = 1
        loss = torch.mean(weight*loss)
        total_loss += loss

        tq.set_description(f'Epoch {epoch + 1}/{args.n_epochs}, LR: %6f, Loss: %.4f'%(
            optimizer.state_dict()['param_groups'][0]['lr'], total_loss/(i+1)))

        # print("Epoch : {}/{} | LR: {} | Loss: {}".format(epoch+1, args.n_epochs,
        #                                                  optimizer.state_dict()['param_groups'][0]['lr'], total_loss / (i + 1)))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()


def validating(epoch, model, optimizer, criterion, valloader, scheduler, device, args, best_epoch, best_acc, best_weights):
    total_loss = 0
    loss = 0
    pred = []
    real = []
    model.eval()
    date_info = datetime.datetime.now()

    with torch.no_grad():
        for i, (images, labels, _) in enumerate(tqdm(valloader, ascii=True, ncols=50)):
            if not (images.shape[0] == args.batch_size):
                continue

            images = np.transpose(images, [0, 3, 1, 2])
            images, labels = images.to(device), labels.to(device).float()
            outputs = model(images)

            # loss = criterion_1(outputs.reshape(-1), labels)
            loss = criterion(torch.sigmoid(outputs).reshape(-1), labels.float())
            # weight = 0.1 ** (1 - labels)
            loss = torch.sum(loss)
            total_loss += loss

            for out in outputs:
                pred.append(torch.sigmoid(out))
            for lab in labels:
                real.append(lab.data.cpu())

        # import matplotlib.pyplot as plt
        # plt.imshow(np.transpose(images[0].squeeze().detach().cpu().numpy(), [1, 2, 0]))

        if scheduler is not None:
            scheduler.step(total_loss / (i + 1))

        print("Epoch : {}/{} | LR: {} | Loss: {}".format(epoch + 1, args.n_epochs,
                                                         optimizer.state_dict()['param_groups'][0]['lr'],
                                                         total_loss / (i + 1)))

        pred = [p.data.cpu().numpy() for p in pred]
        pred2 = pred
        pred3 = np.array(pred.copy())
        pred = [np.round(p) for p in pred]
        pred = np.array(pred)

        acc = sklearn.metrics.recall_score(real, pred, average='macro')

        real = [r.item() for r in real]
        pred2 = np.array(pred2).clip(0.1, 0.9)
        pred3[pred3 <= 0.5] = 0.2
        pred3[pred3 >= 0.5] = 0.8

        kaggle = sklearn.metrics.log_loss(real, pred2)
        kaggle2 = sklearn.metrics.log_loss(real, pred3)

        loss /= len(valloader)

        print(f'Dev loss: %.4f, Acc: %.6f, Kaggle: %.6f, Kaggle(Clip): %.6f' % (loss, acc, kaggle, kaggle2))

        if acc > best_acc:
            best_epoch = epoch
            best_acc = acc
            best_kaggle = kaggle
            best_weights = model.state_dict()
            torch.save(best_weights, "(40percentdata)_BCE_best_model_"+str(date_info)+ ".pkl")

        return acc, best_epoch, best_acc, best_weights


def main(args):

    opt_gpus = [i for i in range(args.gpu, args.gpu+int(args.n_gpus))]
    if len(opt_gpus) > 1:
        print("Using ", len(opt_gpus), " GPUs")
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in opt_gpus)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")
    print(device)

    train_transform = albumentations.Compose([
                                              ShiftScaleRotate(p=0.3, scale_limit=0.25, border_mode=1, rotate_limit=25),
                                              HorizontalFlip(p=0.2),
                                              RandomBrightnessContrast(p=0.3, brightness_limit=0.25, contrast_limit=0.5),
                                              MotionBlur(p=.2),
                                              GaussNoise(p=.2),
                                              Normalize()
    ])
    val_transform = albumentations.Compose([Normalize()])

    # trainset = DeepFakeFrame2("/data/songzhu/deepfake", split='training', train_val_ratio=0.9, transform=train_transform)
    # valset = DeepFakeFrame2("/data/songzhu/deepfake", split='validating', train_val_ratio=0.9, transform=val_transform)

    trainset = DeepFakeFrame("/data/songzhu/deepfake", split='training', train_val_ratio=0.9,
                              transform=train_transform)
    valset = DeepFakeFrame("/data/songzhu/deepfake", split='validating', train_val_ratio=0.9, transform=val_transform)

    print(np.unique(trainset.labels, return_counts=True))

    best = 1e10
    batch_size = args.batch_size

    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False, num_workers=1)
    valloader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=False, num_workers=1)

    net = Xception(in_f=2048)
    # net = XceptionFusion(inf_1=2048, inf_2=2048)
    net = nn.DataParallel(net)
    # net.load_state_dict(torch.load("best_model.pkl"))
    net = net.to(device)
    best_weights = net.state_dict()

    criterion_2 = nn.BCELoss(reduce='none')
    # optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=5e-4, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, mode='min', factor=0.5, verbose=True, min_lr=1e-4)

    # =========================  Training ===================================
    best_epoch = 0
    best_kaggle = np.Inf
    best_acc = 0

    for epoch in range(args.n_epochs):

        train(epoch, net, optimizer, criterion_2, trainloader, None, device, args)
        acc, best_epoch, best_acc, best_weights = validating(epoch, net, optimizer, criterion_2, valloader, scheduler, device, args,
                                             best_epoch=best_epoch, best_acc=best_acc, best_weights=best_weights)
        if acc > best_acc:
            net.load_state_dict(best_weights)

    # ==========================  Testing ====================================
    # validating(epoch, model, testloader, device, args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=2, help='delimited list input of GPUs', type=int)
    parser.add_argument('--n_gpus', default=2, help="num of GPUS to use", type=int)
    parser.add_argument('--n_epochs', default=20, help="num of epochs to train", type=int)
    parser.add_argument('--batch_size', default=128, help="batch_size", type=int)
    args = parser.parse_args()

    opt_gpus = [i for i in range(args.gpu, (args.gpu + args.n_gpus))]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in opt_gpus)


    main(args)
