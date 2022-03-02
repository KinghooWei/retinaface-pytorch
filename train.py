import os
import datetime
import sys
from math import ceil

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.retinaface import RetinaFace
from nets.retinaface_training import MultiBoxLoss, weights_init
from utils.anchors import Anchors
from utils.callbacks import LossHistory
from utils.config import cfg_mnet, cfg_re50
from utils.dataloader import DataGenerator, detection_collate
from utils.utils_fit import fit_one_epoch, evaluate

if __name__ == "__main__":
    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    Cuda = True
    # --------------------------------#
    #   获得训练用的人脸标签与坐标
    # --------------------------------#
    train_label_dir = './data/widerface/train/label.txt'
    val_label_dir = './data/widerface/val/label.txt'

    curr_time = datetime.datetime.now()
    save_path = os.path.join('./logs/', datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S'))
    # -------------------------------#
    #   主干特征提取网络的选择
    #   mobilenet或者resnet50
    # -------------------------------#
    backbone = "mobilenet"
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #   如果不设置model_path，pretrained = False，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    # --------------------------------------------------------------------------------------------------------------------------
    pretrained = False
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   权值文件的下载请看README，可以通过网盘下载。
    #   模型的 预训练权重 比较重要的部分是 主干特征提取网络的权值部分，用于进行特征提取。
    #
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的 冻结阶段 或者 解冻阶段 的参数，来保证模型epoch的连续性。
    #   
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的，pretrain不影响此处的权值加载。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，pretrain = True，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，pretrain = Fasle，Freeze_Train = Fasle，此时从0开始训练，且没有冻结主干的过程。
    #   一般来讲，从0开始训练效果会很差，因为权值太过随机，特征提取效果不明显。
    #
    #   网络一般不从0开始训练，至少会使用主干部分的权值，有些论文提到可以不用预训练，主要原因是他们 数据集较大 且 调参能力优秀。
    #   如果一定要训练网络的主干部分，可以了解imagenet数据集，首先训练分类模型，分类模型的 主干部分 和该模型通用，基于此进行训练。
    # ----------------------------------------------------------------------------------------------------------------------------#
    # model_path = "model_data/Retinaface_mobilenet0.25.pth"
    model_path = 'logs/2022_02_28_09_58_03/Epoch90-Loss4.9664.pth'
    # -------------------------------------------------------------------#
    #   是否进行冻结训练，默认先冻结主干训练后解冻训练。
    # -------------------------------------------------------------------#
    Freeze_Train = True
    # -------------------------------------------------------------------#
    #   用于设置是否使用多线程读取数据，0代表关闭多线程
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    # -------------------------------------------------------------------#
    num_workers = 8

    if backbone == "mobilenet":
        cfg = cfg_mnet
    elif backbone == "resnet50":
        cfg = cfg_re50
    else:
        raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))

    model = RetinaFace(cfg=cfg, pretrained=pretrained)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        # ------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # -------------------------------#
    #   获得先验框anchors
    # -------------------------------#
    anchors = Anchors(cfg, image_size=(cfg['train_image_size'], cfg['train_image_size'])).get_anchors()

    if Cuda:
        model_cuda = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_cuda = model_cuda.cuda()
        anchors = anchors.cuda()

    criterion = MultiBoxLoss(3, 0.35, 7, cfg['variance'], Cuda)

    train_batch_size = 32
    train_dataset = DataGenerator(train_label_dir, cfg['train_image_size'])
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size, num_workers=num_workers,
                              pin_memory=True, collate_fn=detection_collate)
    train_epoch_step = ceil(train_dataset.get_len() / train_batch_size)

    val_batch_size = 16
    val_dataset = DataGenerator(val_label_dir, cfg['train_image_size'], 'val')
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, num_workers=num_workers, pin_memory=True,
                            collate_fn=detection_collate)
    val_epoch_step = ceil(val_dataset.get_len() / val_batch_size)

    train_loss_history = LossHistory(save_path, 'train')
    val_loss_history = LossHistory(save_path, 'val')
    # ---------------------------------------------------------#
    #   训练分为两个阶段，分别是冻结阶段和解冻阶段。
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #   受到BatchNorm层影响，batch_size最小为2，不能为1。
    # ---------------------------------------------------------#
    # ---------------------------------------------------------#
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    # ---------------------------------------------------------#
    if False:
        # ----------------------------------------------------#
        #   冻结阶段训练参数
        #   此时模型的主干被冻结了，特征提取网络不发生改变
        #   占用的显存较小，仅对网络进行微调
        # ----------------------------------------------------#
        lr = 1e-3

        Init_Epoch = 0
        Freeze_Epoch = 40

        optimizer = optim.Adam(model_cuda.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        if Freeze_Train:
            for param in model.body.parameters():
                param.requires_grad = False

        min_train_loss = sys.float_info.max
        min_val_loss = sys.float_info.max
        for epoch in range(Init_Epoch, Freeze_Epoch):
            min_train_loss = fit_one_epoch(model_cuda, model, train_loss_history, optimizer, criterion, epoch, train_epoch_step, train_loader,
                                           Freeze_Epoch, anchors, cfg, Cuda, min_train_loss)
            lr_scheduler.step()
            min_val_loss = evaluate(model_cuda, model, val_loss_history, criterion, epoch, val_epoch_step, val_loader, Freeze_Epoch,
                     anchors, cfg, Cuda, min_val_loss)

    if True:
        # ----------------------------------------------------#
        #   解冻阶段训练参数
        #   此时模型的主干不被冻结了，特征提取网络会发生改变
        #   占用的显存较大，网络所有的参数都会发生改变
        # ----------------------------------------------------#
        # lr = 1e-4
        lr = 0.00000154
        # Batch_size      = 64
        Freeze_Epoch = 90
        Unfreeze_Epoch = 120

        optimizer = optim.Adam(model_cuda.parameters(), lr, weight_decay=5e-4)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)

        # ------------------------------------#
        #   解冻后训练
        # ------------------------------------#
        if Freeze_Train:
            for param in model.body.parameters():
                param.requires_grad = True

        min_train_loss = sys.float_info.max
        min_val_loss = sys.float_info.max
        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            min_train_loss = fit_one_epoch(model_cuda, model, train_loss_history, optimizer, criterion, epoch, train_epoch_step, train_loader,
                                           Unfreeze_Epoch, anchors, cfg, Cuda, min_train_loss)
            lr_scheduler.step()
            min_val_loss = evaluate(model_cuda, model, val_loss_history, criterion, epoch, val_epoch_step, val_loader, Unfreeze_Epoch,
                     anchors, cfg, Cuda, min_val_loss)
