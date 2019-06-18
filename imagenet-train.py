# -*- coding: utf-8 -*-

# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import argparse
import os
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim import lr_scheduler
import tensorboard_logger

from models import (blresnext_model, blresnet_model, blseresnext_model)
from imagenet_utils import get_augmentor, get_imagenet_dataflow, train, validate

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--backbone_net', default='blresnext', type=str, help='backbone network',
                    choices=['blresnext', 'blresnet', 'blseresnext'])
parser.add_argument('-d', '--depth', default=50, type=int, metavar='N',
                    help='depth of resnext (default: 50)', choices=[50, 101, 152, 154])
parser.add_argument('--basewidth', default=4, type=int, help='basewidth')
parser.add_argument('--cardinality', default=32, type=int, help='cardinality')
parser.add_argument('--alpha', default=2, type=int, metavar='N', help='ratio of channels')
parser.add_argument('--beta', default=4, type=int, metavar='N', help='ratio of layers')

parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=18, type=int, metavar='N',
                    help='number of data loading workers (default: 18)')

parser.add_argument('--epochs', default=110, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_scheduler', default='cosine', type=str,
                    help='learning rate scheduler', choices=['step', 'cosine'])
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--input_shape', default=224, type=int, metavar='N', help='input image size')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', type=str, metavar='PATH',
                    help='use pre-trained model')
parser.add_argument('--logdir', default='', type=str, help='log path')


def main():
    global args
    args = parser.parse_args()
    cudnn.benchmark = True

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    strong_augmentor = False
    if args.backbone_net == 'blresnext':
        backbone = blresnext_model
        arch_name = "ImageNet-bLResNeXt-{}-{}x{}d-a{}-b{}".format(
            args.depth, args.cardinality, args.basewidth, args.alpha, args.beta)
        backbone_setting = [args.depth, args.basewidth, args.cardinality, args.alpha, args.beta]
    elif args.backbone_net == 'blresnet':
        backbone = blresnet_model
        arch_name = "ImageNet-bLResNet-{}-a{}-b{}".format(args.depth, args.alpha, args.beta)
        backbone_setting = [args.depth, args.alpha, args.beta]
    elif args.backbone_net == 'blseresnext':
        backbone = blseresnext_model
        arch_name = "ImageNet-bLSEResNeXt-{}-{}x{}d-a{}-b{}".format(
            args.depth, args.cardinality, args.basewidth, args.alpha, args.beta)
        backbone_setting = [args.depth, args.basewidth, args.cardinality, args.alpha, args.beta]
        strong_augmentor = True
    else:
        raise ValueError("Unsupported backbone.")

    # create model
    model = backbone(*backbone_setting).cuda()

    model = torch.nn.DataParallel(model).cuda()

    if args.pretrained is not None:
        print("=> using pre-trained model '{}'".format(arch_name))
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> creating model '{}'".format(arch_name))

    # define loss function (criterion) and optimizer
    train_criterion = nn.CrossEntropyLoss().cuda()
    val_criterion = nn.CrossEntropyLoss().cuda()

    # Data loading code
    valdir = os.path.join(args.data, 'val')
    val_loader = get_imagenet_dataflow(False, valdir, args.batch_size, get_augmentor(
        False, args.input_shape, strong_augmentor), workers=args.workers)

    log_folder = os.path.join(args.logdir, arch_name)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    if args.evaluate:
        val_top1, val_top5, val_losses, val_speed = validate(val_loader, model, val_criterion)
        print('Val@{}: \tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\t'
              'Speed: {:.2f} ms/batch\t'.format(args.input_shape, val_losses, val_top1,
                                                val_top5, val_speed * 1000.0), flush=True)
        return

    traindir = os.path.join(args.data, 'train')
    train_loader = get_imagenet_dataflow(True, traindir, args.batch_size, get_augmentor(
        True, args.input_shape, strong_augmentor), workers=args.workers)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)
    if args.lr_scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
    elif args.lr_scheduler == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    else:
        raise ValueError("Unsupported scheduler.")

    tensorboard_logger.configure(os.path.join(log_folder))
    # optionally resume from a checkpoint
    best_top1 = 0.0
    if args.resume:
        logfile = open(os.path.join(log_folder, 'log.log'), 'a')
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_top1 = checkpoint['best_top1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        logfile = open(os.path.join(log_folder, 'log.log'), 'w')

    print(args, flush=True)
    print(model, flush=True)

    print(args, file=logfile, flush=True)
    print(model, file=logfile, flush=True)

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step(epoch)
        try:
            # get_lr get all lrs for every layer of current epoch, assume the lr for all layers are identical
            lr = scheduler.get_lr()[0]
        except Exception as e:
            lr = None
        # train for one epoch
        train_top1, train_top5, train_losses, train_speed, speed_data_loader, train_steps = \
            train(train_loader,
                  model,
                  train_criterion,
                  optimizer, epoch + 1)
        # evaluate on validation set
        val_top1, val_top5, val_losses, val_speed = validate(val_loader, model, val_criterion)

        print('Train: [{:03d}/{:03d}]\tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch\t'
              'Data loading: {:.2f} ms/batch'.format(epoch + 1, args.epochs, train_losses, train_top1, train_top5,
                                                     train_speed * 1000.0, speed_data_loader * 1000.0),
              file=logfile, flush=True)
        print('Val  : [{:03d}/{:03d}]\tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch'.format(
            epoch + 1, args.epochs, val_losses, val_top1, val_top5, val_speed * 1000.0), file=logfile, flush=True)

        print('Train: [{:03d}/{:03d}]\tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch\t'
              'Data loading: {:.2f} ms/batch'.format(epoch + 1, args.epochs, train_losses, train_top1, train_top5,
                                                     train_speed * 1000.0, speed_data_loader * 1000.0), flush=True)
        print('Val  : [{:03d}/{:03d}]\tLoss: {:4.4f}\tTop@1: {:.4f}\tTop@5: {:.4f}\tSpeed: {:.2f} ms/batch'.format(
            epoch + 1, args.epochs, val_losses, val_top1, val_top5, val_speed * 1000.0), flush=True)

        # remember best prec@1 and save checkpoint
        is_best = val_top1 > best_top1
        best_top1 = max(val_top1, best_top1)

        save_dict = {'epoch': epoch + 1,
                     'arch': arch_name,
                     'state_dict': model.state_dict(),
                     'best_top1': best_top1,
                     'optimizer': optimizer.state_dict(),
                     }

        save_checkpoint(save_dict, is_best, filepath=log_folder)
        if lr is not None:
            tensorboard_logger.log_value('learnnig-rate', lr, epoch + 1)
        tensorboard_logger.log_value('val-top1', val_top1, epoch + 1)
        tensorboard_logger.log_value('val-loss', val_losses, epoch + 1)
        tensorboard_logger.log_value('train-top1', train_top1, epoch + 1)
        tensorboard_logger.log_value('train-loss', train_losses, epoch + 1)
        tensorboard_logger.log_value('best-val-top1', best_top1, epoch + 1)

    logfile.close()


def save_checkpoint(state, is_best, filepath=''):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


if __name__ == '__main__':
    main()
