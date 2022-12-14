#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import moco.loader
import moco.builder
from tqdm import tqdm

import torch.nn.functional as F
from knn_imagenet import kNN

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/apdcephfs/share_916081/liniuslin/Collapse/datasets/picked/',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=65536, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.999, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')

# options for moco v2
parser.add_argument('--mlp', action='store_true',
                    help='use mlp head')
parser.add_argument('--aug-plus', action='store_true',
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = moco.builder.MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    # print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # define loss function (criterion) and optimizer

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    view_pos = os.path.join(args.data, 'pos')
    view_neg = os.path.join(args.data, 'neg')
    view_pos_other = os.path.join(args.data, 'pos_others')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    pos_dataset = datasets.ImageFolder(
        view_pos, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    print('classes', pos_dataset.classes)
    print('idx', pos_dataset.class_to_idx)
    pos_loader = torch.utils.data.DataLoader(
        pos_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    neg_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(view_neg, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    pos_other_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(view_pos_other, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.gpu == 0:
        eval_sim(pos_loader, neg_loader, pos_other_loader, model, args)
    dist.destroy_process_group()


def eval_sim(pos_loader, neg_loader, pos_other_loader, model, args):
    # switch to train mode
    model.eval()
    feature_pos_bank, feature_neg_bank, feature_pos_others, labels_bank = [], [], [], []
    with torch.no_grad():
        for i, ((images_pos, labels1), (images_neg, labels2), (images_pos_other, _)) in tqdm(
                enumerate(zip(pos_loader, neg_loader, pos_other_loader))):
            feature_pos = model(images_pos.cuda(non_blocking=False))
            feature_neg = model(images_neg.cuda(non_blocking=False))
            feature_pos_other = model(images_pos_other.cuda(non_blocking=False))
            feature_pos_bank.append(F.normalize(feature_pos, dim=-1))
            feature_neg_bank.append(F.normalize(feature_neg, dim=-1))
            feature_pos_others.append(F.normalize(feature_pos_other, dim=-1))
            labels_bank.append(labels1.cuda(non_blocking=False))
    feature_pos_bank = torch.cat(feature_pos_bank, dim=0)
    feature_neg_bank = torch.cat(feature_neg_bank, dim=0)
    feature_pos_others = torch.cat(feature_pos_others, dim=0)
    labels_bank = torch.cat(labels_bank, dim=0)
    sim_matrix = feature_pos_bank @ feature_pos_bank.T
    pos_neg = (feature_pos_bank * feature_neg_bank).sum(-1)
    sim_matrix_other = feature_pos_bank @ feature_pos_other.T
    n = sim_matrix.shape[0]
    print('total {} images'.format(n))
    print('avg without diag', sim_matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].mean())
    labels_list = labels_bank.tolist()
    # print('label_list', labels_list)
    labels_set = set(labels_list)
    print('label_set', labels_set)
    print('labels num: ', len(labels_set))
    label_specific_total_avg = 0
    sim_matrix_other_total_avg = 0
    for label in labels_set:
        label_index = (labels_bank == label).nonzero(as_tuple=False).squeeze()

        print('label_index', label_index)
        label_specific_sim = sim_matrix[label_index][:, label_index]
        sim_matrix_other_label = sim_matrix_other[label_index][:, label_index]
        # print('label_specific_sim',label_specific_sim)
        print('spec shape', label_specific_sim.shape)
        print('label', label)
        # print('avg sim with diag', label_specific_sim.mean())
        n = label_specific_sim.shape[0]
        label_specific_sim_avg = label_specific_sim.flatten()[:-1].view(n - 1, n + 1)[:, 1:].mean()
        sim_matrix_other_label_avg = sim_matrix_other_label.flatten()[:-1].view(n - 1, n + 1)[:, 1:].mean()
        label_specific_total_avg += label_specific_sim_avg
        sim_matrix_other_total_avg += sim_matrix_other_label_avg
        print('avg sim without diag', label_specific_sim_avg)
        print('label_sim', label_specific_sim)
        print('pos_neg sim', pos_neg[label_index])
    print('-------------')
    print('label_specific_total_avg', label_specific_total_avg / 10)
    print('other_total_avg', sim_matrix_other_total_avg / 10)
    print('pos_neg avg', pos_neg.mean())


if __name__ == '__main__':
    main()
