# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from itertools import combinations
from moco.utils import softmax_cross_entropy_with_softtarget


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False, symmetric=False, split_view=False,
                 smooth=0, nn_pos=0,
                 inner_dim=2048, strategy=1, arch='resnet18'):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.symmetric = symmetric
        self.split_view = split_view
        self.combs = 3  # combs=3 for 4 views; combs=2 for 6 views
        self.split_num = 2
        self.smooth = smooth
        self.nn_pos = nn_pos
        self.strategy = strategy
        # create the encoders
        # num_classes is the output fc dimension
        if arch.startswith('resnet'):
            self.encoder_q = base_encoder(num_classes=dim, zero_init_residual=True)
            self.encoder_k = base_encoder(num_classes=dim, zero_init_residual=True)
        else:
            self.encoder_q = base_encoder(num_classes=dim)
            self.encoder_k = base_encoder(num_classes=dim)
        if mlp:  # hack: brute-force replacement
            if arch.startswith('resnet'):
                dim_mlp = self.encoder_q.fc.weight.shape[1]
                self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, inner_dim), nn.ReLU(), nn.Linear(inner_dim, dim))
                self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, inner_dim), nn.ReLU(), nn.Linear(inner_dim, dim))
            else:
                dim_mlp = self.encoder_q.classifier.weight.shape[1]
                self.encoder_q.classifier = nn.Sequential(nn.Linear(dim_mlp, inner_dim), nn.ReLU(), nn.Linear(inner_dim, dim))
                self.encoder_k.classifier = nn.Sequential(nn.Linear(dim_mlp, inner_dim), nn.ReLU(), nn.Linear(inner_dim, dim))
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def _local_split(self, x):  # NxCxHxW --> 4NxCx(H/2)x(W/2)
        _side_indent = x.size(2) // self.split_num, x.size(3) // self.split_num
        cols = x.split(_side_indent[1], dim=3)
        xs = []
        for _x in cols:
            xs += _x.split(_side_indent[0], dim=2)
        x = torch.cat(xs, dim=0)
        return x

    def contrastive_loss(self, im_q, im_k):
        # compute query features
        # q = self.encoder_q_fc(self.encoder_q(im_q))  # queries: NxC
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)  # already normalized

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # shuffle for making use of BN
            im_k_, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            # k = self.encoder_k_fc(self.encoder_k(im_k_))  # keys: NxC
            k = self.encoder_k(im_k_)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators, (N,)

        if self.smooth == 0:
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = nn.CrossEntropyLoss().cuda()(logits, labels)
        else:
            labels_prob = torch.zeros_like(logits).cuda()
            if self.strategy == 1:
                v, t = l_neg.topk(self.nn_pos)
                labels_prob[:, 0] = 1 - self.smooth
                labels_prob.scatter_(1, t + 1, self.smooth / self.nn_pos)
            elif self.strategy == 2:
                v, t = l_neg.topk(self.nn_pos + 1)
                labels_prob[:, 0] = 1 - self.smooth
                if self.nn_pos == 10:
                    splits = [0, 1, 4, 10]
                elif self.nn_pos == 20:
                    splits = [0, 2, 8, 20]
                elif self.nn_pos == 5:
                    splits = [0, 1, 3, 5]
                elif self.nn_pos == 30:
                    splits = [0, 3, 12, 30]
                every_splits = [self.smooth / 2, self.smooth / 4, self.smooth / 4]
                for i in range(1, 4):
                    item_smooth = every_splits[i - 1] / (splits[i] - splits[i - 1])
                    labels_prob.scatter_(1, t[:, splits[i - 1]:splits[i]] + 1, item_smooth)
            elif self.strategy == 3:
                v, t = l_neg.topk(11)
                labels_prob[:, 0] = 0.6
                for i in range(10):
                    labels_prob[torch.arange(labels_prob.size(0)), t[:, i] + 1] = 0.2 / (2 ** i)
                labels_prob[torch.arange(labels_prob.size(0)), t[:, 10] + 1] = 0.2 / (2 ** 9)

            elif self.strategy == 4:
                v, t = l_neg.topk(2)
                labels_prob[:, 0] = 0.5
                labels_prob[torch.arange(labels_prob.size(0)), t[:, 0] + 1] = 0.5
            elif self.strategy == 5:
                labels_prob.add_(self.smooth / (labels_prob.size(1) - 1))
                labels_prob[:, 0] = 1 - self.smooth
            elif self.strategy == 6:
                v, t = l_neg.topk(21)
                labels_prob[:, 0] = 1 - self.smooth
                splits = [0, 2, 8, 20]
                every_splits = self.smooth / 3
                for i in range(1, 4):
                    item_smooth = every_splits / (splits[i] - splits[i - 1])
                    labels_prob.scatter_(1, t[:, splits[i - 1]:splits[i]] + 1, item_smooth)
            loss = softmax_cross_entropy_with_softtarget(logits, labels_prob)
            # loss = nn.KLDivLoss().cuda()(nn.LogSoftmax().cuda()(logits), labels_prob)
            # loss = nn.CrossEntropyLoss().cuda()(logits, labels_prob)
        return loss, q, k

    def split_contrastive_loss(self, im_q, im_k):

        im_q_splits = self._local_split(im_q)
        im_q_splits = self.encoder_q(im_q_splits)
        im_q_splits = list(im_q_splits.split(im_q_splits.size(0) // 4, dim=0))  # 4b x c x
        im_q_orthmix = torch.cat(
            list(map(lambda x: sum(x) / self.combs, list(combinations(im_q_splits, r=self.combs)))),
            dim=0)  # 6 of 2combs / 4 of 3combs
        q = self.encoder_q_fc(im_q_orthmix)
        assert len(q.shape) == 2
        q = nn.functional.normalize(q, dim=1)

        with torch.no_grad():
            im_k_, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k_fc(self.encoder_k(im_k_))  # keys: NxC
            k = nn.functional.normalize(k, dim=1)  # already normalized

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # positive logits: 4Nx1
        qs = q.split(k.size(0), dim=0)
        loss = 0
        for q in qs:
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.T

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss += nn.CrossEntropyLoss().cuda()(logits, labels)

        return loss / len(qs), q, k

    def forward(self, im1, im2=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            loss
        """
        if im2 is None:
            return nn.functional.normalize(self.encoder_q(im1), dim=-1)
        # update the key encoder
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()

        # compute loss
        if self.split_view:
            if self.symmetric:
                loss_12, q1, k2 = self.split_contrastive_loss(im1, im2)
                loss_21, q2, k1 = self.split_contrastive_loss(im2, im1)
                loss = loss_12 + loss_21
                k = torch.cat([k1, k2], dim=0)
            else:
                loss, q, k = self.split_contrastive_loss(im1, im2)
        else:
            if self.symmetric:  # asymmetric loss

                loss_12, q1, k2 = self.contrastive_loss(im1, im2)
                loss_21, q2, k1 = self.contrastive_loss(im2, im1)
                loss = loss_12 + loss_21
                k = torch.cat([k1, k2], dim=0)
            else:  # asymmetric loss
                loss, q, k = self.contrastive_loss(im1, im2)

        self._dequeue_and_enqueue(k)

        return loss


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
