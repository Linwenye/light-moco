import torch
from moco.utils import *
import time
from tqdm import tqdm
import torch.nn.functional as F


def kNN(net, trainloader, testloader, K, sigma, dim=512):
    net.eval()
    net_time = AverageMeter('net_time')
    cls_time = AverageMeter('cls_time')
    total = 0
    testsize = testloader.dataset.__len__()
    trainFeatures = torch.empty(trainloader.dataset.__len__(), dim).t().cuda()

    transform_bak = trainloader.dataset.transform
    trainloader.dataset.transform = testloader.dataset.transform
    # temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=512, shuffle=False, num_workers=16)
    labels = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader), desc='loading features', total=len(trainloader)):
            labels.append(concat_all_gather(targets.cuda(non_blocking=True)))
            batchSize = 512
            features = F.normalize(net(inputs))
            features = concat_all_gather(features)
            assert features.size(0) == 512
            trainFeatures[:, batch_idx * batchSize:batch_idx * batchSize + batchSize] = features.data.t()

    # trainLabels = torch.LongTensor([y for x,y in trainloader.dataset.imgs]).cuda()
    trainLabels = torch.cat(labels, dim=0)
    print('label length', len(trainLabels))
    C = trainLabels.max() + 1
    print('C', C)
    trainloader.dataset.transform = transform_bak
    print('train feature shape', trainFeatures.shape)
    # torch.save(trainFeatures,'trainFeature.ckpt')
    top1 = 0.
    top5 = 0.
    end = time.time()
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets) in enumerate(testloader):
            end = time.time()
            targets = targets.cuda(non_blocking=True)
            batchSize = inputs.size(0)
            features = F.normalize(net(inputs))
            net_time.update(time.time() - end)
            end = time.time()

            dist = torch.mm(features, trainFeatures)

            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1, -1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)

            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            yd_transform = yd.clone().div_(sigma).exp_()
            probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1, C), yd_transform.view(batchSize, -1, 1)), 1)
            _, predictions = probs.sort(1, True)

            # Find which predictions match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            cls_time.update(time.time() - end)

            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, 5).sum().item()

            total += targets.size(0)

            print('Test [{}/{}]\t'
                  'Net Time {net_time.val:.3f} ({net_time.avg:.3f})\t'
                  'Cls Time {cls_time.val:.3f} ({cls_time.avg:.3f})\t'
                  'Top1: {:.2f}  Top5: {:.2f}'.format(
                total, testsize, top1 * 100. / total, top5 * 100. / total, net_time=net_time, cls_time=cls_time))

    print('final acc', top1 * 100. / total)

    return top1 / total
