import argparse
import random
from datetime import datetime
from time import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from core import layers
from core import metrics
from core import utils
from core.logger import Logger
from tqdm import trange, tqdm


class LeNet5(layers.ModuleWrapper):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.num_classes = 10
        self.conv1 = layers.ConvVDO(1, 20, 5, padding=0, alpha_shape=(1, 1, 1, 1))
        self.relu1 = nn.ReLU(True)
        self.pool1 = nn.MaxPool2d(2, padding=0)

        self.conv2 = layers.ConvVDO(20, 50, 5, padding=0, alpha_shape=(1, 1, 1, 1))
        self.relu2 = nn.ReLU(True)
        self.pool2 = nn.MaxPool2d(2, padding=0)

        self.flatten = layers.FlattenLayer(800)
        self.dense1 = layers.LinearVDO(800, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.relu3 = nn.ReLU()

        self.dense2 = layers.LinearVDO(500, 10)


def get_args():
    parser = argparse.ArgumentParser(description='SSD training with PyTorch')

    parser.add_argument('--gpus', default=None, help='List of GPUs used for training - e.g 0,1,3')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--type', default='float32', help='Type of tensor: float32, float16, float64. Default: float32')
    parser.add_argument('--print-model', action='store_true', default=False, help='print model to stdout')

    # Optimization options
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('-b', '--batch-size', default=200, type=int, metavar='N', help='mini-batch size (default: 200)')
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='The learning rate.')

    # Checkpoints
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='Just evaluate model')
    parser.add_argument('--save', '-s', type=str, default='', help='Folder to save checkpoints.')
    parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: random)')

    args = parser.parse_args()

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = time_stamp

    if args.gpus is not None:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        cudnn.enabled = True
        cudnn.benchmark = True
        args.device = 'cuda:' + str(args.gpus[0])
        torch.cuda.set_device(args.gpus[0])
        torch.cuda.manual_seed(args.seed)
    else:
        args.gpus = []
        args.device = 'cpu'

    if args.type == 'float64':
        args.dtype = torch.float64
    elif args.type == 'float32':
        args.dtype = torch.float32
    elif args.type == 'float16':
        args.dtype = torch.float16
    else:
        raise ValueError('Wrong type!')  # TODO int8
    return args


def main():
    fmt = {'tr_loss': '3.1e',
           'tr_acc': '.4f',
           'te_acc_det': '.4f',
           'te_acc_stoch': '.4f',
           'te_acc_ens': '.4f',
           'te_acc_perm_sigma': '.4f',
           'te_acc_zero_mean': '.4f',
           'te_acc_perm_sigma_ens': '.4f',
           'te_acc_zero_mean_ens': '.4f',
           'te_nll_det': '.4f',
           'te_nll_stoch': '.4f',
           'te_nll_ens': '.4f',
           'te_nll_perm_sigma': '.4f',
           'te_nll_zero_mean': '.4f',
           'te_nll_perm_sigma_ens': '.4f',
           'te_nll_zero_mean_ens': '.4f',
           'time': '.3f'}
    fmt = {**fmt, **{'la%d' % i: '.4f' for i in range(4)}}
    args = get_args()
    logger = Logger("lenet5-VDO", fmt=fmt)

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train_sampler = torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(trainset), batch_size=args.batch_size,
                                                  drop_last=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_sampler=train_sampler, num_workers=args.workers,
                                              pin_memory=True)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_sampler = torch.utils.data.BatchSampler(torch.utils.data.SequentialSampler(testset),
                                                 batch_size=args.batch_size, drop_last=False)
    testloader = torch.utils.data.DataLoader(testset, batch_sampler=test_sampler, num_workers=args.workers,
                                             pin_memory=True)

    net = LeNet5()
    net = net.to(device=args.device, dtype=args.dtype)
    if args.print_model:
        logger.print(net)
    criterion = metrics.SGVLB(net, len(trainset)).to(device=args.device, dtype=args.dtype)
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    epochs = args.epochs
    lr_start = args.learning_rate
    for epoch in trange(epochs):  # loop over the dataset multiple times
        t0 = time()
        utils.adjust_learning_rate(optimizer, metrics.lr_linear(epoch, 0, epochs, lr_start))
        net.train()
        training_loss = 0
        accs = []
        steps = 0
        for i, (inputs, labels) in enumerate(tqdm(trainloader), 0):
            steps += 1
            inputs, labels = inputs.to(device=args.device, dtype=args.dtype), labels.to(device=args.device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            accs.append(metrics.logit2acc(outputs.data, labels))  # probably a bad way to calculate accuracy
            training_loss += loss.item()

        logger.add(epoch, tr_loss=training_loss / steps, tr_acc=np.mean(accs))

        # Deterministic test
        net.eval()
        acc, nll = utils.evaluate(net, testloader, device=args.device, num_ens=1)
        logger.add(epoch, te_nll_det=nll, te_acc_det=acc)

        # Stochastic test
        net.train()
        acc, nll = utils.evaluate(net, testloader, device=args.device, num_ens=1)
        logger.add(epoch, te_nll_stoch=nll, te_acc_stoch=acc)

        # Test-time averaging
        net.train()
        acc, nll = utils.evaluate(net, testloader, device=args.device, num_ens=20)
        logger.add(epoch, te_nll_ens=nll, te_acc_ens=acc)

        # Zero-mean
        net.train()
        net.dense1.set_flag('zero_mean', True)
        acc, nll = utils.evaluate(net, testloader, device=args.device, num_ens=1)
        net.dense1.set_flag('zero_mean', False)
        logger.add(epoch, te_nll_zero_mean=nll, te_acc_zero_mean=acc)

        # Permuted sigmas
        net.train()
        net.dense1.set_flag('permute_sigma', True)
        acc, nll = utils.evaluate(net, testloader, device=args.device, num_ens=1)
        net.dense1.set_flag('permute_sigma', False)
        logger.add(epoch, te_nll_perm_sigma=nll, te_acc_perm_sigma=acc)

        # Zero-mean test-time averaging
        net.train()
        net.dense1.set_flag('zero_mean', True)
        acc, nll = utils.evaluate(net, testloader, device=args.device, num_ens=20)
        net.dense1.set_flag('zero_mean', False)
        logger.add(epoch, te_nll_zero_mean_ens=nll, te_acc_zero_mean_ens=acc)

        # Permuted sigmas test-time averaging
        net.train()
        net.dense1.set_flag('permute_sigma', True)
        acc, nll = utils.evaluate(net, testloader, device=args.device, num_ens=20)
        net.dense1.set_flag('permute_sigma', False)
        logger.add(epoch, te_nll_perm_sigma_ens=nll, te_acc_perm_sigma_ens=acc)

        logger.add(epoch, time=time() - t0)
        las = [np.mean(net.conv1.log_alpha.data.cpu().numpy()),
               np.mean(net.conv2.log_alpha.data.cpu().numpy()),
               np.mean(net.dense1.log_alpha.data.cpu().numpy()),
               np.mean(net.dense2.log_alpha.data.cpu().numpy())]

        logger.add(epoch, **{'la%d' % i: las[i] for i in range(4)})
        logger.iter_info()
        logger.save(silent=True)
        torch.save(net.state_dict(), logger.checkpoint)

    logger.save()


if __name__ == "__main__":
    main()
