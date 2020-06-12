import os
import scipy
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.multiprocessing as mp

from models import ResNet, CNN, ResNetBN, CNN10, MLP
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import matplotlib.pyplot as plt
from partition import DataPartitioner, split_partition, DataPartitioner2, DataPartitioner3, DataPartitioner4
from attack_and_restore import attack_and_restore
from torch.multiprocessing import Process
import copy
import datetime
from train_test import train, eval
from multiseedplot import multiseedplot


# Global Variable For training
# You just use the following hyper-parameters



def partition_dataset(train_dataset, rank, world_size, num_label, attackers, seed, batch_size, dataset, scatter):
    """ Partitioning MNIST """
    # partition = DataPartitioner3(train_dataset, attackers, world_size, 10, seed, 2).use(rank)
    partition = DataPartitioner4(train_dataset, attackers, world_size, 10, seed, dataset, scatter).use(rank)
    # num_label = 10
    # partition = DataPartitioner(train_dataset, world_size, num_label, p, seed).use(rank)
    # partition, partition_v = split_partition(partition, cut, num_label)
    train_loader = torch.utils.data.DataLoader(partition, batch_size=batch_size, shuffle=True)
    num_part = len(train_dataset)
    num_data = len(partition)
    return train_loader, num_data, num_part


def setup(rank, world_size, seed):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(seed)


def cleanup():
    dist.destroy_process_group()


def demo_basic(rank, world_size, method_a, methods_r, attackers, dataset, modelname,
               power, dir, optimizer, clamp, batch_size, scatter, seed, num_epoch, load,
               sum_epoch, basename, train_dataset, test_loader):
    # Setup torch.distributed
    setup(rank, world_size, seed)
    # assign gpu to each processes
    device_count = torch.cuda.device_count()
    # if world_size <= device_count:
    #     n = device_count // world_size
    #     device_ids = list(range(rank * n, (rank + 1) * n))
    # else:
    #     device_ids = [rank % device_count]
    device = rank % device_count
    attacker_size = len(attackers)
    if rank in attackers:
        memory = {'global': [], 'attack': [], 'local_v': [], 'local_a': [],
                  'alone': [], 'attacked': [], 'dist_local_a': [],
                  'single': {'compen': [], 'restored': []},
                  'cumulative': {'compen': [], 'restored': []},
                  'recursive': {'local_a': [], 'local_vt': [], 'restored': []},
                  'translative': {'local_a': [], 'local_vt': [], 'restored': []},
                  'projective_t': {'local_a': [], 'local_vt': [], 'restored': []},
                  'projective': {'local_a': [], 'local_vt': [], 'restored': []}}
    train_loader, num_data, num_part = partition_dataset(train_dataset, rank, world_size, 10,
                                                         attackers, seed, batch_size, dataset, scatter)
    model = dict()
    if modelname == 'ResNet':
        model['all'] = ResNet().to(device)
    elif modelname == 'ResNetBN':
        model['all'] = ResNetBN().to(device)
    elif modelname == 'CNN':
        model['all'] = CNN().to(device)
    elif modelname == 'CNN10':
        model['all'] = CNN10().to(device)
    elif modelname == 'MLP':
        model['all'] = MLP().to(device)
    for key in ['attacked', 'alone']:
        model[key] = copy.deepcopy(model['all'])
    if load:
        loaded = 1
        path = os.path.join(load[0], 'memory', f'{basename}_epoch{load[1]}.tar')
        if not os.path.exists(path):
            raise ValueError(f'fail loading: no such file {path}')
        checkpoint = torch.load(path)
        if rank == attackers[-1]:
            print('------------------------------------------')
            print(f'start from {path}')
            acc = checkpoint['acc']
            loss = checkpoint['loss']
            # loss = checkpoint['loss']
        for key in ['all', 'attacked', 'alone']:
            model[key].load_state_dict(checkpoint[key])
            # model[key] = checkpoint[key].to(device)
        if rank in attackers:
            for key in checkpoint:
                if key in methods_r:
                    for key2 in memory[key]:
                        memory[key][key2].append(copy.deepcopy(model['all']))
                        memory[key][key2][-1].load_state_dict(checkpoint[key][key2])
                        # memory[key][key2].append(checkpoint[key][key2].to(device))
                elif key == 'dist_local_a':
                    memory[key].append(checkpoint[key])
                elif key not in ['all', 'acc', 'loss'] and type(checkpoint[key]) is not dict:
                    memory[key].append(copy.deepcopy(model['all']))
                    memory[key][-1].load_state_dict(checkpoint[key])
                    # memory[key].append(checkpoint[key].to(device))
    else:
        loaded = 0
        loss = {'all': [], 'attacked': [], 'alone': []}
        acc = {'all': [], 'attacked': [], 'alone': []}
        for method_r in methods_r:
            acc[method_r] = []
    for epoch in range(loaded, num_epoch + loaded):
        # print(model['all'])
        loss_tmp1 = train(model['all'], train_loader, device, optimizer)
        if rank in attackers:
            loss_tmp2, memory = attack_and_restore(model['attacked'], train_loader, device, epoch,
                                                  method_a, attackers, memory, power, methods_r,
                                                  optimizer, clamp, seed)
            memory['attacked'].append(copy.deepcopy(model['attacked']))
            loss_tmp3 = train(model['alone'], train_loader, device, optimizer, attackers)
            memory['alone'].append(copy.deepcopy(model['alone']))
        else:
            train(model['attacked'], train_loader, device, optimizer)
        if rank == attackers[-1]:
            if epoch == loaded:
                print('------------------------------------------')
                print(f'start: {basename.replace("_", "/")}')
            print('------------------------------------------')
            if load:
                print(f'epoch: {epoch + load[1] - 1}')
            else:
                print(f'epoch: {epoch}')
            print(f'power:{power}')
            # accuracy on test & validation set
            # validation set have data that is equal in number over class
            for key in model:
                acc[key].append(eval(model[key], test_loader, device))
                print(f'acc_{key} = {acc[key][-1]}')
            loss['all'].append(loss_tmp1)
            loss['attacked'].append(loss_tmp2)
            loss['alone'].append(loss_tmp3)
            for method_r in methods_r:
                acc[method_r].append(eval(memory[method_r]['restored'][-1], test_loader, device))
                print(f'acc_{method_r} = {acc[method_r][-1]}')
            for key in loss:
                print(f'loss_{key} = {loss[key][-1]}')

    dist.barrier()
    # validate_vt(memory)
    if rank == attackers[-1]:
        print('------------------------------------------')
        print(f'end: {basename.replace("_", "/")}')
        print('------------------------------------------')
        # save parameters, accuracy, and loss
        obj = dict()
        obj['all'] = model['all'].state_dict()
        # obj['all'] = model['all']
        for key in memory:
            if type(memory[key]) is dict:
                if key in methods_r:
                    obj[key] = dict()
                    for key2 in memory[key]:
                        obj[key][key2] = memory[key][key2][-1].state_dict()
                        # obj[key][key2] = memory[key][key2][-1]
            elif key == 'dist_local_a':
                obj[key] = memory[key][-1]
            else:
                obj[key] = memory[key][-1].state_dict()
                # obj[key] = memory[key][-1]
        obj['loss'] = loss
        obj['acc'] = acc
        path = os.path.join(dir, 'memory', f'{basename}_epoch{sum_epoch}.tar')
        torch.save(obj, path)
        print(f'saved {path}')
        # # plot accuracy figures
        # fig, ax = plt.subplots(figsize=(12, 10))
        # # ax.set_title(basename.replace("_", "/"), fontsize=20)
        # for key in acc:
        #     ax.plot(np.arange(sum_epoch), acc[key], lw=2, marker='o', ms=2.5, label=key)
        # for key in ['all', 'attacked', 'alone'] + methods_r:
        #     ax.hlines(max(acc[key]), 0, sum_epoch - 1, label=f'{key}:{max(acc[key])}', linestyles='--')
        # ax.set_ylabel('test accuracy', fontsize=10)
        # ax.set_xlabel('epoch', fontsize=10)
        # ax.set_xlim([0, sum_epoch - 1])
        # ax.set_ylim([10, 90])
        # ax.legend(fontsize=10, loc='upper left')
        # fig.savefig(os.path.join(dir, 'figure', f'{basename}_epoch{sum_epoch}.png'), bbox_inches='tight')
    dist.barrier()
    cleanup()

if __name__ == "__main__":
    # basename = f'{modelname}_{dataset}_cut{cut}_cuta{cut_a}_attack{method_a}_world{world_size}_attacker{attacker_size}_' \
    #            f'distribution{distribution}_optimizer{optimizer[0]}_lr{optimizer[1]}_power{power}_clamp{clamp}'
    dt = datetime.datetime.now()
    dir = f'{dt.year}{str(dt.month).zfill(2)}{str(dt.day).zfill(2)}'
    if not os.path.exists(dir):
        os.mkdir(dir)
        os.mkdir(os.path.join(dir, 'figure'))
        os.mkdir(os.path.join(dir, 'memory'))
    train_datasets = {
        'CIFAR10': dsets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=False),
        'MNIST': dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=False),
        'FashionMNIST': dsets.FashionMNIST(root='./data', train=True, transform=transforms.ToTensor(), download=False),
        'SVHN': dsets.SVHN(root='./data', split='train', transform=transforms.ToTensor(), download=False)
        }
    test_datasets = {'CIFAR10': dsets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor()),
                     'MNIST': dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor()),
                     'FashionMNIST': dsets.FashionMNIST(root='./data', train=False, transform=transforms.ToTensor()),
                     'SVHN': dsets.SVHN(root='./data', split='test', transform=transforms.ToTensor(), download=False)
                     }
    world_size = 9
    participants = list(range(world_size))
    method_a = 'median'
    methods_r = ['projective']
    methods_r = ['cumulative', 'translative', 'projective']
    params = [
#   (dataset, modelname, opt0, opt1, bsz, attacker, power, scatter)
#         ('CIFAR10', 'ResNet', 'SGD', 0.1, 10, 2, 4.5, 0.2),
        # ('CIFAR10', 'ResNet', 'SGD', 0.1, 10, 3, 3.5),
        # ('FashionMNIST', 'CNN', 'SGD', 0.05, 100, 3, 3.5, 0.1),
        # ('FashionMNIST', 'CNN', 'SGD', 0.05, 100, 3, 3.5, 0.2),
        # ('FashionMNIST', 'CNN', 'SGD', 0.05, 100, 3, 2.5, 0.1),
        # ('FashionMNIST', 'CNN', 'SGD', 0.05, 100, 3, 2.5, 0.2),
        ('FashionMNIST', 'CNN', 'SGD', 0.05, 100, 2, 3.5, 0.1),
        # ('FashionMNIST', 'CNN', 'SGD', 0.05, 100, 2, 3.5, 0.2),
        # ('CIFAR10', 'ResNet', 'SGD', 0.1, 10, 3, 2.5, 0.1),
        # ('CIFAR10', 'ResNet', 'SGD', 0.1, 10, 3, 2.5, 0.2),
    ]
    clamps = [2]
    seeds = [41, 42, 43]
    num_epoch = 50
    load = None
    # load = ('20200611', 50)
    for seed in seeds:
        for param in params:
            dataset = param[0]
            modelname = param[1]
            optimizer = (param[2], param[3])
            batch_size = param[4]
            attacker_size = param[5]
            attackers = participants[:attacker_size]
            power = param[6]
            scatter = param[7]
            for clamp in clamps:
                test_loader = torch.utils.data.DataLoader(test_datasets[dataset], batch_size=batch_size, shuffle=False)
                basename = f'{modelname}_{dataset}_world{world_size}_attacker{attacker_size}_' \
                           f'optimizer{optimizer[0]}_lr{optimizer[1]}_power{power}_clamp{clamp}_' \
                           f'bsz{batch_size}_scatter{scatter}'
                if load:
                     sum_epoch = num_epoch + load[1]
                else:
                     sum_epoch = num_epoch
                mp.spawn(demo_basic,
                     args=(world_size, method_a, methods_r, attackers,
                           dataset, modelname,
                           power, dir, optimizer, clamp, batch_size,
                           scatter, seed, num_epoch, load, sum_epoch, f'{basename}_seed{seed}',
                           train_datasets[dataset], test_loader),
                     nprocs=world_size,
                     join=True)

