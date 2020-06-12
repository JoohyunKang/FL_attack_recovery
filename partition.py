import numpy as np
import random
import copy
import torch.distributed as dist
import itertools as it

""" Dataset partitioning helper """
class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

def split_partition(partition, cut, num_label):
    # print(f'partition.index:{partition.index}')
    sub1 = copy.deepcopy(partition)
    sub2 = copy.deepcopy(partition)
    total_length = len(partition)
    train_length = int(total_length * cut)
    print(total_length, train_length)
    val_length = total_length - train_length
    # val_length = total_length - train_length
    if train_length < total_length:
        sub1.index = sub1.index[:train_length]
        val_index = np.array(sub2.index[train_length:])
        # if dist.get_rank() == 0:
        #     print(f'val_index:{val_index}')
        num_min_class = val_length
        indexes = list(np.arange(num_label))
        for label in range(num_label):
            indexes[label] = val_index[np.array(sub2.data.targets)[val_index] == label]
            # if dist.get_rank() == 0:
            #     print(f'indexes[{label}]:{indexes[label]}')
            num_class = len(indexes[label])
            # if dist.get_rank() == 0:
            #     print(f'num_class: {num_class}')
            if num_class < num_min_class:
                num_min_class = num_class
        sub2.index = []
        for label in range(num_label):
            for i in range(num_min_class):
                sub2.index.append(indexes[label][i])
        # if dist.get_rank() == 0:
        #     print(f'sub2.index:{sub2.index}')
        #     print(f'len(sub2.index): {len(sub2.index)}')
    else:
        sub2.index = []
    # print()
    # print(f'rank:{rank}')
    # print(f'partition.index:{partition.index}')
    # print(f'sub1.index:{sub1.index}')
    # print(f'sub2.index:{sub2.index}')
    # print(f'len(partition):{len(partition)}')
    # print(f'len(sub1):{len(sub1)}')
    # print(f'len(sub2):{len(sub2)}')
    return sub1, sub2


class DataPartitioner2(object):
    def __init__(self, data, world_size, num_label, seed, split_per_label=2):
        self.data = data
        self.partitions = []
        random.seed(seed)
        num_data = len(data)
        indexes = np.arange(num_data)
        sorted_idx = list(range(num_label))
        for label in range(num_label):
            sorted_idx[label] = list(indexes[np.array(data.targets) == label])
            random.shuffle(sorted_idx[label])
        selected_idx = [[] for _ in range(world_size)]
        for i in range(world_size):
            for j in range(split_per_label):
                unit_size = int(len(sorted_idx[i])/split_per_label)
                selected_idx[i] += sorted_idx[(i+j) % world_size][unit_size*j:unit_size*(j+1)]
        for i in range(world_size):
            random.shuffle(selected_idx[i])
            self.partitions.append(selected_idx[i])

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


# Similar to DataPartitioner3, but every client have at least one data for each label.
class DataPartitioner4(object):
    def __init__(self, data, attackers, world_size, num_label, seed, dataset, scatter, label_per_split=2):
        attacker_size = len(attackers)
        victim_size = world_size - attacker_size
        self.data = data
        self.partitions = []
        random.seed(seed)
        if world_size != 9 or attacker_size not in [2, 3]:
            raise ValueError('Size of attackers or world is not supported')
        indexes = np.arange(len(data))
        selected_idx = [[] for _ in range(world_size)]
        # Sort samples by label
        sorted_idx = [[] for _ in range(num_label)]
        for label in range(num_label):
            if dataset == 'SVHN':
                sorted_idx[label] += list(indexes[np.array(data.labels) == label])
            else:
                sorted_idx[label] += list(indexes[np.array(data.targets) == label])
            random.shuffle(sorted_idx[label])
            # if dist.get_rank() == 0:
            #     print(len(sorted_idx[label]))
        for label in range(num_label):
            if dist.get_rank() == 0:
                print(f'a: {int(scatter*len(sorted_idx[label]))}')
                print(f'b: {len(sorted_idx[label])}')
                print(f'c: {len(sorted_idx[label][int(-scatter * len(sorted_idx[label])):])}')
            for idx in sorted_idx[label][int(-scatter*len(sorted_idx[label])):]:
                selected_idx[int(random.random()*world_size)].append(idx)
        if dist.get_rank() == 0:
            for rank in range(world_size):
                print(f'f: {len(selected_idx[rank])}')
        count = [0 for _ in range(num_label)]
        combi_a = {3: 6, 2: 5}
        count_a = 0
        count_v = 0
        for i, labels in enumerate(it.combinations(range(num_label), label_per_split)):
            if labels in list(it.combinations(range(combi_a[attacker_size]), label_per_split)):
                for label in labels:
                    unit_size = int(len(sorted_idx[label]) * (1 - scatter) / (num_label - 1))
                    selected_idx[count_a % attacker_size] += sorted_idx[label][count[label]:count[label]+unit_size]
                    if dist.get_rank() == 0:
                        print(f'd: {unit_size}')
                    count[label] += unit_size
                count_a += 1
            else:
                for label in labels:
                    unit_size = int(len(sorted_idx[label]) * (1 - scatter) / (num_label - 1))
                    selected_idx[attacker_size + count_v % victim_size] += sorted_idx[label][count[label]:count[label]+unit_size]
                    count[label] += unit_size
                count_v += 1
        for i in range(world_size):
            random.shuffle(selected_idx[i])
            if dist.get_rank() == 0:
                print(len(selected_idx[i]))
            self.partitions.append(selected_idx[i])

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

# This assumes that the number of world is 9 and attacker is 2 or 3
# Because 10C2 = 45 = 9 x 5, 6C2 = 15 = 3 x 5, 5C2 = 10 = 2 x 5
class DataPartitioner3(object):
    def __init__(self, data, attackers, world_size, num_label, seed, label_per_split=2):
        attacker_size = len(attackers)
        victim_size = world_size - attacker_size
        self.data = data
        self.partitions = []
        random.seed(seed)
        if world_size != 9 or attacker_size not in [2, 3]:
            raise ValueError('Size of attackers or world is not supported')
        indexes = np.arange(len(data))
        # Sort samples by label
        sorted_idx = [None for _ in range(num_label)]
        for label in range(num_label):
            sorted_idx[label] = indexes[np.array(data.targets) == label]
            random.shuffle(sorted_idx[label])
            if dist.get_rank() == 0:
                print(len(sorted_idx[label]))
        selected_idx = [[] for _ in range(world_size)]
        count = [0 for _ in range(num_label)]
        combi_a = {3: 6, 2: 5}
        count_a = 0
        count_v = 0
        for i, labels in enumerate(it.combinations(range(num_label), label_per_split)):
            if labels in list(it.combinations(range(combi_a[attacker_size]), label_per_split)):
                for label in labels:
                    unit_size = int(len(sorted_idx[label]) / (num_label - 1))
                    selected_idx[count_a % attacker_size] += list(sorted_idx[label][count[label]:count[label]+unit_size])
                    count[label] += unit_size
                count_a += 1
            else:
                for label in labels:
                    unit_size = int(len(sorted_idx[label]) / (num_label - 1))
                    selected_idx[attacker_size + count_v % victim_size] += list(sorted_idx[label][count[label]:count[label]+unit_size])
                    count[label] += unit_size
                count_v += 1
        for i in range(world_size):
            random.shuffle(selected_idx[i])
            if dist.get_rank() == 0:
                print(len(selected_idx[i]))
            self.partitions.append(selected_idx[i])

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

# Backup at 20200608
class DataPartitioner(object):

    def __init__(self, data, world_size, num_label, p, seed):
        self.data = data
        self.partitions = []
        random.seed(seed)
        indexes = np.arange(len(data))
        # rng.shuffle(indexes)

        # print(f'data.targets:{data.targets}')
        # self.num_label = num_label
        if world_size % num_label != 0:
            raise ValueError('world_size should be integer multiple of num_label')
        sorted_idx = list(range(num_label))
        selected_idx = [[] for _ in range(world_size)]
        member = [[] for _ in range(num_label)]
        for label in range(num_label):
            # print(f'data.targets == label: {data.targets == label}')
            sorted_idx[label] = indexes[np.array(data.targets) == label]
        for rank in range(world_size):
            # num_per_label = world_size // num_label
            # member[(rank // num_per_label) % num_label].append(rank)
            member[rank % num_label].append(rank)
        for label in range(num_label):
            member_len = len(member[label])
            other_member_len = world_size - member_len
            other_member = []
            for rank in range(world_size):
                if rank not in member[label]:
                    other_member.append(rank)
            for idx in sorted_idx[label]:
                if random.random() < p:
                    rank = random.choices(member[label], [1]*member_len)[0]
                else:
                    rank = random.choices(other_member, [1]*other_member_len)[0]
                selected_idx[rank].append(idx)

        for i in range(world_size):
            # print(f'selected_idx[{i}]:{selected_idx[i]}')
            random.shuffle(selected_idx[i])
            # print(f'selected_idx[{i}]:{selected_idx[i]}')
            self.partitions.append(selected_idx[i])
        # for frac in sizes:
        #     part_len = int(frac * data_len)
        #     self.partitions.append(indexes[0:part_len])
        #     indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


# class DataPartitioner(object):
#
#     def __init__(self, data, sizes, seed):
#         self.data = data
#         self.partitions = []
#         rng = Random()
#         rng.seed(seed)
#         data_len = len(data)
#         indexes = [x for x in range(0, data_len)]
#         rng.shuffle(indexes)
#
#         for frac in sizes:
#             part_len = int(frac * data_len)
#             self.partitions.append(indexes[0:part_len])
#             indexes = indexes[part_len:]
#
#     def use(self, partition):
#         return Partition(self.data, self.partitions[partition])
#
#
