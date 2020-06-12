import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import datetime
import seaborn as sns
import matplotlib.font_manager
from scipy import stats

plt.rcParams["font.family"] = "Times New Roman"

def multiseedplot(basename, seeds, methods_r, load, dir):
    # base, seed = basename.split('_seed')
    acc_list_dict = dict()
    acc_array_dict = dict()
    # loss_list_dict = dict()
    acc_avg_dict = dict()
    acc_sem_dict = dict()
    acc_max_dict = dict()
    # loss_avg_dict = dict()
    label_list = ['all', 'attacked', 'alone'] + methods_r
    for key in label_list:
        acc_list_dict[key] = []
        # loss_list_dict[key] = []
    for seed in seeds:
        path = os.path.join(load[0], 'memory', f'{basename}_seed{seed}_epoch{load[1]}.tar')
        if not os.path.exists(path):
            raise ValueError(f'Fail loading: no such file {path}')
        checkpoint = torch.load(path)
        for key in label_list:
            acc_list_dict[key].append(checkpoint['acc'][key])
            # loss_list_dict[key].append(checkpoint['loss'][key])
    # Average acc and loss over seeds
    for key in label_list:
        acc_array_dict[key] = np.array(acc_list_dict[key])
        acc_avg_dict[key] = np.mean(acc_array_dict[key], axis=0)
        acc_sem_dict[key] = stats.sem(acc_array_dict[key], axis=0)
        acc_max_dict[key] = np.amax(acc_avg_dict[key])
        # loss_avg_dict[key] = np.mean(np.array(loss_list_dict[key]), axis=0)
    # Plot accuracy figures
    fig, ax = plt.subplots(figsize=(12, 10))
    c_dict = {'projective': 'navy', 'attacked': 'gray', 'all': 'darkgrey', 'alone': ''}
    ls_dict = {'projective': 'solid', 'attacked': 'dashed', 'all': 'dashdot'}
    # ax.set_title(basename.replace("_", "/"), fontsize=20)
    # labels = ['all', 'attacked', 'projective']
    labels = label_list
    for key in labels:
        x = np.arange(load[1])
        ax.plot(x, acc_avg_dict[key], lw=2, label=key, color=c_dict[key], ls=ls_dict[key])
        ax.fill_between(x, acc_avg_dict[key] - acc_sem_dict[key], acc_avg_dict[key] + acc_sem_dict[key],
                        facecolor=c_dict[key], alpha=0.2)
    # for key in label_list:
    #     ax.hlines(max(acc_avg_dict[key]), 0, load[1] - 1, label=f'{key}:{max(acc_avg_dict[key])}', linestyles='--')
    ax.grid(True, linestyle='dotted', linewidth=1)
    ax.tick_params(labelsize=15)
    ax.set_ylabel('test accuracy (%)', fontsize=20)
    ax.set_xlabel('epoch', fontsize=20)
    ax.set_ylim([10, 80])
    ax.set_yticks(list(np.arange(10, 81, 10)))
    ax.set_xlim([0, load[1] - 1])
    ax.set_xticks(list(np.arange(0, load[1], 10)))
    ax.legend(fontsize=20, loc='upper left')
    if len(seeds) > 1:
        fig.savefig(os.path.join(dir, 'figure', f'{basename}_multiseed_epoch{load[1]}.png'), bbox_inches='tight')
    else:
        fig.savefig(os.path.join(dir, 'figure', f'{basename}_seed{seeds[0]}_epoch{load[1]}.png'), bbox_inches='tight')
    print(f'save{basename}')
