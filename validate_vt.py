import torch.distributed as dist
from train_test import fit


def validate_vt(memory, attackers, num_epoch, train_loader, device, optimizer):
    if dist.get_rank() not in attackers:
        # for epoch in range(num_epoch):
        memory['projective_ts']
        model = memory['projective_ts']['restored']
        losses = fit(model, train_loader, device, optimizer)
        _ = fit(memory['local_v']
        return distance
