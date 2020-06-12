import torch.distributed as dist
import torch
import copy
from train_test import fit
import numpy as np
import random

def attack_and_restore(model_global, train_loader, device, epoch,
                       method_a, attackers, memory, power, methods_r,
                       optimizer, clamp, seed, attenuation=1):
    # model_global.train()
    random.seed(seed)
    world_size = dist.get_world_size()
    group_a = dist.new_group(attackers)
    attacker_size = len(attackers)
    victim_size = world_size - attacker_size
    # Attack
    if method_a == 'median':
        model = copy.deepcopy(model_global)
        # Train normally and record
        losses = fit(model, train_loader, device, optimizer)
        memory['local_a'].append(copy.deepcopy(model))
        memory['attack'].append(copy.deepcopy(model))
        memory['dist_local_a'].append([])
        with torch.no_grad():
            for param_global, param_local_a, param_attack in zip(model_global.parameters(),
                                                                 memory['local_a'][-1].parameters(),
                                                                 memory['attack'][-1].parameters()):
                # Attack that can maximize harm while followind similar distribution
                # Depart from expected mean by power * std and opposite direction from update direction
                param_update = param_local_a.data - param_global.data
                list_param_update = [torch.zeros_like(param_update) for _ in range(attacker_size)]
                dist.all_gather(list_param_update, param_update, group=group_a)
                tensor_param_update = torch.stack(list_param_update)  # Convert tensor_list to tensor
                mean = tensor_param_update.mean(dim=0)
                std = tensor_param_update.std(dim=0)
                sign = mean.sign()
                memory['dist_local_a'][epoch].append(std)
                param_global.data += mean - sign * std * (power - 0.5 + random.random())
                # Local model for attack
                # param_attack.data = param_global.data / attacker_size
                param_attack.data = param_global.data / attacker_size
                dist.all_reduce(param_attack.data, op=dist.ReduceOp.SUM, group=group_a)
                # Global model
                param_global.data /= world_size
                dist.all_reduce(param_global.data, op=dist.ReduceOp.SUM)
                # Attackers' local model
                param_local_a.data /= attacker_size
                dist.all_reduce(param_local_a.data, op=dist.ReduceOp.SUM, group=group_a)
        memory['global'].append(copy.deepcopy(model_global))
        # Inference
        memory['local_v'].append(copy.deepcopy(memory['global'][0]))
        with torch.no_grad():
            for param_local_v, param_global, param_attack in zip(memory['local_v'][-1].parameters(),
                                                                 memory['global'][-1].parameters(),
                                                                 memory['attack'][-1].parameters()):
                param_local_v.data = (world_size * param_global.data - attacker_size * param_attack.data) / victim_size
    avg_loss = sum(losses) / len(losses)
    # Restore
    if 'cumulative' in methods_r:
        # Initialization of compensation term
        if epoch == 0:
            memory['cumulative']['compen'].append(copy.deepcopy(memory['global'][0]))
            with torch.no_grad():
                for param in memory['cumulative']['compen'][epoch].parameters():
                    param.data = torch.zeros_like(param.data)
        else:
            memory['cumulative']['compen'].append(copy.deepcopy(memory['cumulative']['compen'][epoch-1]))
        # Restore
        memory['cumulative']['restored'].append(copy.deepcopy(memory['global'][epoch]))
        with torch.no_grad():
            for param_compen, param_restored, param_local_a, param_attack \
                    in zip(memory['cumulative']['compen'][epoch].parameters(),
                           memory['cumulative']['restored'][epoch].parameters(),
                           memory['local_a'][epoch].parameters(),
                           memory['attack'][epoch].parameters()):
                param_compen.data *= 1 - attenuation
                param_compen.data += param_local_a.data - param_attack.data
                param_compen.data *= attacker_size / world_size
                param_restored.data += param_compen.data
    if 'translative' in methods_r:
        # Train
        if epoch == 0:
            memory['translative']['local_a'].append(copy.deepcopy(memory['local_a'][0]))
            memory['translative']['local_vt'].append(copy.deepcopy(memory['local_v'][0]))
        else:
            memory['translative']['local_a'].append(copy.deepcopy(memory['translative']['restored'][epoch-1]))
            _ = fit(memory['translative']['local_a'][epoch], train_loader, device, optimizer)
            with torch.no_grad():
                for param in memory['translative']['local_a'][epoch].parameters():
                    param.data /= attacker_size
                    dist.all_reduce(param.data, op=dist.ReduceOp.SUM, group=group_a)
            # Translation
            memory['translative']['local_vt'].append(copy.deepcopy(memory['global'][0]))
            with torch.no_grad():
                for param_local_vt, param_local_v, param_restored, param_global in \
                        zip(memory['translative']['local_vt'][epoch].parameters(),
                            memory['local_v'][epoch].parameters(),
                            memory['translative']['restored'][epoch-1].parameters(),
                            memory['global'][epoch-1].parameters()):
                    param_local_vt.data = param_local_v.data + param_restored.data - param_global.data
        # Restore
        memory['translative']['restored'].append(copy.deepcopy(memory['global'][0]))
        with torch.no_grad():
            for param_restored, param_local_a, param_local_vt in zip(memory['translative']['restored'][epoch].parameters(),
                                                                     memory['translative']['local_a'][epoch].parameters(),
                                                                     memory['translative']['local_vt'][epoch].parameters()):
                # param_restored.data = (attacker_size * cut_a * param_local_a.data +
                #                        victim_size * param_local_vt.data) / (attacker_size * cut_a + victim_size)
                param_restored.data = (attacker_size * param_local_a.data +
                                       victim_size * param_local_vt.data) / world_size
    if 'projective_t' in methods_r:
        # Train
        if epoch == 0:
            memory['projective_t']['local_a'].append(copy.deepcopy(memory['local_a'][0]))
            memory['projective_t']['local_vt'].append(copy.deepcopy(memory['local_v'][0]))
        else:
            memory['projective_t']['local_a'].append(copy.deepcopy(memory['projective_t']['restored'][epoch-1]))
            _ = fit(memory['projective_t']['local_a'][epoch], train_loader, device, optimizer)
            with torch.no_grad():
                for param in memory['projective_t']['local_a'][epoch].parameters():
                    param.data /= attacker_size
                    dist.all_reduce(param.data, op=dist.ReduceOp.SUM, group=group_a)
            # Translation
            memory['projective_t']['local_vt'].append(copy.deepcopy(memory['global'][0]))
            with torch.no_grad():
                for param_local_vt, param_local_v, param_local_a, param_local_a0 in \
                        zip(memory['projective_t']['local_vt'][epoch].parameters(),
                            memory['local_v'][epoch].parameters(),
                            memory['projective_t']['local_a'][epoch].parameters(),
                            memory['local_a'][epoch].parameters()):
                    param_local_vt.data = param_local_v.data + param_local_a.data - param_local_a0.data
        # Restore
        memory['projective_t']['restored'].append(copy.deepcopy(memory['global'][0]))
        with torch.no_grad():
            for param_restored, param_local_a, param_local_vt in zip(memory['projective_t']['restored'][epoch].parameters(),
                                                                     memory['projective_t']['local_a'][epoch].parameters(),
                                                                     memory['projective_t']['local_vt'][epoch].parameters()):
                # param_restored.data = (attacker_size * cut_a * param_local_a.data +
                #                        victim_size * param_local_vt.data) / (attacker_size* cut_a + victim_size)
                param_restored.data = (attacker_size * param_local_a.data +
                                       victim_size * param_local_vt.data) / world_size
    if 'projective' in methods_r:
        # Train
        if epoch == 0:
            memory['projective']['local_a'].append(copy.deepcopy(memory['local_a'][0]))
            memory['projective']['local_vt'].append(copy.deepcopy(memory['local_v'][0]))
        else:
            memory['projective']['local_a'].append(copy.deepcopy(memory['projective']['restored'][epoch-1]))
            _ = fit(memory['projective']['local_a'][epoch], train_loader, device, optimizer)
            # Projection
            memory['projective']['local_vt'].append(copy.deepcopy(memory['global'][0]))
            count = 0
            with torch.no_grad():
                for param_local_vt, param_local_v, param_local_a, param_local_a0, param_restored in \
                        zip(memory['projective']['local_vt'][epoch].parameters(),
                            memory['local_v'][epoch].parameters(),
                            memory['projective']['local_a'][epoch].parameters(),
                            memory['local_a'][epoch].parameters(),
                            memory['projective']['restored'][epoch-1].parameters()):
                    # Calculate standart deviation of local_a on restored global model
                    param_update = param_local_a.data - param_restored.data
                    list_param_update = [torch.zeros_like(param_update) for _ in range(attacker_size)]
                    dist.all_gather(list_param_update, param_update, group=group_a)
                    tensor_param_update = torch.stack(list_param_update)  # Convert tensor_list to tensor
                    std = tensor_param_update.std(dim=0)
                    # if dist.get_rank() == 0:
                    # print(f'---------------------count:{count}---------------')
                    # print(f'std:{std}')
                    # print(f'memory_std_ratio:{memory["dist_local_a"][epoch][count]}')
                    std_ratio = std / memory['dist_local_a'][epoch][count]
                    cond = torch.isnan(std_ratio) | torch.isinf(std_ratio) | torch.eq(std_ratio, torch.zeros_like(std_ratio))
                    std_ratio[cond] = 0
                    # std_ratio[] = 0
                    # if dist.get_rank() == 0:
                    #     print(f'cond:{cond}')
                    # std_ratio = ~cond * torch.clamp(std_ratio, 0,  clamp)
                    std_ratio = torch.clamp(std_ratio, 0,  clamp)
                    # if dist.get_rank() == 0:
                    #     print(f'std_ratio:{std_ratio}')
                    # Record average of local_a
                    param_local_a.data /= attacker_size
                    dist.all_reduce(param_local_a.data, op=dist.ReduceOp.SUM, group=group_a)
                    param_local_vt.data = param_local_a.data + (param_local_v.data - param_local_a0.data) * std_ratio
                    # param_local_vt.data = param_restored.data + (param_local_v.data - param_global.data) + \
                    #                       (param_local_a.data - param_restored.data) - (param_local_a0.data - param_global.data)
                    count += 1
                # avg_std_ratio = 0
                # for cnt in range(count):
                #     avg_std_ratio -= avg_std_ratio / (cnt + 1)
                #     avg_std_ratio += torch.mean(memory['dist_local_a'][epoch][cnt]) / (cnt + 1)
                #     print(f'avg_std_ratio:{avg_std_ratio}')
                # for param_local_vt, param_local_v, param_local_a, param_local_a0 in \
                #         zip(memory['projective']['local_vt'][epoch].parameters(),
                #             memory['local_v'][epoch].parameters(),
                #             memory['projective']['local_a'][epoch].parameters(),
                #             memory['local_a'][epoch].parameters()):
                #     # Additive mapping with scaling
                #     param_local_vt.data = param_local_a.data + (param_local_v.data - param_local_a0.data) * avg_std_ratio
        # Restore
        memory['projective']['restored'].append(copy.deepcopy(memory['global'][0]))
        with torch.no_grad():
            for param_restored, param_local_a, param_local_vt in zip(memory['projective']['restored'][epoch].parameters(),
                                                                     memory['projective']['local_a'][epoch].parameters(),
                                                                     memory['projective']['local_vt'][epoch].parameters()):
                # param_restored.data = (attacker_size * cut_a * param_local_a.data +
                #                        victim_size * param_local_vt.data) / (attacker_size * cut_a + victim_size)
                param_restored.data = (attacker_size * param_local_a.data +
                                       victim_size * param_local_vt.data) / world_size
    if 'projective_r' in methods_r:
        # Train
        if epoch == 0:
            memory['projective_r']['local_a'].append(copy.deepcopy(memory['local_a'][0]))
            memory['projective_r']['local_vt'].append(copy.deepcopy(memory['local_v'][0]))
        else:
            memory['projective_r']['local_a'].append(copy.deepcopy(memory['projective_r']['restored'][epoch-1]))
            _ = fit(memory['projective_r']['local_a'][epoch], train_loader, device, optimizer)
            with torch.no_grad():
                for param in memory['projective_r']['local_a'][epoch].parameters():
                    param.data /= attacker_size
                    dist.all_reduce(param.data, op=dist.ReduceOp.SUM, group=group_a)
            # Projection
            memory['projective_r']['local_vt'].append(copy.deepcopy(memory['global'][0]))
            with torch.no_grad():
                for param_local_vt, param_local_v, param_local_a, param_local_a0 in \
                        zip(memory['projective_r']['local_vt'][epoch].parameters(),
                            memory['local_v'][epoch].parameters(),
                            memory['projective_r']['local_a'][epoch].parameters(),
                            memory['local_a'][epoch].parameters()):
                    v1 = torch.flatten(copy.deepcopy(param_local_v.data)).cpu().numpy()
                    v2 = torch.flatten(copy.deepcopy(param_local_a.data)).cpu().numpy()
                    v3 = torch.flatten(copy.deepcopy(param_local_a0.data)).cpu().numpy()
                    # Unit vector
                    u1 = v1 / np.linalg.norm(v1)
                    u2 = v2 / np.linalg.norm(v2)
                    # Get angle between two vectors
                    a = np.arccos(np.clip(np.dot(u1, u2), -1.0, 1.0))
                    # Get rotation matrix using two vectors and angle
                    # Affine containing delta_(local_a) and delta_(local_v)
                    # Gram-Schmidt orthogonalization
                    n1 = v1 / np.linalg.norm(v1)
                    v2 = v2 - np.dot(n1, v2) * n1
                    n2 = v2 / np.linalg.norm(v2)
                    I = np.identity(len(n1))
                    R = I + (np.outer(n2, n1) - np.outer(n1, n2)) * np.sin(a) + \
                        (np.outer(n1, n1) + np.outer(n2, n2)) * (np.cos(a) - 1)
                    # Rotation and scaling
                    u1_t = np.matmul(R, u1)
                    v1_t = u1_t * np.linalg.norm(v2) / np.linalg.norm(v3)
                    param_local_vt.data = torch.reshape(torch.from_numpy(v1_t), param_local_v.data.size()).to(device)
            # with torch.no_grad():
            #     for param_local_vt, param_restored, param_local_v, param_global, param_local_a, param_local_a0 in \
            #             zip(memory['projective_r']['local_vt'][epoch].parameters(),
            #                 memory['projective_r']['restored'][epoch-1].parameters(),
            #                 memory['local_v'][epoch].parameters(),
            #                 memory['global'][epoch-1].parameters(),
            #                 memory['projective_r']['local_a'][epoch].parameters(),
            #                 memory['local_a'][epoch].parameters()):
            #         param_local_vt.data = param_restored.data + (param_local_v.data - param_global.data) * \
            #                               (param_local_a.data - param_restored.data) / (param_local_a0.data - param_global.data)
        # Restore
        memory['projective_r']['restored'].append(copy.deepcopy(memory['global'][0]))
        with torch.no_grad():
            for param_restored, param_local_a, param_local_vt in zip(memory['projective_r']['restored'][epoch].parameters(),
                                                                     memory['projective_r']['local_a'][epoch].parameters(),
                                                                     memory['projective_r']['local_vt'][epoch].parameters()):
                # param_restored.data = (attacker_size * cut_a * param_local_a.data +
                #                        victim_size * param_local_vt.data) / (attacker_size * cut_a + victim_size)
                param_restored.data = (attacker_size * param_local_a.data +
                                       victim_size * param_local_vt.data) / world_size
    return avg_loss, memory
