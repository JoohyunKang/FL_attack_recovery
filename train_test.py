import torch.distributed as dist
import torch
import numpy as np
import torch.nn as nn

def train(model, train_loader, device, optimizer, group=None):
    world_size = dist.get_world_size()
    if group:
        pg = dist.new_group(group)
    losses = fit(model, train_loader, device, optimizer)
    with torch.no_grad():
        for param in model.parameters():
            if group:
                param.data /= len(group)
                dist.all_reduce(param.data, op=dist.ReduceOp.SUM, group=pg)
            else:
                param.data /= world_size
                dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
    avg_loss = sum(losses)/len(losses)
    return avg_loss


def fit(model, train_loader, device, optimizer):
    model.train()
    if optimizer[0] == 'Adam':
        optim = torch.optim.Adam(model.parameters(), lr=optimizer[1])
    elif optimizer[0] == 'SGD':
        optim = torch.optim.SGD(model.parameters(), lr=optimizer[1])
    elif optimizer[0] == 'SGDm':
        optim = torch.optim.SGD(model.parameters(), lr=optimizer[1], momentum=optimizer[2])
    losses = []
    for i, data in enumerate(train_loader):
        # if i % fold == epoch % fold:
        image = data[0].type(torch.FloatTensor).to(device)
        label = data[1].type(torch.LongTensor).to(device)
        pred_label = model(image)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred_label, label)
        losses.append(loss.item())
        optim.zero_grad()
        loss.backward()
        optim.step()
    return losses


def eval(model, test_loader, device):
    model.eval()
    pred_labels = []
    real_labels = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            image = data[0].type(torch.FloatTensor).to(device)
            label = data[1].type(torch.LongTensor).to(device)
            real_labels += list(label.cpu().detach().numpy())
            pred_label = model(image)
            pred_label = list(pred_label.cpu().detach().numpy())
            pred_labels += pred_label
    real_labels = np.array(real_labels)
    pred_labels = np.array(pred_labels)
    pred_labels = pred_labels.argmax(axis=1)
    acc = sum(real_labels == pred_labels) / len(real_labels) * 100
    return acc
