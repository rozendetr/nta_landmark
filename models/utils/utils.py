
import torch
import os

def load_net(path_load):
    """
    Load net from file ckpt
    :param net:
    :param path_load:
    :return:
    """
    print('==> Loaded net..')
    checkpoint = torch.load(path_load)
    state_dict = checkpoint['net']
    accs = checkpoint['accs']
    start_epoch = checkpoint['epoch']
    return state_dict, accs, start_epoch


def save_net(path_save, state_dict, accs, epoch):
    """
    Save met to file ckpt
    :param net:
    :param path_save:
    :param acc_mean:
    :param epoch:
    :return:
    """
    state = {'net': state_dict,
             'accs': accs,
             'epoch': epoch}
    torch.save(state, path_save)


