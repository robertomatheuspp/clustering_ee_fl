#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
# from torch_geometric import datasets as datasets_3d
from sampling import read_dataset_and_split
import numpy as np



def get_dataset_split(dataset, base_dir="../../data/"):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    apply_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    if dataset == 'mnist':
        base_dir += "mnist/"

        train_dataset = datasets.MNIST(base_dir, transform=apply_transform, train=True, download=True)
        test_dataset  = datasets.MNIST(base_dir, transform=apply_transform, train=False, download=True)


    elif dataset == "cifar10":
        base_dir += "cifar10/"
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(base_dir, transform=apply_transform, train=True, download=True)
        test_dataset = datasets.CIFAR10(base_dir, transform=apply_transform, train=False, download=True)
    # elif dataset == "modelnet": # needs to be pre-processed to make it image format.
        # train_dataset = datasets_3d.ModelNet(data_dir + "modelnet/", train=True)
    else:
        base_dir = "fmnist/"
        train_dataset = datasets.FashionMNIST(base_dir, transform=apply_transform, train=True, download=True)
        test_dataset = datasets.FashionMNIST(base_dir, transform=apply_transform, train=False, download=True)



    user_groups = read_dataset_and_split(train_dataset)

    return train_dataset, test_dataset, user_groups




def count_user2class(group2idx_list, labels, num_classes=10):
    """
    Counts how many samples of each class there is in every group.


    :param group2idx_list: a list containing of length = number of groups. When accessing gth entry, we will have
                           all the indexes referent to that group.
    :param labels: A list that will be indexed by labels[group2idx_list[ g ]] to access the true labels of the gth group.
    :param num_classes:
    :return:
    """

    map_user_class_qt = np.zeros((len(group2idx_list), num_classes))

    for idx_group in range(len(group2idx_list)):
        idxs = group2idx_list[idx_group]
        idxs_train = idxs

        values, counts = np.unique(labels[idxs_train], return_counts=True)
        # print (idx_user, values)

        map_user_class_qt[idx_group][values] = counts

    return map_user_class_qt















def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return

