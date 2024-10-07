#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms



def read_dataset_and_split(dataset):
    """
    Split dataset  according to their label type.

    :param dataset:
    :param num_splits:
    :return:
    """

    labels = np.array(dataset.targets)
    unique_labels = np.sort(np.unique(labels))
    num_splits = len(unique_labels)

    dict_group = {i: np.array([]) for i in range(num_splits)}
    for j in range(len(unique_labels)):
        dict_group[j] = np.where(unique_labels[j] == labels)


    return dict_group

def non_iid_dirichlet_sampling(y_train, n_parties, alpha=0.4, min_require_size = 10):
    """
        Adapted from https://github.com/Xtra-Computing/NIID-Bench

        The idea here is that 'y_train' will be devided into 'n_parties' according to a Dirichlet distribution
        with parameter 'beta'. Each of the 'n_parties' has at least 'min_require_size' samples.

        This can be used in combination with 'get_dataset_split' by:

            from sampling import non_iid_dirichlet_sampling
            from utils import get_dataset_split, count_user2class
            import seaborn as sns

            from sampling import non_iid_dirichlet_sampling
            from utils import get_dataset_split, count_user2class

            num_clients_per_part = 10

            # loading data
            dataset_name = "cifar10"
            train_dataset, test_dataset, split2dataIdx  = get_dataset_split(dataset=dataset_name)

            # We will consider two clusters of data, one with classes [0,4] and another with [5,9]
            map_groups2class_split = [
                np.squeeze( np.hstack([split2dataIdx[idx] for idx in [0,1,2,3,4]]) ),
                np.squeeze( np.hstack([split2dataIdx[idx] for idx in [5,6,7,8,9]]) ),
            ]

            clusteringSol = [[]]*len(map_groups2class_split)
            clusteringSol[0] = list(np.arange(num_clients_per_part))
            clusteringSol[1] = (np.arange(num_clients_per_part) + num_clients_per_part).tolist()

            labels = np.array(train_dataset.targets)
            user2dataIdxs = []
            for idx, cur in enumerate(map_groups2class_split):
                original_idx = np.squeeze(map_groups2class_split[idx])
                cur_out = non_iid_dirichlet_sampling(labels[original_idx], n_parties=num_clients_per_part)
                for elem in cur_out.values():
                    user2dataIdxs.append(original_idx[elem])


            # PLOTTING
            map_user_class_qt = count_user2class(user2dataIdxs, labels, num_classes=10)

            # plotting results
            s = sns.heatmap(map_user_class_qt / map_user_class_qt.sum(0))
            # summing the rows will result in 1.
            s.set(xlabel='Class idx', ylabel='Client idx', title="Percentage of each class per user: Col sums to 1");


    :param y_train: array containing the labels of the dataset (could also be a subset)
    :param n_parties: number of partitions in which to split the data
    :param alpha: parameter of the Dirichlet distribution
    :param min_require_size: size of the smallest number of elements in each partition

    :return: a dictionary with 'n_parties' entries. Each entry contains
             a list indexing the samples that belong to the jth partition.
             For example, y_label[ dataidx_map[j] ] will contain all the labels of the jth element.
    """
    y_train = np.squeeze(y_train)
    n_train = y_train.shape[0]

    # auxiliary variable not to avoid infinity loops
    tmp = 0


    tmp_min_size = 0
    while tmp_min_size < min_require_size and tmp < 1e4:
        tmp += 1
        idx_batch = [[] for _ in range(n_parties)]
        for cur_label in np.unique(y_train):
            # choosing only elements that contain the kth class
            idx_k = np.where(y_train == cur_label)[0]
            np.random.shuffle(idx_k)

            # randomly draw proportions to each partition
            proportions = np.random.dirichlet(np.repeat(alpha, n_parties))


            # Balance these proportions
            proportions = np.array([p * (len(idx_j) < n_train / n_parties)
                                    for p, idx_j in zip(proportions, idx_batch)])

            # make it from 0-1
            proportions = proportions / proportions.sum()

            # make it in quantitative number
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

            idx_batch = [idx_j + idx.tolist()
                         for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

            # smallest number of elements (used in th while loop)
            tmp_min_size = min([len(idx_j) for idx_j in idx_batch])

    dataidx_map = {}

    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        dataidx_map[j] = idx_batch[j]

    return dataidx_map

def select_idx_users_f(num_users, clusteringSol, type="classical", qty_to_pick=-1):
    """
        Generate different ways of selecting users.
        This is going to be applied in the FL process.

        There are three acceted ways:

        1) Classical Centralized FL: selects all the clients
        2) Randomly picking 'qty_to_pick' clients.
        3) diff_together: Randomly picking 'qty_to_pick' client from the same group
        4) similar_together (ferenda's): Randomly picking 1 client from each of the G groups

        If 'qty_to_pick' is not defined, it is set to the number of clusters.
        If type="classical" then 'qty_to_pick' will always gonna be 'num_users'

        'clusteringSol' stores the solution of the clustering.
        Index is the cluster number and it contains the clients of that specific cluster. For instance,

            clusteringSol = [[]]*num_splits
            clusteringSol[0] = [0, 2]
            clusteringSol[1] = [1, 3]


    :param num_users:
    :param clusteringSol:
    :param type:
    :param qty_to_pick:
    :return:
    """

    # clusteringSol = np.array(clusteringSol).squeeze()
    num_groups = len(clusteringSol)

    if qty_to_pick < 1:
        qty_to_pick = num_groups

    if type == "centralized":
        idxs_users = np.arange(num_users)
        return np.array([0])
    elif type == "random" or type == "FedCor":
        idxs_users = np.random.choice(range(0, num_users), qty_to_pick, replace=False)
    elif type == "RepulsiveClust":
        # how many groups to pick in order to have at least "qty_to_pick" users

        qty_user_per_group = num_users / num_groups
        min_nb_groups = int(np.ceil(qty_to_pick / qty_user_per_group))

        # randomly pick a cluster
        g = np.random.choice(range(0, num_groups), min_nb_groups, replace=False)

        # all elements that belong to this group
        idxs_users = np.array(clusteringSol[g]).reshape(-1).squeeze()

        # select only a few
        selected_idx = np.random.choice(range(0, len(idxs_users)), qty_to_pick, replace=False)
        idxs_users = idxs_users[selected_idx]

    elif type == "SimilarClust":

        # how many users to pick per group.
        qty_user_per_group_pick = max(int( np.ceil(qty_to_pick / num_groups) ), 1)

        # This replicates what Fernanda was doing
        idxs_users = []
        for g in range(num_groups):
            cur_idxs_users = np.array(clusteringSol[g])[np.newaxis].squeeze().tolist()#[np.newaxis]
            # print(cur_idxs_users)
            if isinstance(cur_idxs_users, list):
                if len(cur_idxs_users) <= 0:
                    continue
            else:
                cur_idxs_users = [cur_idxs_users]
            selected_idx = np.random.choice(range(0, len(cur_idxs_users)), qty_user_per_group_pick, replace=False)

            cur_idxs_users = np.array(cur_idxs_users)
            idxs_users.append(cur_idxs_users[selected_idx])

        idxs_users = np.squeeze(np.reshape(idxs_users, -1))
        idxs_users = np.random.choice(idxs_users, min(qty_to_pick, len(idxs_users)), replace=False)

    idxs_users = np.squeeze(np.reshape(idxs_users, -1))
    np.random.shuffle(idxs_users)

    return idxs_users

if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))

