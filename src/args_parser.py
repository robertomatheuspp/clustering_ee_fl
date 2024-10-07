#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Dataset non-iid scenario arguments
    parser.add_argument('--base_dir_dataset', type=str, default='../data/',
                        help="Base directory where to store downloaded datasets.")
    parser.add_argument('--base_dir_outputs', type=str, default='../output/',
                        help="Base directory where to store logs and csv outputs.")

    parser.add_argument('--dataset_name', type=str, default='mnist',
                        choices=["mnist", "fmnist", "cifar10"],
                        help="name of dataset")
    parser.add_argument('--num_dataset_partition', type=int, default=2,
                        help="number of partitions in the dataset, e.g., if equal to two then 0-4 and 5-9.")

    parser.add_argument('--gpu', default=True,
                        help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help="type of optimizer")
    parser.add_argument('--dirichlet_alpha', type=float, default=1,
                        help="use Dirichlet_noniid sampling, set the alpha of Dir here")
    parser.add_argument('--verbose', type=int, default=True,
                        help='verbose')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (we will consider 0, 1, 2, 3, 4)')

    # Clustering of users arguments
    parser.add_argument('--method_type', type=str, default="FedCor",
                        choices=["centralized", "random", "SimilarClust", "RepulsiveClust", "FedCor", "CovClust", "Power-d"],
                        help='Method to use for training')

    parser.add_argument('--num_groups_users', type=int, default=-1,
                        help='number of groups in which the users should be clustered. By default this is equal to "num_dataset_partition" argument. ')
    parser.add_argument('--num_user_per_partition', type=int, default=5,
                        help='number of user in each group. Total number of users will be num_user_per_partition*num_dataset_partition')
    parser.add_argument('--num_user_to_pick', type=int, default=-1,
                        help='number of users to pick at each communication round. By default this is equal to "num_dataset_partition" argument. ')


    # Federated learning arguments
    parser.add_argument('--epochs', type=int, default=1,
                        help="number of rounds of training")
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=64,#10,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')


    # GPR arguments
    parser.add_argument('--fedcor_warmup', type=int, default=10,
                        help='length of warm up phase for GP')
    parser.add_argument('--fedcor_gpr_begin', type=int, default=0,
                        help='the round begin to sample and train GP')
    parser.add_argument('--fedcor_group_size', type=int, default=10,
                        help='length of history round to sample for GP, equal to M in paper')
    parser.add_argument('--fedcor_GPR_interval', type=int, default=5,
                        help='interval of sampling and training of GP, namely, Delta t')
    parser.add_argument('--fedcor_GPR_gamma', type=float, default=0.8,
                        help='gamma for training GP')
    parser.add_argument('--fedcor_GPR_Epoch', type=int, default=100,
                        help='number of optimization iterations of GP')
    parser.add_argument('--fedcor_update_mean', action='store_true',
                        help="Whether to update the mean of the GPR")
    parser.add_argument('--fedcor_poly_norm', type=int, default=0,
                        help='whether to normalize the poly kernel, set 1 to normalize')
    parser.add_argument('--fedcor_dimension', type=int, default=15,
                        help='dimension of embedding in GP')
    parser.add_argument('--fedcor_train_method', type=str, default='MML',
                        help='method of training GP (MML,LOO)')
    parser.add_argument('--fedcor_discount', type=float, default=0.9,
                        help='annealing coefficient, i.e., beta in paper')
    parser.add_argument('--fedcor_epsilon_greedy', type=float, default=0.0,
                        help='use epsilon-greedy in FedGP, set epsilon here')
    parser.add_argument('--fedcor_dynamic_C', action='store_true',
                        help='use dynamic GP clients selection')
    parser.add_argument('--fedcor_dynamic_TH', type=float, default=0.0,
                        help='dynamic selection threshold')

    # Power-d arguments
    parser.add_argument('--power_d_val',type = int,default = 30,
                        help='d in Pow-d selection')

    # Active Federated Learning arguments
    parser.add_argument('--afl_alpha1',type = float,default=0.75,
                        help = 'alpha_1 in ALF')
    parser.add_argument('--afl_alpha2',type = float,default=0.01,
                        help = 'alpha_2 in AFL')
    parser.add_argument('--afl_alpha3',type = float,default=0.1,
                        help='alpha_3 in AFL')

    # model arguments
    parser.add_argument('--model', type=str, default='cnn',
                        help='model name (mlp, cnn and resnet)')
    # parser.add_argument('--kernel_sizes', type=int, default=[5,5],nargs="*",
    #                     help='kernel size in each convolutional layer')
    # parser.add_argument('--num_filters', type=int, default=[32,64],nargs = "*",
    #                     help="number of filters in each convolutional layer.")
    # parser.add_argument('--padding', action='store_true',
    #                     help='use padding in each convolutional layer')
    # parser.add_argument('--mlp_layers',type= int,default=[64,],nargs="*",
    #                     help="numbers of dimensions of each hidden layer in MLP, or fc layers in CNN")
    # parser.add_argument('--depth',type = int,default = 20,
    #                     help = "The depth of ResNet. Only valid when model is resnet")

    args = parser.parse_args()

    args = vars(args)

    if args["num_groups_users"] == -1:
        args["num_groups_users"] = args["num_dataset_partition"]

    if args["num_user_to_pick"] == -1:
        args["num_user_to_pick"] = args["num_dataset_partition"]

    return args



if __name__ == '__main__':
    args = args_parser()
    print(args)
