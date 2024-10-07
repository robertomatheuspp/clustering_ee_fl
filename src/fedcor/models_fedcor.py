#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import abc
import math


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden=[64, 30], dim_out=10):
        super(MLP, self).__init__()
        self.dim_in = dim_in
        self.layers = []
        self.relus = []
        self.dropouts = []

        if len(dim_hidden) > 0:
            self.layers.append(nn.Linear(dim_in, dim_hidden[0]))
            self.relus.append(nn.ReLU())
            self.dropouts.append(nn.Dropout())
            for n in range(len(dim_hidden) - 1):
                self.layers.append(nn.Linear(dim_hidden[n], dim_hidden[n + 1]))
                self.relus.append(nn.ReLU())
                self.dropouts.append(nn.Dropout())
            self.layers.append(nn.Linear(dim_hidden[-1], dim_out))
        else:
            # logistic regression
            self.layers.append(nn.Linear(dim_in, dim_out))

        self.layers = nn.ModuleList(self.layers)
        self.relus = nn.ModuleList(self.relus)
        self.dropouts = nn.ModuleList(self.dropouts)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
        for n in range(len(self.relus)):
            x = self.layers[n](x)
            x = self.dropouts[n](x)
            x = self.relus[n](x)
        x = self.layers[-1](x)
        return F.log_softmax(x, dim=1)


class NaiveCNN(nn.Module):
    def __init__(self, input_shape=[3, 32, 32], num_classes=10, final_pool=True):
        super(NaiveCNN, self).__init__()
        self.convs = []
        self.fcs = []
        self.final_pool = final_pool
        num_filters = [32, 64, 64 ]
        kernel_sizes = [3, 3, 3]
        padding = False
        mlp_layers = [64]
        if len(kernel_sizes) < len(num_filters):
            exlist = [kernel_sizes[-1] for i in range(len(num_filters) - len(kernel_sizes))]
            kernel_sizes.extend(exlist)
        elif len(kernel_sizes) > len(num_filters):
            exlist = [num_filters[-1] for i in range(len(kernel_sizes) - len(num_filters))]
            num_filters.extend(exlist)
        output_shape = np.array(input_shape)
        for ksize in kernel_sizes[:-1] if not final_pool else kernel_sizes:
            if padding:
                pad = ksize // 2
                output_shape[1:] = (output_shape[1:] + 2 * pad - ksize - 1) // 2 + 1
            else:
                output_shape[1:] = (output_shape[1:] - ksize - 1) // 2 + 1
        if not final_pool:
            if padding:
                pad = kernel_sizes[-1] // 2
                output_shape[1:] = output_shape[1:] + 2 * pad - kernel_sizes[-1] + 1
            else:
                output_shape[1:] = output_shape[1:] - kernel_sizes[-1] + 1
        output_shape[0] = num_filters[-1]
        conv_out_length = output_shape[0] * output_shape[1] * output_shape[2]

        self.convs.append(nn.Conv2d(input_shape[0], num_filters[0], kernel_size=kernel_sizes[0],
                                    padding=kernel_sizes[0] // 2 if padding else 0))
        for n in range(len(num_filters) - 1):
            self.convs.append(
                nn.Conv2d(num_filters[n], num_filters[n + 1], kernel_size=kernel_sizes[n + 1],
                          padding=kernel_sizes[n + 1] // 2 if padding else 0))
        # self.conv2_drop = nn.Dropout2d()
        self.fcs.append(nn.Linear(conv_out_length, mlp_layers[0]))
        for n in range(len(mlp_layers) - 1):
            self.fcs.append(nn.Linear(mlp_layers[n], mlp_layers[n + 1]))
        self.fcs.append(nn.Linear(mlp_layers[-1], num_classes))

        self.convs = nn.ModuleList(self.convs)
        self.fcs = nn.ModuleList(self.fcs)

    def forward(self, x):
        for n in range(len(self.convs) - 1 if not self.final_pool else len(self.convs)):
            x = F.relu(F.max_pool2d(self.convs[n](x), 2))
        if not self.final_pool:
            x = F.relu(self.convs[-1](x))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        for n in range(len(self.fcs) - 1):
            x = F.relu(self.fcs[n](x))
        x = self.fcs[-1](x)
        return F.log_softmax(x, dim=1)


