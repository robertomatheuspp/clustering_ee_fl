#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)



class LocalUpdate(object):
    def __init__(self, device, dataset,
                 idxs, local_bs, logger,
                 idx_client=0, local_ep=10,
                 optimizer="sgd", lr=0.01, verbose=False,
                 seed=0):
        self.logger = logger
        self.verbose = verbose
        self.device = device
        self.local_bs = local_bs
        self.lr = lr
        self.optimizer = optimizer
        self.local_ep = local_ep
        self.seed = seed

        self.g = torch.Generator()
        self.g.manual_seed(self.seed)

        self.idx_client = idx_client

        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs))

        # self.trainloader= self.train_val_test(
        #     dataset, list(idxs))

        # Default criterion set to NLL loss function
        self.criterion = nn.NLLLoss().to(self.device)

    def seed_worker_fn(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs[:int(0.8*len(idxs))]
        # idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        # idxs_test = idxs[int(0.9*len(idxs)):]
        idxs_val = idxs[int(0.8*len(idxs)):]
        idxs_test = idxs_val

        # print (self.local_bs, len(idxs_train))
        if self.local_bs <= 0:
            self.local_bs = int(len(idxs_train))
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)),
                                 # batch_size=int(len(idxs_val)/10),
                                 # num_workers=num_workers,
                                 worker_init_fn=self.seed_worker_fn,
                                 generator=self.g,
                                 shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)),
                                # batch_size=int(len(idxs_test)/10),
                                worker_init_fn=self.seed_worker_fn,
                                generator=self.g,
                                shuffle=False)
        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optimizer for the local updates
        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.lr,
                                        # momentum=3e-4, weight_decay=1.0)
                                        momentum=0.5, weight_decay=0)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr,
                                         weight_decay=1e-4)

        for iter in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.verbose and (batch_idx % 100 == 0):
                    print('| Global Round :{} | Client: {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, self.idx_client, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                if self.logger:
                    self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss


def test_inference(device, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = device
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
