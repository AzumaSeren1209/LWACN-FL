# File: client_avg_mask_adaptive.py
# Author: Zijun Wang, Ziyuan Feng
# Date: [2025.05.03]
# Description: This module implements a layer-wise adaptive compression strategy
#              based on PFLlib (https://github.com/TsingZ0/PFLlib)
# License: Apache License 2.0 (inherited from PFLlib)
import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from flcore.compresser.mask_encoder import EncodeMaskedSparser

import torch.nn as nn
import copy
from collections import defaultdict
from collections import OrderedDict

class clientAVGMaskAdaptive(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.compress_rate = [args.min_compress_rate] * sum(1 for _ in self.model.named_parameters())  # Save compression rates for all layers
        self.min_rate = args.min_compress_rate
        self.max_rate = args.max_compress_rate
        self.compress_bits = args.compress_bits
        self.adaptive_rate = args.adaptive_rate
        self.compress_k = [0] * sum(1 for _ in self.model.named_parameters())  # Calculate the value of k to be used by the encoder
        self.calculate_k()
        self.encoder = EncodeMaskedSparser(0, args.compress_bits)

        # Saving a compressed model
        self.compressed_model = [0] * sum(1 for _ in self.model.named_parameters())

        # Record the label information entropy
        self.label_counts = defaultdict(int)
        self.total_samples = 0
        # clear the tags
        self.global_round = 0
        # Number of record categories
        self.num_classes = args.num_classes

        # Sliding window to record history loss
        self.sample_step = 5
        self.loss_window = []
        self.max_loss = 0
        self.delta_loss = 0


        # Record the maximum number of parameters in all layers, the total number of layers
        self.max_param_count = max([param.numel() for (name,param) in self.model.named_parameters()])
        self.max_depth = sum(1 for _ in self.model.named_parameters())

        # Poor recording accuracy
        self.acc_list = []
        self.delta_acc = 0

        # Layer Importance Parameter: \alpha,\beta,\gamma,\eta
        self.layer_importance_w = [0.5,0.2,0.1,0.2]

        # noniid
        #  self.mu_w = [1,0.8,0.1,1]
        self.mu_w = [0.5,0.4,0.1,1]

        # iid
        # dself.mu_w = [0.01, 0.01, 1, 1.5]

        self.max_mu = sum(self.mu_w[0:3]) + (self.layer_importance_w[0]+self.layer_importance_w[1]) * (np.e ** (-self.layer_importance_w[2] * self.max_rate)) * (1 + self.layer_importance_w[3])
        self.min_mu = 0

        # Record the average number of bits transmitted
        self.bit_trans = []

        # Record model
        self.model_name = args.model_str





    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        self.global_round += 1
        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Calculate tag information entropy
                # Extract tags and count them
                unique_labels, counts = torch.unique(y, return_counts=True)

                # Update global label distribution
                for label, count in zip(unique_labels, counts):
                    self.label_counts[int(label)] += count
                self.total_samples += y.size(0)

                # Sliding window to record history loss
                with torch.no_grad():
                    if len(self.loss_window) < self.sample_step:
                        self.loss_window.append(loss)
                    else:
                        # Maintain the loss queue and calculate the loss difference
                        loss_history = sum(self.loss_window) / len(self.loss_window)
                        self.loss_window.pop(0)
                        self.loss_window.append(loss)
                        loss_new = sum(self.loss_window) / len(self.loss_window)

                        self.delta_loss = abs(loss_new - loss_history)

                        if self.delta_loss > self.max_loss:
                            self.max_loss = self.delta_loss

                        self.delta_loss = self.delta_loss / self.max_loss  # 归一化

                # 记Record neighboring accuracy differences
                with torch.no_grad():
                    # top5 Accuracy
                    acc = calculate_accuracy(output, y, (1,5))
                    if len(self.acc_list) < 2:
                        self.acc_list.append(acc)
                        if len(self.acc_list) == 2:
                            self.delta_acc = acc[1] - acc[0]
                    else:
                        self.acc_list.pop(0)
                        self.acc_list.append(acc)
                        self.delta_acc = acc[1] - acc[0]

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        # Compression of the trained model
        count = 0
        for (name, param) in self.model.named_parameters():
            # For a given layer, the parameters of the original mod are compressed
            # Setting the encoder parameters
            self.encoder.set_k(self.compress_k[count])
            if param.requires_grad:
                compressed_weight = self.encoder(param.data)  # compression parameter
                self.compressed_model[count] = compressed_weight  # Putting the compressed parameters into the compression model
            count += 1



        # Record the amount of bits transferred by the model
        self.bit_trans.append(sum([calculate_bits(len(self.compressed_model[i][1]), self.compress_k[i], self.compress_bits) for i in range(len(self.compressed_model))]))

    def calculate_k(self):
        # Calculate the value of k corresponding to the compression rate, traversing each convolutional layer
        count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.compress_k[count] = int(param.numel() * (1 - self.compress_rate[count])) + 1
                #print(f"{name}:{param.numel()}")

                count += 1

    def fresh_compress_rate(self,global_model):
        # 首First calculate the cosine similarity between the global and current parameters
        cos_sims = [0] * sum(1 for _ in self.model.named_parameters())
        count = 0
        for (name, param),(_,global_param) in zip(self.model.named_parameters(), global_model.named_parameters()):
            if param.requires_grad:
                cos_sims[count] = safe_cosine_similarity(param.data, global_param.data)
                # normalize
                cos_sims[count] = (cos_sims[count] + 1)/2
            count += 1

        # Then calculate the information entropy based on the labeling statistics of the record.
        # Calculate the current information entropy
        entropy = self.compute_entropy()
        # normalize
        entropy = entropy / np.log(self.num_classes)

        # Get the current delta_loss, which is already maintained at the time of train

        # Computing layer importance
        layer = 1
        alpha,beta,gamma,eta = self.layer_importance_w
        a1,a2,a3,a4 = self.mu_w
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                p_j = param.numel()
                compress_rate = self.compress_rate[layer-1]
                L = ((alpha * (1 - p_j / self.max_param_count) + beta * (layer / self.max_depth)) *
                     (np.e ** (- gamma * compress_rate)) *
                     (1 + eta * (max(0,-self.delta_acc))))
                mu = a1 * (cos_sims[layer-1]) + a2 * (entropy) + a3 * self.delta_loss + a4 * L
                
                mu = (mu / self.max_mu) ** 2
                mu = (mu) / (1 - mu)
                self.compress_rate[layer-1] = self.min_rate - (2 * (self.min_rate - self.max_rate) / (1 + np.e ** (self.adaptive_rate * mu)))
            layer+= 1
        #  Adjust the k value according to the brand new compression rate
        self.calculate_k()

        if self.global_round % 200 == 0:
            self.label_counts.clear()

    def compute_entropy(self):
        counts = torch.tensor(
            list(self.label_counts.values()),
            dtype=torch.float32,
            device=self.device
        )

        # Calculating probability distributions (avoiding division by zero)
        probs = counts / (self.total_samples + 1e-8)

        # Calculate information entropy (only for non-zero probabilities)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        return entropy.item() 


def safe_cosine_similarity(A, B):
    A_flat = torch.cat([a.view(-1) for a in A.values()]) if isinstance(A, dict) else A.view(-1)
    B_flat = torch.cat([b.view(-1) for b in B.values()]) if isinstance(B, dict) else B.view(-1)


    if torch.allclose(A_flat, torch.zeros_like(A_flat)) or torch.allclose(B_flat, torch.zeros_like(B_flat)):
        return 0.0
    A_normalized = A_flat / (torch.norm(A_flat, p=2)) + 1e-10
    B_normalized = B_flat / (torch.norm(B_flat, p=2)) + 1e-10
    return torch.dot(A_normalized, B_normalized).clamp(-1.0, 1.0).item()

def calculate_accuracy(output, y, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = y.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(y.view(1, -1).expand_as(pred))

        accuracies = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            accuracies.append(correct_k.mul_(100.0 / batch_size))
        return accuracies

def calculate_bits(d, k, b):
    v_bits = k * 32  # The number of bits in the vector v
    mask_bits = d * b  # The number of bits in the mask
    total_bits = v_bits + mask_bits  # Total bits
    return total_bits





