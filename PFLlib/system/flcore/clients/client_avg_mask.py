# File: client_avg_mask.py
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


class clientAVGMask(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.compress_rate = [args.min_compress_rate] * sum(1 for _ in self.model.named_parameters())  # 保存所有层的压缩率
        self.min_rate = args.min_compress_rate
        self.max_rate = args.max_compress_rate
        self.compress_k = [0] * sum(1 for _ in self.model.named_parameters())  # 计算出encoder需要使用的k值
        self.calculate_k()
        self.encoder = EncodeMaskedSparser(0, args.compress_bits)

        # Saving a compressed model
        self.compressed_model = [0] * sum(1 for _ in self.model.named_parameters())
        self.bit_trans = 0
        self.compress_bits = args.compress_bits



    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

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
                compressed_weight = self.encoder(param.data)  # compress parameter
                self.compressed_model[count] = compressed_weight  # Put the compressed parameters into the compression model
            count += 1

        # Save Bit Transmission
        self.bit_trans = (
            sum([calculate_bits(len(self.compressed_model[i][1]), self.compress_k[i], self.compress_bits) for i in
                 range(len(self.compressed_model))]))


    def calculate_k(self):
        # Calculate the value of k corresponding to the compression rate, traversing each convolutional layer
        count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.compress_k[count] = int(param.numel() * (1 - self.compress_rate[count])) + 1
                count += 1

def calculate_bits(d, k, b):
    v_bits = k * 32  # The number of bits in the vector v
    mask_bits = d * b  # The number of bits in the mask
    total_bits = v_bits + mask_bits  # Total bits
    return total_bits