import copy
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client


class clientAVG(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.bit_trans = sum([param.numel() * 32 for (name,param) in self.model.named_parameters()])
        print(self.bit_trans)


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

    def calculate_k(self):
        # 计算对应压缩率的k值,遍历每一个卷积层、
        count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.compress_k[count] = int(param.numel() * (1 - self.compress_rate[count])) + 1
                count += 1




