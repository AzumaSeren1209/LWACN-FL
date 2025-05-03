# File: server_avg_mask.py
# Author: Zijun Wang, Ziyuan Feng
# Date: [2025.05.03]
# Description: This module implements a layer-wise adaptive compression strategy
#              based on PFLlib (https://github.com/TsingZ0/PFLlib)
# License: Apache License 2.0 (inherited from PFLlib)
import time
from flcore.clients.client_avg_mask import clientAVGMask
from flcore.servers.serverbase import Server
from flcore.compresser.mask_decoder import DecodeMaskedSparser
from threading import Thread
import random
import copy

class FedAvgMask(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVGMask)
        self.decoder = DecodeMaskedSparser(0,args.compress_bits)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            print("Start Training...")

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()
        print("TransBits = ")
        print(self.clients[0].bit_trans)

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientAVG)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    # Modify the core receive Model function, add a decompression function in it.
    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        # Randomly select a few clients to deactivate
        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        # Record the weights and models of all living clients. It's important to note that this is a simulation method, and there is no real communication going on; instead, the client list is taken directly from the server's list of clients
        # Fetch the corresponding client weights and models
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id) # Save client id
                self.uploaded_weights.append(client.train_samples)  # Saves the number of samples trained by the client, which is used to adjust the weights when weighting and averaging the model parameters
                # Decompress the model parameters
                decompressed_model = copy.deepcopy(self.global_model)
                compressed_model_params = client.compressed_model   # Get client-side compressed parameters

                # Get the compression ratio for each layer
                compress_k = client.compress_k
                count = 0
                for name, param in decompressed_model.named_parameters():
                    # Set the compression ratio
                    self.decoder.set_k(compress_k[count])
                    if param.requires_grad:
                        # Unzip the parameters
                        self.decoder(param,compressed_model_params[count][0],compressed_model_params[count][1])
                    count += 1
                self.uploaded_models.append(decompressed_model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
