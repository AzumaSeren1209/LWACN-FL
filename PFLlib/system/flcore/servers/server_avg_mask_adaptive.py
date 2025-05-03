import time
from flcore.clients.client_avg_mask_adaptive import clientAVGMaskAdaptive
from flcore.servers.serverbase import Server
from flcore.compresser.mask_decoder import DecodeMaskedSparser
from threading import Thread
import torch
import random
import copy
import os

class FedAvgMaskAdaptive(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientAVGMaskAdaptive)
        self.decoder = DecodeMaskedSparser(0,args.compress_bits)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        self.save_name = args.save_name


    def train(self):
        r = 0
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

            print(f"-------------Freshing_compress_rate-------------")
            self.refresh_clients_compress_rate()    # Updated client compression rate




            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

            if i % 10000 == 0:
                self.save_results(i)
                self.save_global_model(i)
                r = i
                # Calculate the average transmitted bits
                bit_trans = []
                for client in self.clients:
                    bit_trans.append(torch.mean(torch.tensor(client.bit_trans).float()))

                print(f"Client bit trans: {torch.mean(torch.tensor(bit_trans))}")
                self.save_bit_trans(torch.mean(torch.tensor((bit_trans))))


        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        # Calculate the average transmitted bits
        bit_trans = []
        for client in self.clients:
            bit_trans.append(torch.mean(torch.tensor(client.bit_trans).float()))

        print(f"Client bit trans: {torch.mean(torch.tensor(bit_trans))}")
        self.save_bit_trans(torch.mean(torch.tensor((bit_trans))))



        self.save_results(-114514)
        self.save_global_model(-114514)

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

        # Record the weights and models of all living clients.It's important to note that this is a simulation method, and there is no actual communication going on; instead, it's directly from the server's list of clients
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
                self.uploaded_weights.append(client.train_samples)  # Save the number of samples trained by the client for adjusting the weights when weighted averaging the model parameters
                # Decompress the model parameters
                decompressed_model = copy.deepcopy(self.global_model)
                compressed_model_params = client.compressed_model   # Get client-side compressed parameters

                # Get the compression rate for each layer
                compress_k = client.compress_k

                count = 0
                for name, param in decompressed_model.named_parameters():
                    # Set the compression rate
                    self.decoder.set_k(compress_k[count])
                    if param.requires_grad:
                        # Unzip the parameters
                        self.decoder(param,compressed_model_params[count][0],compressed_model_params[count][1])
                    count += 1

                self.uploaded_models.append(decompressed_model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def refresh_clients_compress_rate(self):
        for client in self.selected_clients:
            client.fresh_compress_rate(self.global_model)

    def save_bit_trans(self, bit_trans):
        # 1. Ensure that the catalog exists
        os.makedirs("../results", exist_ok=True)  # Automatic catalog creation

        # 2. Constructing document paths
        file = os.path.join("../results", f"{self.save_name}_bit_trans.txt")

        # 3. Secure writing (using with to automatically close files)
        try:
            with open(file, "a") as f:
                # Harmonized treatment of various input types
                if torch.is_tensor(bit_trans):
                    f.write(str(bit_trans.tolist()))
                elif isinstance(bit_trans, (list, np.ndarray)):
                    f.write(str(list(bit_trans)))
                else:
                    f.write(str(bit_trans))
                f.write(", ")
            print(f"✓ data successfully saved to{file}")
        except Exception as e:
            print(f"✗ fail to save: {str(e)}")



