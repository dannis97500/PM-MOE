# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
import copy
import time
from flcore.clients.clientavgDBE import clientAvgDBE, PMOE_clientAvgDBE
from flcore.servers.serverbase import Server
from threading import Thread
import os
import pickle
import torch

class FedAvgDBE(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()

        # initialization period
        self.set_clients(clientAvgDBE)
        self.selected_clients = self.clients
        for client in self.selected_clients:
            client.train() # no DBE

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
        
        global_mean = 0
        for cid, w in zip(self.uploaded_ids, self.uploaded_weights):
            global_mean += self.clients[cid].running_mean * w
        print('>>>> global_mean <<<<', global_mean)
        for client in self.selected_clients:
            client.global_mean = global_mean.data.clone()

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        print('featrue map shape: ', self.clients[0].client_mean.shape)
        print('featrue map numel: ', self.clients[0].client_mean.numel())


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()  # client id, client weights, client model param
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


class P_MOEFedAvgDBE(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()

        # initialization period
        self.set_clients(PMOE_clientAvgDBE)
        
        self.selected_clients = self.clients
        # for client in self.selected_clients:             # --------please annotate in pmoe finetune --------
        #     client.train() # no DBE                      # --------please annotate in pmoe finetune --------
        # load client in pmoe 
        for i in range(len(self.selected_clients)):    # --------please cancel annotate in pmoe finetune --------
            client = self.selected_clients[i]          # --------please cancel annotate in pmoe finetune --------
            loaded_client = self.load_clients(client)  # --------please cancel annotate in pmoe finetune --------
            self.selected_clients[i] = loaded_client   # --------please cancel annotate in pmoe finetune --------
        
        
        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
        
        global_mean = 0
        for cid, w in zip(self.uploaded_ids, self.uploaded_weights):
            global_mean += self.clients[cid].running_mean * w
        print('>>>> global_mean <<<<', global_mean)
        for client in self.selected_clients:
            client.global_mean = global_mean.data.clone()

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.load_model()                               # --------please cancel annotate in pmoe finetune --------
        self.Budget = []
        print('featrue map shape: ', self.clients[0].client_mean.shape)
        print('featrue map numel: ', self.clients[0].client_mean.numel())
        
        self.fintuned_heads = []
        
        self.temp_test_acc = 0 # for save best model
        self.temp_max_test_acc = 0 # for save best model
        self.save_best_model = False

    def save_clients(self, client):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_client_object_" + str(client.id) + ".pkl")
        
        
        with open(model_path, 'wb') as f:
            pickle.dump(client, f)
            
        # torch.save(self.model, model_path)
        print(f"client {str(client.id)} saved")

    def load_clients(self, client):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_client_object_" + str(client.id) + ".pkl")
        assert (os.path.exists(model_path))
        with open(model_path, 'rb') as f:
            client = pickle.load(f)
        return client
    
    
    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate model")
                self.evaluate()
                
                self.save_global_model()
                self.save_results()

            for client in self.selected_clients:
                self.save_clients(client)
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()  # client id, client weights, client model param
            self.aggregate_parameters() 

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break


        print("self.rs_test_acc=",self.rs_test_acc)
        print("self.rs_test_auc=",self.rs_test_auc)
        print("self.rs_train_loss=",self.rs_train_loss)
        
        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        # self.save_results()
        # self.save_global_model()
        
        
    def moefinetune(self):
        s_t = time.time()
        self.selected_clients = self.select_clients()
 
        
        # self.rs_test_acc = []
        # self.rs_test_auc = []
        # self.rs_train_loss = []
        
        
        for client in self.selected_clients:
            self.fintuned_heads.append(client.client_mean) 

        print(f"\n-------------Before evaluate:-------------")
        self.evaluate()  
        
        print(f"\n-------------MOE finetuning:-------------")

        index = 0
        for client in self.selected_clients:
            print("client {} start moe finetune !!!!".format(index))
            experts = copy.deepcopy(self.fintuned_heads) 
            client.set_moe_experts(experts)  # sever send moe experts
            client.moe_finetune()  # clinet moe finetune
            self.evaluate()  # after each client finetune eval 
            index += 1 # self.selected_clients 
        
        
        self.Budget.append(time.time() - s_t)
        # print('-'*50, self.Budget[-1])
        print("self.rs_test_acc=",self.rs_test_acc)
        print("self.rs_test_auc=",self.rs_test_auc)
        print("self.rs_train_loss=",self.rs_train_loss)
        
        
        print("\nBest global accuracy.")
        print(max(self.rs_test_acc))
        print("\nBest global auc.")
        print(max(self.rs_test_auc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[:])/len(self.Budget[:]))
        # print(time.time()-s_t)
