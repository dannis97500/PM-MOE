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

import random
import time
from flcore.clients.clientrep import clientRep, PMOE_clientRep
from flcore.servers.serverbase import Server
from threading import Thread
import copy

class FedRep(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientRep)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
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
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientRep)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
        

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model.base)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

class PMOE_FedRep(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(PMOE_clientRep)

        for i in range(len(self.clients)):              # --------please cancel annotate in pmoe finetune --------
            client = self.clients[i]                    # --------please cancel annotate in pmoe finetune --------
            loaded_client = self.load_clients(client)   # --------please cancel annotate in pmoe finetune --------
            self.clients[i] = loaded_client             # --------please cancel annotate in pmoe finetune --------
        
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.load_model()                             # --------please cancel annotate in pmoe finetune --------
        self.Budget = []
        
        self.fintuned_heads = []
        self.temp_test_acc = 0     # for save best model
        self.temp_max_test_acc = 0 # for save best model
        self.save_best_model = False


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
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

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("self.rs_test_acc=",self.rs_test_acc)
        print("self.rs_test_auc=",self.rs_test_auc)
        print("self.rs_train_loss=",self.rs_train_loss)
        
        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(PMOE_clientRep)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()
        
    def moefinetune(self):
        s_t = time.time()
        self.selected_clients = self.select_clients()
 
        
        # self.rs_test_acc = []
        # self.rs_test_auc = []
        # self.rs_train_loss = []
        
        
        for client in self.selected_clients:
            self.fintuned_heads.append(client.model.head)

        print(f"\n-------------Before evaluate:-------------")
        self.evaluate()
        
        print(f"\n-------------MOE finetuning:-------------")
        self.moe_fine_tune = True  
        index = 0
        for client in self.selected_clients:
            print("client {} start moe finetune !!!!".format(index))
            head_list = copy.deepcopy(self.fintuned_heads) 
            client.set_moe_experts(head_list)  # sever send moe experts
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

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                    client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model.base)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples