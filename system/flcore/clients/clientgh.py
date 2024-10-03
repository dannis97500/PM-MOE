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

import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from collections import defaultdict
from flcore.trainmodel.moe.moe import ExtractorToPMoE
from sklearn import metrics
from sklearn.preprocessing import label_binarize



class clientGH(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.args = args
        self.moe_fine_tuning_epochs = args.moe_fine_tuning_epochs
        self.trained_experts = None  # moe experts
        self.is_moe_finetune = False
        self.lock_experts = args.lock_experts

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

        
    def set_parameters(self, head):
        for new_param, old_param in zip(head.parameters(), self.model.head.parameters()):
            old_param.data = new_param.data.clone()

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        self.protos = agg_func(protos)
        


class PMOE_clientGH(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.args = args
        self.moe_fine_tuning_epochs = args.moe_fine_tuning_epochs
        self.trained_experts = None  # moe experts
        self.is_moe_finetune = False
        self.lock_experts = args.lock_experts

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

        
    def set_parameters(self, head):
        for new_param, old_param in zip(head.parameters(), self.model.head.parameters()):
            old_param.data = new_param.data.clone()

    def collect_protos(self):
        trainloader = self.load_train_data()
        self.model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = self.model.base(x)

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

        self.protos = agg_func(protos)
        
        
    def set_moe_experts(self, fintuned_heads):
        self.trained_experts = fintuned_heads
        
     # after personalized finetune use moe control heads
    def moe_finetune(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        
        self.model.train() # MoeHeadSpilt
        
        self.is_moe_finetune = True
        
        assert self.trained_experts is not None
        
        if self.dataset == "AGNews":
            self.model.moe = ExtractorToPMoE(trained_experts = self.trained_experts,
                                gate_input_dim = 32, 
                                args = self.args).to(self.args.device)
        else:
            self.model.moe = ExtractorToPMoE(trained_experts = self.trained_experts,
                                    gate_input_dim = len(trainloader.dataset[0][0].reshape(-1)), 
                                    args = self.args).to(self.args.device)
            
        
        for param in self.model.parameters():
            param.requires_grad = False
            
            # moe 
            for param in self.model.moe.gating.parameters():
                param.requires_grad = True
                
            # expert 
            if self.lock_experts == 1: # 0--lock  1--unlock
                for param in self.model.moe.experts.parameters():
                    param.requires_grad = True
        
        # reset optimi
        self.moe_opt= torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        
        for epoch in range(self.args.moe_fine_tuning_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                
                if self.dataset == "AGNews":
                    text, text_lengths = x
                    emb = self.model.base.embedding(text)
                    rep = self.model.moe(emb.mean(1))
                else:
                    rep = self.model.moe(x)
                    
                output = self.model.head(rep)
                loss = self.loss(output, y)

                self.moe_opt.zero_grad()
                loss.backward()
                self.moe_opt.step()
                
        
        self.train_time_cost['total_cost'] += time.time() - start_time
        
        
    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                if self.is_moe_finetune == True:
                    if self.dataset == "AGNews":
                        text, text_lengths = x
                        emb = self.model.base.embedding(text)
                        rep = self.model.moe(emb.mean(1))
                    else:
                        rep = self.model.moe(x)
                    output = self.model.head(rep)
                else:
                    output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                
                if self.is_moe_finetune == True:
                    
                    if self.dataset == "AGNews":
                        text, text_lengths = x
                        emb = self.model.base.embedding(text)
                        rep = self.model.moe(emb.mean(1))
                    else:
                        rep = self.model.moe(x)
                    output = self.model.head(rep)
                else:
                    output = self.model(x)
                    
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos