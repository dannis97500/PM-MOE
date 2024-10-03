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

import numpy as np
import time
from flcore.clients.clientbase import Client
from flcore.trainmodel.moe.moe import ExtractorToPMoE
import torch
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import copy

class clientFedMoE(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = args
        self.model.local_extra = copy.deepcopy(self.model.base)
        

    def train(self):
        trainloader = self.load_train_data()
        
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
            
        # reset experts
        self.trained_experts = []
        self.trained_experts.append(self.model.local_extra)
        self.trained_experts.append(self.model.base)
        
        if self.dataset == "AGNews":
            self.model.moe = ExtractorToPMoE(trained_experts = self.trained_experts,
                                gate_input_dim = 32, 
                                args = self.args).to(self.args.device)
        else:
            self.model.moe = ExtractorToPMoE(trained_experts = self.trained_experts,
                                    gate_input_dim = len(trainloader.dataset[0][0].reshape(-1)), 
                                    args = self.args).to(self.args.device)
        
        self.moe_opt= torch.optim.SGD(self.model.parameters(), lr=self.args.local_learning_rate)
        
        for epoch in range(max_local_epochs):
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

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.base.parameters()):
            old_param.data = new_param.data.clone()


  