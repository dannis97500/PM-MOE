import torch
import torch.nn as nn
import torch.optim as optim
from flcore.trainmodel.moe.gate import Gating, CNNGating
from flcore.trainmodel.models import fastText
import torch.nn.functional as F
class MoE(nn.Module):
    def __init__(self, trained_experts):
        super(MoE, self).__init__()
        self.experts = nn.ModuleList(trained_experts)
        num_experts = len(trained_experts)
        # Assuming all experts have the same input dimension
        input_dim = trained_experts[0].in_features
        self.gating = Gating(input_dim, num_experts)

    def forward(self, x):
        # Get the weights from the gating network
        weights = self.gating(x)

        # Calculate the expert outputs
        outputs = torch.stack([expert(x) for expert in self.experts], dim=2)

        # Adjust the weights tensor shape to match the expert outputs
        weights = weights.unsqueeze(1).expand_as(outputs)

        # Multiply the expert outputs with the weights and
        # sum along the third dimension
        return torch.sum(outputs * weights, dim=2)
# top k moe
class ToPMoE(nn.Module):
    def __init__(self, trained_experts, gate_input_dim, args):
        super(ToPMoE, self).__init__()
        self.experts = nn.ModuleList(trained_experts)
        num_experts = len(trained_experts)
        # Assuming all experts have the same input dimension
        self.gating = Gating(gate_input_dim, num_experts)
        self.k = args.topk

    def forward(self, x):
        # Get the weights from the gating network
        weights = self.gating(x.flatten(1))  # [10,20]
        
        weights_values, indices = torch.topk(weights, self.k, dim=-1, largest=True, sorted=True, out=None) 
        
        # Calculate the expert
        results = []
        for i in range(x.size(0)): 
            expert_results = [self.experts[idx](x[i]) for idx in indices[i]]
            stacked_expert_results = torch.stack(expert_results) # [10,10] 
            results.append(stacked_expert_results)
            
        final_results = torch.stack(results)  
        weights_x = weights_values.unsqueeze(-1).expand_as(final_results)

        # outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        # # Adjust the weights tensor shape to match the expert outputs
        # weights = weights.unsqueeze(1).expand_as(outputs)

        # Multiply the expert outputs with the weights and
        # sum along the third dimension
        return torch.sum(final_results * weights_x, dim=1) 

class ExtractorToPMoE(nn.Module): 
    def __init__(self, trained_experts, gate_input_dim, args):
        super(ExtractorToPMoE, self).__init__()
        self.experts = nn.ModuleList(trained_experts)
        num_experts = len(trained_experts)
        # Assuming all experts have the same input dimension
        self.gating = Gating(gate_input_dim, num_experts)
        self.k = args.topk

    def forward(self, x):
        # Get the weights from the gating network
        weights = self.gating(x.flatten(1))  # [10,20]
        
        weights_values, indices = torch.topk(weights, self.k, dim=-1, largest=True, sorted=True, out=None) 
        
        # Calculate the expert
        results = []
        if isinstance(self.experts[0],fastText):
            for i in range(x.size(0)): 
                expert_results=[]
                for idx in indices[i]:
                    expert_output = self.experts[idx].fc1(x[i].unsqueeze(0))
                    h = self.experts[idx].fc(expert_output)
                    out = F.log_softmax(h, dim=1).flatten(0)
                    expert_results.append(out)
                # expert_results = [F.log_softmax(self.experts[idx].fc(self.experts[idx].fc1(x[i].mean(1))), dim=1).flatten(0) for idx in indices[i]]
                stacked_expert_results = torch.stack(expert_results) # [10,10] 
                results.append(stacked_expert_results)
        else:
            for i in range(x.size(0)): 
                expert_results = [self.experts[idx](x[i].unsqueeze(0)).flatten(0) for idx in indices[i]]
                stacked_expert_results = torch.stack(expert_results) # [10,10] 
                results.append(stacked_expert_results)
            
        final_results = torch.stack(results)  
        weights_x = weights_values.unsqueeze(-1).expand_as(final_results)

        # outputs = torch.stack([expert(x) for expert in self.experts], dim=2)
        # # Adjust the weights tensor shape to match the expert outputs
        # weights = weights.unsqueeze(1).expand_as(outputs)

        # Multiply the expert outputs with the weights and
        # sum along the third dimension
        return torch.sum(final_results * weights_x, dim=1) 


class ParamToPMoE(nn.Module):
    def __init__(self, trained_experts, args):
        super(ParamToPMoE, self).__init__()
        self.experts = trained_experts # nn.params
        num_experts = len(trained_experts)
        # Assuming all experts have the same input dimension
        exp_dim = trained_experts[0].shape[0]  # same shape with input data -- rep
        self.gating = Gating(exp_dim, num_experts)
        self.k = args.topk
        
    def forward(self, x):
        # Get the weights from the gating network
        weights = self.gating(x)  # [10,20]
        
        weights_values, indices = torch.topk(weights, self.k, dim=-1, largest=True, sorted=True, out=None) 
        
        # Calculate the expert
        results = []
        for i in range(x.size(0)): 
            expert_results = [self.experts[idx] for idx in indices[i]]
            stacked_expert_results = torch.stack(expert_results) # [10,10] 
            results.append(stacked_expert_results)
            
        final_results = torch.stack(results)  
        weights_x = weights_values.unsqueeze(-1).expand_as(final_results)

        return torch.sum(final_results * weights_x, dim=1) 


class PatchMoE(nn.Module):
    def __init__(self, trained_experts, data_type="cifar10"):
        super(PatchMoE, self).__init__()
        self.data_type = data_type
        self.experts = nn.ModuleList(trained_experts)
        self.num_experts = len(trained_experts)
        # Assuming all experts have the same input dimension
        self.input_dim = trained_experts[0].in_features * self.num_experts
        self.gating = Gating(self.input_dim, self.num_experts)
        self.trained_experts = trained_experts

    def forward(self, x):
        batchsize = x.shape[0]
        x_flattened = x.reshape(batchsize, -1) # 10, 3072
        
        # Get the weights from the gating network
        weights = self.gating(x_flattened)
        
        segments = [x_flattened[:, i * self.trained_experts[0].in_features:(i + 1) * self.trained_experts[0].in_features] for i in range(self.num_experts)]

        outputs = []
        for expert, segment in zip(self.experts, segments):
            
            output = expert(segment)
            
            outputs.append(output)
        # Calculate the expert outputs
        # outputs = torch.stack([expert(x) for expert in self.experts], dim=2)

        stacked_outputs = torch.stack(outputs, dim=2) 
        
        # Adjust the weights tensor shape to match the expert outputs
        weights = weights.unsqueeze(1)
        
        # Multiply the expert outputs with the weights and
        # sum along the third dimension
        weight_out = stacked_outputs * weights  
        weight_out_t = weight_out.transpose(-2, -1) 
        
        if self.data_type =="cifar10":
            return weight_out_t.flatten(1).view(batchsize, 3, 32, 32) 
        
        return torch.sum(stacked_outputs * weights, dim=2)



class PatchCNNMoE(nn.Module):
    def __init__(self, trained_experts, data_type="cifar10"):
        super(PatchCNNMoE, self).__init__()
        self.data_type = data_type
        self.experts = nn.ModuleList(trained_experts)
        self.num_experts = len(trained_experts)
        # Assuming all experts have the same input dimension
        self.input_dim = trained_experts[0].in_channels
        self.gating = CNNGating(self.input_dim, self.num_experts)
        self.trained_experts = trained_experts

    def split_tensor(self, x, n):

        assert len(x.shape) == 4, "Input tensor must be 4-dimensional"
        
        p, q = x.shape[-2], x.shape[-1]
        

        if p % n == 0:
            split_dim = -2  
            block_size = p // n
        elif q % n == 0:
            split_dim = -1  
            block_size = q // n
        else:
            raise ValueError(f"Neither dimension {p} nor {q} can be evenly divided by {n}")
        
        split_x = torch.chunk(x, chunks=n, dim=split_dim)
        
        return split_x
    
    def forward(self, x):
        batchsize = x.shape[0]
        split_x = self.split_tensor(x, self.num_experts)
        
        weights = self.gating(x)
        # top k
        
        outputs = []
        for expert, segment in zip(self.experts, split_x):
            
            output = expert(segment)
            
            outputs.append(output)
        # Calculate the expert outputs
        # outputs = torch.stack([expert(x) for expert in self.experts], dim=2)

        stacked_outputs = torch.stack(outputs, dim=2) 
        
        # Adjust the weights tensor shape to match the expert outputs
        weights = weights.unsqueeze(1)
        
        # Multiply the expert outputs with the weights and
        # sum along the third dimension
        weight_out = stacked_outputs * weights  
        weight_out_t = weight_out.transpose(-2, -1) 
        weight_out_t = weight_out_t.flatten(1)
        
        if self.data_type =="cifar10":
            return weight_out_t 
        
        return torch.sum(stacked_outputs * weights, dim=2)
