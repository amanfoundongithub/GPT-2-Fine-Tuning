import torch.nn as nn 
import torch 

from transformers import GPT2LMHeadModel

class LoRABase(nn.Module):
    """
    LoRA Base Layer 
    """
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        self.scaling = alpha/rank 
        self.A = nn.Parameter(torch.randn(in_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
    
    def forward(self, x):
        return self.scaling * (x @ self.A @ self.B)

class LoRA(nn.Module):
    """ 
    Main LoRA module
    """
    def __init__(self, linear_layer, rank, alpha):
        super().__init__()
        self.linear_layer = linear_layer
        self.lora = LoRABase(in_dim = linear_layer.nx, out_dim = linear_layer.nf, rank = rank, alpha = alpha)
    
    def forward(self, x):
        return self.linear_layer(x) + self.lora(x) 

class GPT2withLoRA(nn.Module):
    """ 
    Implements GPT-2 with LoRA Adaptation
    
    Args
    ---
    gpt_model : GPT Model 
    rank : LoRA adaptation to consider
    alpha : scaling factor 
    print_count : to print the count or not
    """
    def __init__(self, 
                 gpt_model : GPT2LMHeadModel, 
                 rank : int, 
                 alpha : float, 
                 print_count : bool = True):
        super().__init__()
        
        # GPT2 model 
        self.model = gpt_model
        
        # Count params before the addition
        count_before_total = sum(p.numel() for p in self.model.parameters())
        count_before = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Disable gradients for the parameters of the model 
        for param in self.model.parameters():
            param.requires_grad = False  
        
        # Add LoRA matrices to it 
        for block in self.model.transformer.h:
            # First attention
            block.attn.c_attn = LoRA(block.attn.c_attn, rank = rank, alpha = alpha)
        
            # Second attention
            block.attn.c_proj = LoRA(block.attn.c_proj, rank = rank, alpha = alpha)
           
        
        # Count params after the addition 
        count_after_total = sum(p.numel() for p in self.model.parameters())
        count_after = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        if print_count: 
            print("--------------------------------")
            print("EFFECT OF ADDING LoRA:")
            print(f"\nBefore training:\n\tTotal parameters : {count_before_total}\n\tTrainable parameters : {count_before}")
            print(f"\nAfter training:\n\tTotal parameters : {count_after_total}\n\tTrainable parameters : {count_after}")
            print("--------------------------------")
        
        
    def forward(self, input_ids , labels, inputs_embeds = None):
        return self.model(
            input_ids = input_ids,
            inputs_embeds = inputs_embeds,
            labels = labels 
        ) 
    
    def generate(self, input_ids, max_length = 100):
        return self.model.generate(
            input_ids = input_ids,
            max_length = max_length
        )
    
    def save(self, filename : str):
        torch.save(self.state_dict(), filename)
    
    def load(self, filename : str):
        self.load_state_dict(torch.load(filename))