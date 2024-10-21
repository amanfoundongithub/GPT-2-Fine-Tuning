import torch.nn as nn 
import torch 

from transformers import GPT2LMHeadModel

class GPT2withTraditionalFineTuning(nn.Module):
    """ 
    Implements GPT-2 with Traditional Last Layer Fine Tuning 
    
    Args
    ---
    gpt_model : GPT Model 
    print_count : to print the count or not
    """
    def __init__(self, 
                 gpt_model : GPT2LMHeadModel, 
                 print_count : bool = True):
        
        super().__init__()
        self.model = gpt_model
        
        # Count params before the addition
        count_before_total = sum(p.numel() for p in self.model.parameters())
        count_before = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        for param in self.model.parameters():
            param.requires_grad = False 
         
        for param in self.model.lm_head.parameters():
            param.requires_grad = True
        
        # Count params after the addition
        count_after_total = sum(p.numel() for p in self.model.parameters())
        count_after = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        if print_count: 
            print("--------------------------------")
            print("EFFECT OF Traditional Fine Tuning:")
            print(f"\nBefore training:\n\tTotal parameters : {count_before_total}\n\tTrainable parameters : {count_before}")
            print(f"\nAfter training:\n\tTotal parameters : {count_after_total}\n\tTrainable parameters : {count_after}")
            print("--------------------------------")
        
    def save(self, filename):
        torch.save(self.state_dict(), filename)
    
    def load(self, filename):
        self.load_state_dict(torch.load(filename))
    
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