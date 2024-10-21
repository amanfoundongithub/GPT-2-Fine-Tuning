
import torch.nn as nn 
import torch 

from transformers import GPT2LMHeadModel

class SoftPrompt(nn.Module):
    """
    Soft prompt tuning module 
    """
    def __init__(self, embedding_dim, prompt_length):
        
        super().__init__()
        
        self.prompt_matrix = nn.Parameter(
            torch.randn(prompt_length, embedding_dim)
        )
    
    
    def forward(self, batch_size):
        # Unsqueeze 1
        prompt_matrix = self.prompt_matrix.unsqueeze(0)
        # Expand along batch_size dimension
        prompt_matrix = prompt_matrix.expand(batch_size, -1, -1)
        
        return prompt_matrix

class GPT2withSoftPrompt(nn.Module):
    
    def __init__(self, 
                 prompt_length : int, 
                 embedding_dim : int, 
                 gpt_model : GPT2LMHeadModel,
                 print_count : bool = True):

        
        super().__init__()
        
        self.model = gpt_model
        
        # Count params before the addition
        count_before_total = sum(p.numel() for p in self.model.parameters())
        count_before = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        for param in self.model.parameters():
            param.requires_grad = False 
        
        self.prompt = SoftPrompt(
            embedding_dim = embedding_dim,
            prompt_length = prompt_length
        )
        self.prompt_length = prompt_length
        
        count_after_total = sum(p.numel() for p in self.parameters())
        
        count_after = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        if print_count: 
            print("--------------------------------")
            print("EFFECT OF ADDING Soft Prompt:")
            print(f"\nBefore training:\n\tTotal parameters : {count_before_total}\n\tTrainable parameters : {count_before}")
            print(f"\nAfter training:\n\tTotal parameters : {count_after_total}\n\tTrainable parameters : {count_after}")
            print("--------------------------------")
    
    def forward(self, input_ids, labels):
        src = input_ids
        tgt = labels 
        
        batch_size = src.size(0)
        
        # Get the embeddings from the lookup table
        src_embeddings = self.model.transformer.wte(src)
        
        # Get prompts to prepend
        src_prompts = self.prompt(batch_size)
        
        # Prepend the prompt
        src_embeddings = torch.cat([src_prompts, src_embeddings], dim=1)
        
        # Ignore the prompt tokens
        ignore_prompt_tokens = torch.full((batch_size, self.prompt_length), -100, dtype=torch.long).to(src.device)
        
        # Add them to the target tokens
        tgt = torch.cat([ignore_prompt_tokens, tgt], dim=1)
        
        return self.model(
            inputs_embeds = src_embeddings,
            labels = tgt 
        )
    
    
    def generate(self,input_ids, max_length = 100):
        src = input_ids
        
        batch_size = src.size(0)
        
        # Get the embeddings from the lookup table
        src_embeddings = self.model.transformer.wte(src)
        
        # Get prompts to prepend
        src_prompts = self.prompt(batch_size)
        
        # Prepend the prompt
        src_embeddings = torch.cat([src_prompts, src_embeddings], dim=1)
        
        return self.model.generate(
            inputs_embeds = src_embeddings,
            max_length = max_length
        )
    
    def save(self, filename):
        torch.save(self.state_dict(), filename)
    
    def load(self, filename):
        self.load_state_dict(torch.load(filename))
    
    
        
        
        
        