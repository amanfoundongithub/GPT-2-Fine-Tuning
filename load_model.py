from transformers import GPT2LMHeadModel, GPT2Tokenizer

from GPT2withLoRA import GPT2withLoRA
from GPT2withTradFT import GPT2withTraditionalFineTuning
from GPT2withSoftPrompt import GPT2withSoftPrompt

from torch.utils.data import DataLoader
from torch.nn import Module
from rouge_score import rouge_scorer
from tqdm import tqdm 
import torch 


 # Initialize ROUGE scorer for calculating ROUGE score 
rouge_scorer_fn = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    

# Returns the GPT model, tokenizer and embedding dimension
def get_gpt2_with_soft_prompt(prompt_length):
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2") 
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # Model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    # Embedding dimension of GPT-2
    embedding_dimension = model.config.n_embd
    
    gpt2_soft = GPT2withSoftPrompt(
        prompt_length = prompt_length,
        embedding_dim = embedding_dimension,
        gpt_model = model
    )

    return {
        "tokenizer" : tokenizer,
        "model" : gpt2_soft,
        "embedding_dim" : embedding_dimension,
    }


def get_gpt2_with_lora(rank : int, alpha : float, filename : str = None):
    """ 
    Creates a GPT-2 model with LoRA addition
    
    Args 
    ----
    rank : LoRA matrix rank
    alpha : scaling factor 
    filename : (Optional) if a gpt-2 model with LoRA exists, then it will load the file 
    
    Returns 
    ----
    A dictionary with entries as follows:
    * tokenizer : GPT-2 Tokenizer
    * model : GPT-2 model with LoRA
    * embedding_dim : Embedding dimension of the GPT-2 model 
    """
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2") 
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # Model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    # Embedding dimension of GPT-2
    embedding_dimension = model.config.n_embd
    
    # GPT-2 with LoRA matrices 
    gpt2_lora = GPT2withLoRA(gpt_model = model, 
                             rank = rank, 
                             alpha = alpha)

    if filename is not None: 
        gpt2_lora.load(filename) 
    
    return {
        "tokenizer" : tokenizer,
        "model" : gpt2_lora, 
        "embedding_dim" : embedding_dimension,
    }


def get_gpt2_with_traditional_fine_tuning(filename : str = None):
    """ 
    Creates a GPT-2 model with Last Layer Fine tuning 
    
    Args 
    ----
    filename : (Optional) if a gpt-2 model fine tuned exists, then it will load the file 
    
    Returns 
    ----
    A dictionary with entries as follows:
    * tokenizer : GPT-2 Tokenizer
    * model : GPT-2 model with LoRA
    * embedding_dim : Embedding dimension of the GPT-2 model 
    """
    # Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2") 
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    
    # Model
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    gpt2_trad_ft = GPT2withTraditionalFineTuning(gpt_model = model) 
    
    # Embedding dimension of GPT-2
    embedding_dimension = model.config.n_embd
    
    if filename is not None: 
        gpt2_trad_ft.load(filename) 
        
    return {
        "tokenizer" : tokenizer,
        "model" : gpt2_trad_ft, 
        "embedding_dim" : embedding_dimension,
    }


def evaluate_model(gpt_model : Module,
                   gpt_tokenizer : GPT2Tokenizer,
                   dataloader : DataLoader,
                   typeof : str = "train" | "valid" | "test"):

    # Evaluation loop
    loss = 0.0
    rouge1 = 0.0
    rouge2 = 0.0
    rougeL = 0.0
    
    count = 0
    
    gpt_model.eval()
    print("-----------------------------------------") 
    with torch.no_grad():  
        for src, tgt in tqdm(dataloader):
            # Batch size 
            batch_size = src.size(0)
            
            # Forward pass
            output_logits = gpt_model(
                input_ids = src,
                labels = tgt,
            )
    
            # Accumulate loss
            loss += output_logits.loss.item()
    
            # Generate predictions
            generated_ids = gpt_model.generate(input_ids=src, max_length=150)
    
            # Decode the predicted and reference sentences
            generated_texts = [gpt_tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
            reference_texts = [gpt_tokenizer.decode(t, skip_special_tokens=True) for t in tgt]
    
            # Calculate ROUGE scores for training batch
            rouge_scores = [rouge_scorer_fn.score(ref, gen) for ref, gen in zip(reference_texts, generated_texts)]
    
            # Average ROUGE scores for training batch
            avg_rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / batch_size
            avg_rouge2 = sum([score['rouge2'].fmeasure for score in rouge_scores]) / batch_size
            avg_rougeL = sum([score['rougeL'].fmeasure for score in rouge_scores]) / batch_size
    
            rouge1 += avg_rouge1
            rouge2 += avg_rouge2
            rougeL += avg_rougeL
    
            count += 1
    
    loss = loss / count
    rouge1 = rouge1 / count
    rouge2 = rouge2 / count
    rougeL = rougeL / count
    
    if typeof == "train":
        print(f"\nTraining Summmary:")
    elif typeof == "valid":
        print(f"\nValidation Summary:")
    else: 
        print(f"\nTesting summary:")
    
    print(f"\nLoss: {loss}\nROUGE-1: {rouge1} ROUGE-2: {rouge2} ROUGE-L: {rougeL}\n")
    print("-----------------------------------------") 
    
