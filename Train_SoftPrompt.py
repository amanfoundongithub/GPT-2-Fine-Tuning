import torch 
import torch.optim as optim 
import time 

from tqdm import tqdm 
from rouge_score import rouge_scorer
from load_model import get_gpt2_with_soft_prompt
from load_data import load_train_data, load_valid_data, create_data_loader

###########################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROMPT_LENGTH = 10

NO_OF_EPOCHS = 4
###########################


#######################################################################################################

# Load the GPT-2 with LoRA model
lora_dict = get_gpt2_with_soft_prompt(prompt_length = PROMPT_LENGTH)

gpt_model = lora_dict.get("model").to(DEVICE)
gpt_tokenizer = lora_dict.get("tokenizer") 

# Create a trainer for optimizing the training
optimizer = optim.Adam(gpt_model.parameters(), lr = 5e-4)

# Initialize ROUGE scorer for calculating ROUGE score 
rouge_scorer_fn = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#######################################################################################################

#######################################################################################################
# Load the training dataset
training_data = load_train_data(tokenizer = gpt_tokenizer)

# Create a dataloader
training_dataloader = create_data_loader(training_data, DEVICE)

# Load the validation dataset
validation_data = load_valid_data(tokenizer = gpt_tokenizer)

# Create a dataloader
validation_dataloader = create_data_loader(validation_data, DEVICE)
#######################################################################################################

#######################################################################################################

# Function to print GPU memory usage
def print_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert bytes to MB
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # Convert bytes to MB
        print(f"GPU memory allocated: {allocated:.2f} MB")
        print(f"GPU memory reserved: {reserved:.2f} MB")
    else:
        print("CUDA is not available")

def evaluate(dataloader , train = True):
    # Set the mode to eval
    gpt_model.eval() 
    
    loss = 0.0
    rouge1 = 0.0
    rouge2 = 0.0
    rougeL = 0.0
    
    count = 0
    
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
    
            # Accumulate training loss
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
    
    if train:
        print(f"\nTraining summmary:")
    else: 
        print(f"\nValidation summary:")
    
    print(f"\nLoss: {loss}\nROUGE-1: {rouge1} ROUGE-2: {rouge2} ROUGE-L: {rougeL}\n")
    print("-----------------------------------------") 
#######################################################################################################

#------------------------------------ TRAINING BEGINS HERE -------------------------------------------#
#######################################################################################################

# Evaluate metrics on the training set before training
print("Evaluating metrics on the training set before training...")
evaluate(training_dataloader, train = True) 

# Evaluate metrics on the validation set before training
print("Evaluating metrics on the validation set before training...")
evaluate(validation_dataloader, train = False)

# Training loop  
for epoch in range(NO_OF_EPOCHS):
    print("-----------------------------------------")
    print(f"Epoch #{epoch + 1}\n")
    
    epoch_loss = 0.0 
    epoch_rouge1 = 0.0 
    epoch_rouge2 = 0.0 
    epoch_rougeL = 0.0 
    
    count = 0
    
    gpt_model.train() 
    
    for src, tgt in tqdm(training_dataloader):
        # Batch size 
        batch_size = src.size(0)
            
        # Forward pass
        output_logits = gpt_model(
                input_ids = src,
                labels = tgt,
        )
        
    
        # Get the loss value
        output_loss = output_logits.loss 
        
        # Backward pass
        optimizer.zero_grad()
        output_loss.backward()
        optimizer.step()
        
        epoch_loss += output_loss.item() 
        count += 1
    
    print_gpu_memory_usage()
    
    print(f"\nLoss : {epoch_loss/count}\n")

    print("Evaluating metrics on the training set...")
    evaluate(training_dataloader, train = True) 
    
    print("Evaluating metrics on the validation set...")
    evaluate(validation_dataloader, train = False)
    
    print("-----------------------------------------")


#######################################################################################################

#------------------------------------ TRAINING ENDS HERE ---------------------------------------------#
gpt_model.save("gpt2_with_soft_prompt.pt") 

 