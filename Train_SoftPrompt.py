import torch 
import torch.optim as optim 
import time 

from tqdm import tqdm 
from load_model import get_gpt2_with_soft_prompt, evaluate_model
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
# Evaluate metrics on the training set before training
print("Evaluating metrics on the training set before training...")
evaluate_model(gpt_model, gpt_tokenizer, training_dataloader, "train")

# Evaluate metrics on the validation set before training
print("Evaluating metrics on the validation set before training...")
evaluate_model(gpt_model, gpt_tokenizer, validation_dataloader, "valid")
#######################################################################################################

#------------------------------------ TRAINING BEGINS HERE -------------------------------------------#
#######################################################################################################
# Evaluate metrics on the training set before training
print("Evaluating metrics on the training set before training...")
evaluate_model(gpt_model, gpt_tokenizer, training_dataloader, "train")

# Evaluate metrics on the validation set before training
print("Evaluating metrics on the validation set before training...")
evaluate_model(gpt_model, gpt_tokenizer, validation_dataloader, "valid")
#######################################################################################################

#------------------------------------ TRAINING BEGINS HERE -------------------------------------------#
#######################################################################################################
for epoch in range(NO_OF_EPOCHS):
    print("-----------------------------------------")
    print(f"Epoch #{epoch + 1}\n")

    gpt_model.train() 
    
    for src, tgt in tqdm(training_dataloader):
        # Batch size 
        batch_size = src.size(0)
            
        # Forward pass
        output_logits = gpt_model(input_ids = src, labels = tgt)
        
        # Get the loss value
        output_loss = output_logits.loss 
        
        # Backward pass
        optimizer.zero_grad()
        output_loss.backward()
        optimizer.step()
    
    print("Evaluating metrics on the training set...")
    evaluate_model(gpt_model, gpt_tokenizer, training_dataloader, "train")
    print("Evaluating metrics on the validation set...")
    evaluate_model(gpt_model, gpt_tokenizer, validation_dataloader, "valid")
    print("-----------------------------------------")
#######################################################################################################
#------------------------------------ TRAINING ENDS HERE ---------------------------------------------#
gpt_model.save("gpt2_with_soft_prompt.pt") 

 