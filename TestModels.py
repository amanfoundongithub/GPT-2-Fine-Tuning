import torch

from load_model import get_gpt2_with_lora, get_gpt2_with_traditional_fine_tuning, get_gpt2_with_soft_prompt, evaluate_model
from load_data import load_test_data, create_data_loader


###########################
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROMPT_LENGTH = 10

RANK   = 8
ALPHA  = 16
###########################


# ------------------------ EVALUATION OF LORA ON TEST SET ------------------------------------------- #
#######################################################################################################
# Load the GPT-2 with LoRA model pretrained 
lora_dict = get_gpt2_with_lora(RANK, ALPHA, filename = "gpt2_with_lora.pt")

gpt_model = lora_dict.get("model").to(DEVICE)
gpt_tokenizer = lora_dict.get("tokenizer") 
#######################################################################################################

#######################################################################################################
# Load the testing dataset
testing_dataset = load_test_data(tokenizer = gpt_tokenizer)

# Convert to pytorch dataloader
testing_dataloader = create_data_loader(testing_dataset)
#######################################################################################################

#######################################################################################################
# Evaluate metrics 
print("Evaluating LoRA on Test Set...")
evaluate_model(gpt_model, gpt_tokenizer, testing_dataloader, typeof = "test")
#######################################################################################################


# ------------------------ EVALUATION OF FT ON TEST SET --------------------------------------------- #
#######################################################################################################
# Load the GPT-2 with fine tuned model pretrained 
lora_dict = get_gpt2_with_traditional_fine_tuning(filename = "gpt2_with_traditional_fine_tuning.pt")

gpt_model = lora_dict.get("model").to(DEVICE)
gpt_tokenizer = lora_dict.get("tokenizer") 
#######################################################################################################

#######################################################################################################
# Evaluate metrics 
print("Evaluating Traditional Fine Tuning on Test Set...")
evaluate_model(gpt_model, gpt_tokenizer, testing_dataloader, typeof = "test")
#######################################################################################################


# ------------------------ EVALUATION OF SOFT PROMPT ON TEST SET ------------------------------------ #
#######################################################################################################
# Load the GPT-2 with fine tuned model pretrained 
lora_dict = get_gpt2_with_soft_prompt(prompt_length = PROMPT_LENGTH, filename = "gpt2_with_soft_prompt.pt")

gpt_model = lora_dict.get("model").to(DEVICE)
gpt_tokenizer = lora_dict.get("tokenizer") 
#######################################################################################################

#######################################################################################################
# Evaluate metrics 
print("Evaluating Soft Prompting on Test Set...")
evaluate_model(gpt_model, gpt_tokenizer, testing_dataloader, typeof = "test")
#######################################################################################################