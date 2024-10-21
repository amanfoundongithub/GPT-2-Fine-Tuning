import pandas as pd 
import torch.utils.data as tud 
import torch  

from tqdm import tqdm 
from transformers import GPT2Tokenizer

# To show progress we have initialized tqdm for pandas 
tqdm.pandas() 

def tokenize_row(row, tokenizer : GPT2Tokenizer):
    # Input -> article
    inputs = tokenizer(row["article"], return_tensors = "pt", 
                           max_length = 256, padding = "max_length", 
                           truncation = True)
    # Target -> highlights 
    highlights = tokenizer(row["highlights"], return_tensors = "pt", 
                               max_length = 256, truncation = True,
                               padding = "max_length")
        
    return inputs.input_ids.squeeze(0), highlights.input_ids.squeeze(0)
    

def load_train_data(tokenizer : GPT2Tokenizer) -> pd.DataFrame:
    
    print("Loading Training Data...")
    df = pd.read_csv('./cnn_dailymail/train.csv')
    print("... CSV File converted. Now, tokenizing the sentences...")
    
    df["input"], df["target"] = zip(* df.progress_apply(lambda row : tokenize_row(row, tokenizer), axis = 1)) 
    
    df = df[["input", "target"]]
    
    print("... Tokenized sentences!")

    return df 

def load_valid_data(tokenizer : GPT2Tokenizer) -> pd.DataFrame:
    
    print("Loading Validation Data...")
    df = pd.read_csv('./cnn_dailymail/validation.csv')
    print("... CSV File converted. Now, tokenizing the sentences...")
    
    df["input"], df["target"] = zip(* df.progress_apply(lambda row : tokenize_row(row, tokenizer), axis = 1)) 
    
    df = df[["input", "target"]]
    print("... Tokenized sentences!")
    
    return df 


def load_test_data(tokenizer : GPT2Tokenizer) -> pd.DataFrame:
    
    print("Loading Testing Data...")
    df = pd.read_csv('./cnn_dailymail/test.csv')
    print("... CSV File converted. Now, tokenizing the sentences...")
    
    df["input"], df["target"] = zip(* df.progress_apply(lambda row : tokenize_row(row, tokenizer), axis = 1)) 
    
    df = df[["input", "target"]]
    print("... Tokenized sentences!")
    return df

def create_data_loader(data : pd.DataFrame, device = "cpu") -> tud.DataLoader:
    # Convert it to Pytorch dataset
    input_tensors = torch.stack(data["input"].tolist()).to(device)
    output_tensors = torch.stack(data["target"].tolist()).to(device) 

    # Dataset
    dataset = tud.TensorDataset(input_tensors, output_tensors)

    # Pytorch dataloader for training
    dataloader = tud.DataLoader(dataset = dataset,
                                batch_size = 16,
                                shuffle = False)
    
    return dataloader

