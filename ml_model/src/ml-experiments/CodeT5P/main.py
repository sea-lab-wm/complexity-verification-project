import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5EncoderModel, AutoTokenizer
from torch import nn, optim
from torch.nn import functional as F
import time
import matplotlib.pyplot as plt
import wandb

from model import CodeT5Classifier
from dataset import CodeDataset
from utils import RANDOM_SEED, balance_dataset, calc_accuracy_loader, train_classifier_simple

from sklearn import metrics

def main():
    # Load the dataset
    ROOT='/home/nadeeshan/ML-Experiments-2/complexity-verification-project/ml_model/model/NewExperiments/CodeT5P/data/'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = "Salesforce/codet5p-220m"

    # For reproducibility due to the shuffling in the training data loader
    torch.manual_seed(RANDOM_SEED)
    
        
    ## Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    ## Create the datasets
    train_csv = ROOT + 'train.csv'
    val_csv = ROOT + 'val.csv'
    test_csv = ROOT + 'test.csv'

    train_csv = ROOT + balance_dataset(train_csv, target='PBU')

    train_dataset = CodeDataset(tokenizer, train_csv, target='PBU', code_features=["LOC", "PE gen", "PE spec (java)"])
    val_dataset = CodeDataset(tokenizer, val_csv, target='PBU', code_features=["LOC", "PE gen"])
    test_dataset = CodeDataset(tokenizer, test_csv, target='PBU', code_features=["LOC", "PE gen"])

    num_workers = 0
    batch_size = 24

    # Create DataLoaders
    train_loader = DataLoader(dataset=train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, 
                             batch_size=batch_size, 
                             shuffle=False, 
                             num_workers=num_workers)


    # Instantiate the model
    model = CodeT5Classifier(
        encoder_model_name=checkpoint, 
        input_dim=768, 
        hidden_dim=256, 
        num_classes=2)
    model.to(device)

    ########################################
    ## Fine Tunning For the Specific task ##
    ########################################
    # Define the optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
    num_epochs = 10
    
    start_time = time.time()
    
    wandb.init(project="verification", entity="ngimhana")

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # Save the fine-tuned model
    torch.save(model.state_dict(), 'CodeT5P/codet5+_finetuned.pth')

    train_accuracy,_,_ = calc_accuracy_loader(train_loader, model, device)
    val_accuracy,_,_ = calc_accuracy_loader(val_loader, model, device)
    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")

    model_state_dict = torch.load("CodeT5P/codet5+_finetuned.pth")
    model.load_state_dict(model_state_dict)
    test_accuracy, correct_predictions, targets= calc_accuracy_loader(test_loader, model, device)
    print(metrics.classification_report(targets, correct_predictions))

    print(f"Test accuracy: {test_accuracy*100:.2f}%")


## Run the main function
main()