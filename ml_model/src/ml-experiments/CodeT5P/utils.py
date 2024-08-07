import torch
import wandb
from imblearn.over_sampling import SMOTE, RandomOverSampler
import pandas as pd

RANDOM_SEED=123

def calc_loss_batch(input_batch, code_features, target_batch, attention_mask, model):
    logits = model(input_ids=input_batch, attention_mask=attention_mask, code_features=code_features)  # Logits of last output token
    loss = torch.nn.functional.cross_entropy(logits, target_batch)
    return loss


@torch.no_grad() # Disable gradient tracking for efficiency
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    correct_predictions_list, target_list = [], []

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for batch_id, data in enumerate(data_loader):
        input_batch, code_features, attention_mask, target_batch = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
        if batch_id < num_batches:
            logits = model(input_ids=input_batch, attention_mask=attention_mask, code_features=code_features) # Logits of last output token
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()

            target_list.extend(target_batch.cpu().numpy())
            correct_predictions_list.extend(predicted_labels.cpu().numpy())

        else:
            break
    return correct_predictions / num_examples, correct_predictions_list, target_list

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for batch_id, data in enumerate(data_loader):
        input_batch, code_features, attention_mask, target_batch = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
        if batch_id < num_batches:
            loss = calc_loss_batch(input_batch, code_features, target_batch, attention_mask, model)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

# Overall the same as `train_model_simple` in chapter 5
def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen= 0

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        total_loss = 0
        examples_seen = 0
        correct_predictions = 0

        for batch_id, data in enumerate(train_loader):
            input_batch, code_features, attention_mask, target_batch = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device)
            optimizer.zero_grad() # Reset loss gradients from previous batch
            
            ## output
            logits = model(input_ids=input_batch, attention_mask=attention_mask, code_features=code_features)
            loss = torch.nn.functional.cross_entropy(logits, target_batch)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients

            predicted_labels = torch.argmax(logits, dim=-1)
            examples_seen += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
            
            total_loss += loss.item()
        
        num_batches_train = len(train_loader)
        if len(train_loader) == 0:
            return float("nan")
        elif num_batches_train is None:
            num_batches_train = len(train_loader)
        else:
            # Reduce the number of batches to match the total number of batches in the data loader
            # if num_batches exceeds the number of batches in the data loader
            num_batches_train = min(num_batches_train, len(train_loader))
        train_loss = total_loss / num_batches_train
        train_acc, _, _ = calc_accuracy_loader(train_loader, model, device)
        train_acc = train_acc * 100.

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        wandb.log({"train_loss": train_loss, "train_acc": train_acc})
        print('Train #{} Loss: {:.4f} Acc: {:.4f}%'.format(epoch, train_loss, train_acc))

        model.eval() ## set the model to evaluation mode
        with torch.no_grad(): ## disable gradient calculation
            val_loss = calc_loss_loader(val_loader, model, device)
            val_acc,_,_ = calc_accuracy_loader(val_loader, model, device)
            val_acc = val_acc * 100.
        val_losses.append(val_loss)
        val_accs.append(val_acc) 
        print('Val #{} Loss: {:.4f} Acc: {:.4f}%'.format(epoch, val_loss, val_acc)) 
        wandb.log({"val_loss": val_loss, "val_acc": val_acc})  
        

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def balance_dataset(csv_file, target):
    ## use imblearn SMOTE to balance the dataset

    df = pd.read_csv(csv_file)

    X = df.drop(target, axis=1) ## drop the target column
    y = df[target]

    ros = RandomOverSampler(sampling_strategy='auto', random_state=RANDOM_SEED) ## resample all classes but the majority class
    X_sm, y_sm = ros.fit_resample(X, y)

    df_sm = pd.concat([X_sm, y_sm], axis=1)

    ## save the balanced dataset
    df_sm.to_csv('CodeT5P/data/balanced_' + csv_file.split("/")[-1], index=False)

    return 'balanced_train.csv'
