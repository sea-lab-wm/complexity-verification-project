from transformers import T5EncoderModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from rtdl_num_embeddings import LinearReLUEmbeddings

# Define the combined model
class CodeT5Classifier(nn.Module):
    
    def __init__(self, encoder_model_name, input_dim, hidden_dim, num_classes):
        super(CodeT5Classifier, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained(encoder_model_name) # Load the encoder model
        self.fc1 = nn.Linear(input_dim, hidden_dim) ## Fully connected layer from encoder output to hidden layer
        self.fc2 = nn.Linear(hidden_dim, 100) ## Fully connected layer from hidden layer to output layer
        self.fc3 = nn.Linear(100, 50) ## Fully connected layer from hidden layer to output layer
        self.fc4 = nn.Linear(50, num_classes) ## Fully connected layer from hidden layer to output layer
    
    def forward(self, input_ids, code_features=None, attention_mask=None):
        # Pass the inputs through the encoder
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        ## get the last hidden state because it contains the embeddings of the entire sequence  
        sequence_output = encoder_outputs.last_hidden_state[:, -1, :]

        ## concat code_feature and sequence_output
        # concated_embedding = torch.cat((sequence_output, code_features), dim=1)
        concated_embedding = sequence_output
        # Pass the embeddings through the MLP classifier
        x = F.relu(self.fc1(concated_embedding))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.3)
        logits = self.fc4(x)
        return logits